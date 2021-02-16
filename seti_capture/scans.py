import numpy as np
import os
import xarray as xr
from blimpy import Waterfall
from blimpy.io import file_wrapper as fw
import matplotlib.pyplot as plt
from setigen import waterfall_utils, distributions, sample_from_obs, unit_utils
from setigen.funcs import paths, t_profiles, f_profiles, bp_profiles
from setigen.frame import Frame
import astropy.units as u
import math
import random
import pandas as pd

test_file_dir = '../LBand/TIC154089169'

test_files = [f'{test_file_dir}/'+f for f in os.listdir(test_file_dir)]




class Scans():
    """Class to streamline handling of any set of observation scans"""
    #__slots__  = ["Data", "header", "scans"]
    MAX_IMSHOW_POINTS = (4096, 1268)
    def __init__(self, path=None, f_start=None, f_stop=None, max_load=1., in_blimpy=False):
        if isinstance(path, str):
            try:
                self.data = self.xcadence([path+'/'+f for f in os.listdir(path)], f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=in_blimpy)
                self.paths = [path]
                self.header = self.data.attrs
                self.scans = self.data.data_vars
            except:
                self.data = self.h5_to_xarray(path,f_start=f_start, f_stop=f_stop, max_load=1., in_blimpy=in_blimpy)
        elif isinstance(path, list):
            self.data = self.xcadence(path, f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=in_blimpy)

            self.paths = [p for p in path]
            self.header = self.data.attrs
            self.scans = self.data.data_vars
        elif isinstance(path, type(None)):
            self.data = None
            self.paths = []
            self.header = None
            self.scans = None
        elif isinstance(path, Waterfall):
            self.data = self.h5_to_xarray(path, f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=True, waterfall_obj=path)

        else:
            self.data = self.xcadence(path, f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=in_blimpy)
            self.paths = [path]
            self.header = self.data.attrs
            self.scans = self.data.data_vars
    
    def __len__(self):
        return len(self.scans)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.scans[key]
        elif isinstance(key, int):
            return self.scans[f'scan_{key+1}']

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise IndexError(f"Index must be of type 'int'")
        if not isinstance(value, xr.DataArray):
            raise ValueError(f"scan_{key+1} must be of type 'xarray.DataArray'")
        self.scans[f"scan_{key+1}"] = value

    def __iter__(self):
        return iter([self.scans[key] for key in self.scans.keys()])
        

    def waterfall_generator(self, frame=None, fchans=1024, f_begin=1340, f_end=None, f_shift=102400):
        
        if frame is None:
            frame = self[0]
            frame_index = 0
        
        fch1 = self.header['fch1'][frame_index]
        nchans = self.header['nchans'][frame_index]
        df = abs(self.header['foff'][frame_index])
        tchans = self.header['n_ints_in_file'][frame_index]

        if f_end is None or f_end > fch1:
            f_stop = fch1
            f_end = fch1
        else:
            f_stop=f_end

        if f_begin is None or f_begin < fch1 - nchans * df:
            f_start = fch1 - fchans * df
            f_begin = fch1 - nchans *df
        else:
            f_start = f_end - fchans*df

        if f_shift is None:
            f_shift = fchans

        # iterate down frequencies, starting from highest
        print(f"\nStarting at {f_begin}")
        print(f"Ending at {f_end}")
        print(f"Iterating with width = {f_shift*df}")
        while f_start >= f_begin:
            #yield self.frames[frame].data
            yield self.to_blimpy(f_start=f_start, f_stop=f_stop, t_start=0, t_stop=tchans)[0]
            f_start -= f_shift*df
            f_stop -= f_shift*df


    def _num_frames(self, frame=None, fchans=1024, f_begin=1340, f_end=None, f_shift=102400):
        
        if frame is None:
            frame = self[0]
            frame_index=0
        
        fch1 = self.header['fch1'][frame_index]
        nchans = self.header['nchans'][frame_index]
        df = abs(self.header['foff'][frame_index])
        tchans = self.header['n_ints_in_file'][frame_index]

        if f_end is None or f_end > fch1:
            f_stop = fch1
            f_end = fch1
        else:
            f_stop=f_end

        if f_begin is None or f_begin < fch1 - nchans * df:
            f_start = fch1 - fchans * df
            f_begin = fch1 - nchans *df
        else:
            f_start = f_end - fchans*df

        if f_shift is None:
            f_shift = fchans
        i=0
        while f_start >= f_begin:
            i+=1
            f_start -= f_shift*df
            f_stop -= f_shift*df
        return i
    
    def _turbo_runner(self, waterfall_itr, max_drift=5, num_inserted=2, drifts=None, snrs=None, widths=None, snrRatio=None, min_snr=15, default_snr=40, default_width=4, default_drift=0, default_snr_noise=0.5):
        from turbo_seti.find_doppler.find_doppler import FindDoppler
        #from .signal_inject import *
        
        def insert_signal(frame, freq, drift, snr, width):
            print("Inserting signal", freq, drift, snr, width, frame.df, frame.dt, frame.fch1, frame.tchans, frame.fchans)
            tchans = frame.tchans*u.pixel
            dt = frame.dt*u.s

            #fexs = freq*u.MHz - math.copysign(1, drift)*width*u.Hz
            fexs = freq - math.copysign(1, drift)*width*u.Hz
            #fex = (freq*u.mhz + ((drift*u.hz/u.s)*dt*tchans/u.pixel)
            #           + math.copysign(1, drift)*width*u.hz)/u.hz
            fex = (freq + ((drift*u.Hz/u.s)*dt*tchans/u.pixel)
                       + math.copysign(1, drift)*width*u.Hz)/u.Hz

            signal = frame.add_signal(paths.constant_path(f_start=freq,
                                               drift_rate=drift*u.Hz/u.s),
                                               t_profiles.constant_t_profile(level=frame.get_intensity(snr=snr)),
                                               f_profiles.gaussian_f_profile(width=width*u.Hz),
                                               bp_profiles.constant_bp_profile(level=1),
                                               bounding_f_range=(min(fexs/u.Hz,fex), max(fexs/u.Hz,fex)))

        def noise_parameters(frame, near=False, drift_max=2):
            parameters = [0]*2
            
            drift = random.uniform(-1,1)*drift_max
            parameters[1] = drift

            if near:
                parameters[0] = (frame.get_frequency(frame.fchans//2) + random.uniform(-1, 1)*(drift*frame.dt*frame.tchans))#*u.MHz#*u.Hz/u.MHz
            else :
                parameters[0] = frame.get_frequency(int(random.uniform(0, frame.fchans)))#*u.Hz/u.MHz
            return parameters


        #signal_capture = {'num_recovered':[], 'num_inserted':[], 'frequency':[], 'drift_rate':[], 'SNR':[], 'width':[], '}
        signal_capture = {'num_recovered':[], 'num_inserted':[], 'capture_ratio':[], 'inserted_frequency':[], 'frequency_cap':[], 'inserted_drift':[], 'drift_cap':[],'inserted_snr':[], 'snr_cap':[], 'inserted_width':[], 'distance_noise_center':[], 'noise_drift':[], 'noise_snr':[]}
        #signal_capture += [[num_recovered, num_inserted, temp_freq/1e6, drift, snr, width, [data], [noise[0]/1e6, noise[1], snr_noise]]]

        turbo_vals = []
        #signal_capture=[]

        for i, waterfall in enumerate(waterfall_itr):
            if not hasattr(self, 'temp_added_signals'):
                self.temp_added_signals = {}

            temp_frame = Frame(waterfall=waterfall)
            
            temp_freq = temp_frame.get_frequency(temp_frame.fchans//2)*u.MHz#*u.Hz/u.MHz

            noise_recovered=0

            if drifts is None:
                drift=default_drift
            if snrs is None:
                snr=default_snr_noise*default_snr
            if widths is None:
                width=default_width
            else:
                width = widths[i]

            drift_noise = max_drift

            if snrRatio is None:
                snr_noise = default_snr_noise
            width_noise = width

            #insert_signal(temp_frame, temp_freq, drift, snr, width)
            self.add_constant_signal(f_start=temp_freq, drift_rate=drift, snr=snr, width=width, temp=temp_frame)
            noise = [0*u.Hz, 0]
            for _ in range(num_inserted - 1):
                noise = noise_parameters(temp_frame, False, drift_noise)
                #insert_signal(temp_frame, noise[0], noise[1], snr_noise, width_noise)
                self.add_constant_signal(f_start=noise[0], drift_rate=noise[1], snr=snr_noise, width=width_noise, temp=temp_frame)

            # save_frame(temp_frame)
            #if not hasattr(self, 'temp_frames'):
            #    self.temp_frames = []
            #else:
            #    self.temp_frames.append(temp_frame)

            temp_frame.save_fil(filename=f"frame_{i}.fil")

            find_seti_event = FindDoppler(f"frame_{i}.fil", max_drift=max_drift, snr=min_snr)
            find_seti_event.search()

            f = open(f"frame_{i}.dat", 'r')
            try:
                # Maybe just use 'read_dat'  from 'find_events' here?
                data = [dataline.split() for dataline in [line for line in f.readlines() if line[0]!='#']]
                data = [[float(val) for val in dataline] for dataline in data]
            except:
                data = []
            if len(data) == 0:
                data = [[0.0]*12]
                num_recovered=0
            else:
                num_recovered = len(data)

            turbo_vals += [data]
            #signal_capture += [[num_recovered, num_inserted, temp_freq.value/1e6, drift, snr, width, [data], [noise[0].value/1e6, noise[1], snr_noise]]]
            acquired_data = [[num_recovered, num_inserted, temp_freq.value/1e6, drift, snr, width, [data], [noise[0]/1e6, noise[1], snr_noise]]]
            signal_capture['num_recovered'].append(num_recovered)
            signal_capture['num_inserted'].append(num_inserted)
            signal_capture['capture_ratio'].append(num_recovered / num_inserted)
            signal_capture['inserted_frequency'].append(temp_freq.value/1e6)
            signal_capture['frequency_cap'].append(acquired_data[0][6][0][0][1])
            signal_capture['inserted_drift'].append(acquired_data[0][3])
            signal_capture['drift_cap'].append(acquired_data[0][6][0][0][3])
            signal_capture['inserted_snr'].append(acquired_data[0][4])
            signal_capture['snr_cap'].append(acquired_data[0][6][0][0][2])
            signal_capture['inserted_width'].append(acquired_data[0][5])
            signal_capture['distance_noise_center'].append(acquired_data[0][7][0] - acquired_data[0][2])
            signal_capture['noise_drift'].append(acquired_data[0][7][1])
            signal_capture['noise_snr'].append(acquired_data[0][7][2])
            #print(i, signal_capture[i])
            #print("")
            #try:
            #    os.remove(f"frame_{i}.h5")
            #    os.remove(f"frame_{i}.fil")
            #    os.system(f"frame_{i}.dat")
            #    os.remove(f"frame_{i}.log")
            #except:
            #    pass
        return signal_capture



    def inject_recover_signal(self, 
            fchans=1024, 
            f_begin=1340, 
            f_end=None, 
            f_shift=102400, 
            max_drift=5, 
            num_inserted=2, 
            min_snr=15, default_drift=0, default_snr=40, default_width=4, default_snr_noise=0.5):
        """
        turboSETI efficiency tests using signal injection and recovery
        Based off of previous code written by Krishna Bhaattaram

        Outputs an xarray Dataset
        """
        waterfall_itr = self.waterfall_generator(frame=None, fchans=fchans, f_begin=f_begin, f_end=f_end, f_shift=f_shift)
        N = self._num_frames(frame=None, fchans=fchans, f_begin=f_begin, f_end=f_end, f_shift=f_shift)
        turbo_vals = self._turbo_runner(waterfall_itr, max_drift=max_drift, num_inserted=num_inserted, min_snr=min_snr, default_drift=default_drift, default_snr=default_snr, default_width=default_width, default_snr_noise=default_snr_noise)
        df = pd.DataFrame.from_dict(turbo_vals)
        df.to_csv(f"signal_injection_recovery_fstart_{f_begin}_driftmax_{max_drift}_width_{default_width}.csv", index=False, header=df.columns)

        try:
            os.system("rm *h5")
            os.system("rm *fil")
            os.system("rm *.dat")
            os.system("rm *.log")
        except:
            pass

        return df.to_xarray()

    def recover_added_signals(self):
        for frame, signals in self.added_signals.items():
            if len(signals) == 0:
                continue
            #frame.save_fil(filename=f"frame_{}")

    def add_frames(self):
        """Creates a setigen Frame object
        DISCLAIMER: ONLY TESTED FOR ONE DataArray OBJ (i.e., only works for one scan as a 
        DataArray)"""
        
        self.frames = [Frame(blimpy_waterfall) for blimpy_waterfall in self.to_blimpy()]
        self.data.attrs['frames'] = [fr for fr in self.frames]
        self.added_signals = {key:[] for key in self.data.attrs['frames']}

    def add_variable_signal(self,
            sig_path='const', 
            t_profile='const', 
            f_profile='gauss', 
            bp_profile='const', 
            bounding_f_range=None,
            integrate_path=False,
            integrate_t_profile=False,
            integrate_f_profile=False,
            t_subsamples=10,
            f_subsamples=10,
            all_frames=False):
        """Wraps setigen 'add_signal' method to add a synthetic signal given parameters"""
        if not hasattr(self, 'frames'):
            self.add_frames()
        if not add_all:
            frame = self.frames[0]
            if sig_path == 'const':
                sig_path = paths.constant_path(f_start=frame.get_frequency(200), drift_rate=2*u.Hz/u.s)
            if t_profile == 'const':
                t_profile = t_profiles.constant_t_profile(level=frame.get_intensity(snr=30))
            if f_profile == 'gauss':
                f_profile = f_profiles.gaussian_f_profile(width=4*u.Hz)
            if bp_profile == 'const':
                bp_profile = bp_profiles.constant_bp_profile(level=1)

            signal = frame.add_signal(sig_path, t_profile, f_profile, bp_profile)
            
            self.added_signals[frame].append(signal)

    def add_constant_signal(self,
            f_start=None,
            drift_rate=4,
            snr=40,
            width=4,
            f_profile_type='gaussian',
            all_frames=False,
            temp=None):
        """Wraps 'add_variable_signal' to create a constant signal.
        Saves signal data with metadata as an xarray DataArray"""
            #signal = frame.add_constant_signal(f_
        if f_start is None:
            f_start = self.scans['scan_1'].freq.values[0]


        if not hasattr(self, 'frames'):
            self.add_frames()

        if not all_frames:
            frame = self.frames[0]
            if temp is not None:
                frame = temp
            signal = frame.add_constant_signal(f_start, drift_rate, snr, width, f_profile_type=f_profile_type)
            if temp is not None:
                freq_coord = frame.fs
                time_coord = frame.ts
            else:
                freq_coord = self.scans['scan_1'].freq.values[:-1]
                time_coord = self.scans['scan_1']['time_1'].values
            signal_DataArray = xr.DataArray(
                    #np.asarray(signal),
                    frame.get_data(),
                    # had to use index [:-1] to get the length of the data to match the length of 
                    #   frequency values.  They had the same start frequency value, so I chopped
                    #   off the last frequency...hopefully this doesn't cause many issues
                    coords=[time_coord, freq_coord],
                    dims = ['time', 'freq'],
                    attrs={'f_start':f_start,
                        'drift_rate':drift_rate,
                        'snr':snr,
                        'width':width,
                        'f_profile_type':f_profile_type,
                        'constant_signal':True
                        }
                    )

            if temp is not None:
                self.temp_added_signals[temp] = signal_DataArray

            else:
                self.added_signals[frame].append(signal_DataArray)

    def render_added_signals(self, all_signals=False, zoom=False):
        """Plots synthetic signals added by setigen
        By default, plots first signal listed in 'self.added_signals' and assumes
        no other added signals exist
        """
        if not all_signals:
            frame_to_plot = None
            frame_index = None
            for i, key in enumerate(self.added_signals.keys()):
                if len(self.added_signals[key])!=0:
                    frame_to_plot = key
                    frame_index = i
                    break
            if isinstance(frame_to_plot, Frame):
                fig = plt.figure(figsize=(10,6))
                if not zoom:
                    frame_to_plot.bl_render()
                    plt.savefig(f"{self.header['source_name'][frame_index]}_rendering.png", bbox_inches='tight')
                    try:
                        plt.show()
                    except:
                        raise Warning(f"'{self.source[frame_index]}_rendering.png' saved but failed to display")
                    plt.clf()
                else:
                    lower_limit = self.added_signals[frame_to_plot][0].attrs['f_start']
                    plt.xlim(lower_limit, lower_limit+10)#if self.added_signals[frame_to_plot[0].attrs['constant_signal']:
                        #upper_limit = lower_limit + 
                    #plt.xlim()
                    plt.savefig(f"{self.header['source_name'][frame_index]}_rendering.png", bbox_inches='tight')
                    try:
                        plt.show()
                    except:
                        raise Warning(f"'{self.source[frame_index]}_rendering.png' saved but failed to display")
                    plt.clf()


    def h5_to_xarray(self, path, out='arr',f_start=None, f_stop=None, max_load=1., out_dir=None, in_blimpy=False, waterfall_obj=None):
        """Converts h5 file to an xarray DataArray"""
        if in_blimpy and waterfall_obj is not None:
            container = waterfall_obj.container
            #container = path.container
        else:
            container = fw.open_file(path, f_start=f_start, f_stop=f_stop, max_load=max_load)
        snr_data = np.asarray([array for array in [data[0] for data in container.data]])
        #snr_data = np.asarray([data[0] for data in container.data])
        timestamps = container.populate_timestamps()
        frequencies= container.populate_freqs()
        header = container.header
        if not in_blimpy:
            del header['DIMENSION_LABELS']
        header['n_ints_in_file'] = container.selection_shape[0]
        dataset = xr.Dataset(
                data_vars=dict(
                    snr=(['time', 'freq'], snr_data),
                    ),
                coords = dict(
                    frequency=(['freq'], frequencies),
                    time=timestamps,
                    ),
                attrs = header
                )
        if out_dir is not None:
            dataset.to_netcdf(out_dir, format='NETCDF4')

        data_array = xr.DataArray(
                #data=dict(
                #    snr=(['time','freq'], snr_data),
                #    ),
                snr_data,
                coords=[timestamps, frequencies],
                dims=['time', 'freq'],
                attrs=header
                )
                #coords=dict(
                #    frequency=(['freq'], frequencies),
                #    time=timestamps,
                #    ),
                #attrs=header
                #)
        if out=='dataset':
            return dataset
        else:
            return data_array

    def xcadence(self,path_list, f_start=None, f_stop=None, max_load=1., in_blimpy=False):
        data_arrays = [self.h5_to_xarray(path, f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=in_blimpy) for path in path_list]
        #bool_list = []
        #for array, i in enumerate(data_arrays):
        #    if i == int(len(data_arrays) - 1):
        #        continue
        #    bool_list.append(array.freq == data_arrays[i+1].freq)
        
        def _generate_coords_dict(arrays_list):
            out_dict = {'freq':(['freq'], arrays_list[0].freq.values)}
            for k in range(len(arrays_list)):
                out_dict[f'time_{k+1}'] = ([f'time_{k+1}'], arrays_list[k].time.values)
            return out_dict
        
        #if not sum(bool_list):
        #    pass
            #ds = xr.Dataset(
            #        data_vars={f'scan_{i+1}':([f'time_{i+1}', f'freq{i+1}'])}
            #            )    
        #else:
        ds = xr.Dataset(
                    data_vars=dict(
                        {f'scan_{i+1}':([f'time_{i+1}', 'freq'], snr_data.data) for i, snr_data in enumerate(data_arrays)}
                        ),
                    coords=_generate_coords_dict(data_arrays),
                    #dict(
                        #{f'time{i+1}':([f'time_{i+1}'], time_array.time) for time_array, i in enumerate(data_arrays) if i!=0 else f'frequency':(['freq'], data_arrays[0].freq)}
                        #frequency=(['freq'], data_arrays[0].freq), 
                        #time1=(['time1'],arr1.time), 
                        #time2=(['time2'], arr2.time), 
                        #time3=(['time3'], arr3.time),
                    attrs={key:[array.attrs[key] for array in data_arrays] for key in data_arrays[0].attrs.keys()})
        return ds

    def rebin(d, n_x=None, n_y=None, n_z=None):
        """ Rebin data by averaging bins together
        Args:
        d (np.array): data
        n_x (int): number of bins in x dir to rebin into one
        n_y (int): number of bins in y dir to rebin into one
        Returns:
        d: rebinned data with shape (n_x, n_y)
        """
        n_x = 1 if n_x is None else n_x
        n_y = 1 if n_y is None else n_y
        n_z = 1 if n_z is None else n_z

        if d.ndim == 3:
            d = d[:int(d.shape[0] // n_x) * n_x, :int(d.shape[1] // n_y) * n_y, :int(d.shape[2] // n_z) * n_z]
            d = d.reshape((d.shape[0] // n_x, n_x, d.shape[1] // n_y, n_y, d.shape[2] // n_z, n_z))
            d = d.mean(axis=5)
            d = d.mean(axis=3)
            d = d.mean(axis=1)
        elif d.ndim == 2:
            d = d[:int(d.shape[0] // n_x) * n_x, :int(d.shape[1] // n_y) * n_y]
            d = d.reshape((d.shape[0] // n_x, n_x, d.shape[1] // n_y, n_y))
            d = d.mean(axis=3)
            d = d.mean(axis=1)
        elif d.ndim == 1:
            d = d[:int(d.shape[0] // n_x) * n_x]
            d = d.reshape((d.shape[0] // n_x, n_x))
            d = d.mean(axis=1)
        else:
            raise RuntimeError("Only NDIM <= 3 supported")
        return d
    def __repr__(self):
        if self.data is None:
            return f"Empty Scans object"
        return f"Scans of {self.data.attrs['source_name']}"# over {self.data.freq.values[0]} to {self.data.freq.values[-1]}"
    def to_blimpy(self, f_start=None, f_stop=None, t_start=None, t_stop=None, max_load=1.):
        """Returns list of scans as Waterfall objects"""
        if f_start is None:
            f_start = self.scans['scan_1'].freq.values[0]
        if f_stop is None:
            f_stop = self.scans['scan_1'].freq.values[-1]
        return [Waterfall(p, f_start=f_start, f_stop=f_stop, max_load=max_load, t_start=t_start, t_stop=t_stop) for p in self.paths]
    def from_blimpy(self, path, f_start=None, f_stop=None, max_load=1.):
        return Scans(path, f_start=f_start, f_stop=f_stop, max_load=max_load, in_blimpy=True)


    def plot_waterfall(self, scan, f_start=None, f_stop=None, **kwargs):
        """Plot waterfall of data from scan"""
        #print(type(scan))
        plot_f, plot_data = self.grab_data(scan, f_start=f_start, f_stop=f_stop)
        dec_fac_x, dec_fac_y = 1, 1
        
        #print(self.grab_data(scan, f_start=f_start, f_stop=f_stop))
        #print(plot_data.shape)

        if plot_data.shape[0] > self.MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data.shape[0] / self.MAX_IMSHOW_POINTS[0]
        if plot_data.shape[1] > self.MAX_IMSHOW_POINTS[1]:
            dec_fac_y = int(np.ceil(plot_data.shape[1] / self.MAX_IMSHOW_POINTS[1]))
        plot_data = rebin(plot_data, dec_fac_x, dec_fac_y)
        
        extent = (plot_f[0], plot_f[-1], (scan.timestamps[-1]-scan.timestamps[0])*24.*60.*60,0.0)
        kwargs['cmap']=kwargs.get('cmap', 'viridis')
        plot_data = 10.0*np.log10(plot_data)

        vmin = plot_data.min()
        vmax = plot_data.max()
        normalized_plot_data = (plot_data - vmin) / (vmax - vmin)
        
        this_plot = plt.imshow(normalized_plot_data, aspect='auto', rasterized=True, interpolation='nearest', extent=extent, **kwargs)
        return this_plot
    
    def plot_cadence(self, f_start=None, f_stop=None, t_start=None, t_stop=None):
        plt.close(fig='all')
        n_plots = len(self.scans)
        fig = plt.subplots(n_plots, sharex=True, sharey=True, figsize=(10,2*n_plots))
        t0 = self.scans['scan_1']
        dummy, plot_data1 = self.grab_data(1, f_start=f_start, f_stop=f_stop)

        dec_fac_x, dec_fac_y = 1, 1
        if plot_data1.shape[0] > self.MAX_IMSHOW_POINTS[0]:
            dec_fac_x = plot_data1.shape[0] / self.MAX_IMSHOW_POINTS[0]
        if plot_data1.shape[1] > self.MAX_IMSHOW_POINTS[1]:
            dec_fac_y = int(np.ceil(plot_data1.shape[1] / self.MAX_IMSHOW_POINTS[1]))
        plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)
        
        if f_start is None:
            f_start = self.scans['scan_1'].freq.values[0]
        if f_stop is None:
            f_stop = self.scans['scan_1'].freq.values[0]
        mid_f = np.abs(f_start+f_stop)/2

        subplots = []

        for i, scan in enumerate(self.scans): 
            #plot_data = plot_data.astype('float32')
            subplot = plt.subplot(n_plots, 1, i+1)
            subplots.append(subplot)

            this_plot = self.plot_waterfall(self.scans[scan], f_start=f_start, f_stop=f_stop)
            #if self.header['foff']
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig("cadence_xseti.png", bbox_inches='tight')
        return subplots

    def grab_data(self, scan, f_start=None, f_stop=None, t_start=None, t_stop=None):
        if isinstance(scan, str):
            raise Warning("Requested 'scan' is of type 'str', defaulting to 'scan_1'")
            scan = self.scans['scan_1']
        elif isinstance(scan, int):
            scan = self.scans[f'scan_{scan}']
        if f_start is None:
            f_start = scan.freq.values[0]
        if f_stop is None:
            f_stop = scan.freq.values[-1]
        i0 = np.argmin(np.abs(scan.freq.values - f_start))
        i1 = np.argmin(np.abs(scan.freq.values - f_stop))
        if i0 < i1:
            plot_f = scan.freq.values[i0:i1 + 1]
            plot_data = np.squeeze(scan.data[t_start:t_stop, ..., i0:i1 + 1])
        else:                    
            plot_f = scan.freq.values[i0:i1 + 1]
            plot_data = np.squeeze(scan.data[t_start:t_stop, ..., i1:i0 + 1])
        #try:
        #    i0 = np.argmin(np.abs(scan.freq.values - f_start))
        #    i1 = np.argmin(np.abs(scan.freq.values - f_stop))
        #    if i0 < i1:
        #        plot_f = scan.freq.values[i0:i1 + 1]
        #        plot_data = np.squeeze(scan.data[t_start:t_stop, ..., i0:i1 + 1])
        #    else:                    
        #        plot_f = scan.freq.values[i0:i1 + 1]
        #        plot_data = np.squeeze(scan.data[t_start:t_stop, ..., i1:i0 + 1])
        #except:
        #    raise Exception("Too much data requested")
        #print(plot_f, plot_data)
        print(plot_data.shape)
        return plot_f, plot_data


def test_xcadence(test_h5_dir='../LBand/TIC154089169', max_load=20., f_start=None, f_stop=1025):
    files = [f'{test_h5_dir}/'+f for f in os.listdir(test_h5_dir)]
    del files[2]
    dataset = xcadence(files, f_start=f_start, f_stop=f_stop, max_load=max_load)
    print(dataset)
    return dataset

        

def test_h5_to_xarray(test_h5_dir='../LBand/TIC154089169', max_load=20., f_start=None, f_stop=1025):
    for FILE in [f'{test_h5_dir}/'+f for f in os.listdir(test_h5_dir)]:
        try:
            dataset = h5_to_xarray(FILE, f_start=f_start, f_stop=f_stop, max_load=max_load)
            print(dataset)
        except:
            continue

def test_Scans(test_h5_dir='../LBand/TIC154089169', max_load=20., f_start=None, f_stop=1025):
    files = [f'{test_h5_dir}/'+f for f in os.listdir(test_h5_dir)]
    del files[2]
    S = Scans(files, f_start=f_start, f_stop=f_stop, max_load=max_load)
    #print(dataset)
    print(S)
    return S

if __name__=='__main__':
    #test(max_load=40, f_stop=1034)
    test_xcadence()
