# Make it easier to issue calls without asynchronous commands
# import nest_asyncio
# nest_asyncio.apply()

import sys
sys.path.append('/home/daq1/github/rfmux/')

import os
import rfmux
import shutil
import warnings
import subprocess
import numpy as np
from time import sleep
from tqdm.auto import tqdm
from .util import volts_to_dbm, remove_internal_phaseshift, convert_parser_to_z
from .util import volts_per_roc
from hidfmux.core.utils.transferfunctions import apply_pfb_correction
from hidfmux.core.utils import transferfunctions
from hidfmux.analysis.noise_processing import separate_iq_fft_to_i_and_q


class CRS:
    def __init__(self, crs_sn = 27): 
        """ srs object d. Not that the system must be 
        configured using CRS.configure_system before measurements 

        Parameters:
        crs_sn (int): CRS serial number 
        """ 
        self.crs_sn = crs_sn
        s = rfmux.load_session('!HardwareMap [ !CRS { ' + f'serial: "{crs_sn:04d}"' + ' } ]')
        self.d = s.query(rfmux.CRS).one()
        self.volts_per_roc = volts_per_roc
        
    async def configure_system(self, clock_source="SMA", full_scale_dbm = 1, analog_bank_high = False,
                               verbose = True, nyquist_zones = {1: 1, 2: 1, 3: 1, 4: 1}):
        """
        Resolves the system, sets the timestamp port, sets the clock source, and sets the full scale
        in dBm 
        
        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal voltage controlled
            crystal oscillator or 'SMA' for the external 10 MHz reference (5 Vpp I think) 
        full_scale_dbm (int): full scale power in dBm. range is [???, 7?]
        verbose (bool): If True, gets and prints the clocking source 
        nyquist_zones (dict): keys (int) are module indicies and values (int) are nyquist zone numbers (1 or 2)
        """
        await self.d.resolve()

        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted: 
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)  
        
        await self.d.set_clock_source(clock_source)
        await self.d.set_analog_bank(high = analog_bank_high)
        self.analog_bank_high = analog_bank_high
        for module_index in range(1, 5):
            await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module_index)
            m = module_index
            if self.analog_bank_high:
                m += 4
            await self.d.set_dac_scale(full_scale_dbm, self.d.UNITS.DBM, m)
        self.full_scale_dbm = full_scale_dbm
        
        sleep(1)
        if verbose:
            print('System configured')
            print("Clocking source is", await self.d.get_clock_source())

        await self.d.set_analog_bank(high = analog_bank_high)

        self.set_nyquist_zone(nyquist_zones)

    async def set_nyquist_zone(self, nyquist_zones):
        if any([nz not in [1, 2] for nz in nyquist_zones.values()]):
            raise ValueError('Nyquist zone must be in [1, 2]')
        for module_index, zone in nyquist_zones.items():
            self.d.set_nyquist_zone(zone, module = module_index) 
        self.nyquist_zones = nyquist_zones 

    async def set_sequential_ncos(self, ncos):
        self.ncos = ncos

    async def set_module_index(self, module_index):
        self.module_index = module_index
        
    async def set_nco(self, module_index, nco_freq, verbose = True):
        """Set the NCO frequency
        
        Parameters:
        module_index (int): module index
        nco_freq (float): NCO frequency in Hz. This should not be a round number
        verbose (bool): If True, gets and prints the NCO frequency
        """
        if self.analog_bank_high:
            module_index += 4
        await self.d.set_nco_frequency(nco_freq, self.d.UNITS.HZ, module = module_index)
        self.nco_freq = await self.d.get_nco_frequency(self.d.UNITS.HZ, module=module_index)
        if verbose:
            print(f'NCO is {self.nco_freq}') 

        self.theta_offset = 0
        f, z = await self.sweep([[nco_freq - 150e6], [nco_freq], [nco_freq + 150e6]], 
                                [-55, -55, -55], nsamps = 1000, verbose = False)
        self.theta_offset = np.mean(np.angle(z))
        
            
    async def write_tones(self, fres, ares):
        """
        Writes an array of tones given frequencies and amplitudes 

        Parameters:
        fres (array-like): list of frequencies in Hz 
        ares (array-like): list of powers in dBm (+- 2 dBm precision for now) 
        """
        try:
            nco = self.nco_freq
        except:
            raise Exception('NCO frequency has not been set')
        fres = np.asarray(fres) 
        ares = np.asarray(ares)
        # Randomize frequencies a little
        fres += np.random.uniform(-50, 50, fres.shape)
        comb_sampling_freq = transferfunctions.get_comb_sampling_freq()
        threshold = 101.
        fres[fres%(comb_sampling_freq/512) < threshold] += threshold
        fres_nyq = convert_freq_to_nyq(fres, self.nyquist_zones[self.module_index], 
                                              adc_sampling_rate=5e9)

        if any(ares > self.full_scale_dbm):
            raise ValueError(f'ares must not exceed {self.full_scale_dbm} dBm: raise full_scale_dbm or lower powers')
        if any(ares < -60) and len(ares) < 100:
            warnings.warn(f"values in ares are < 60 dBm: digitization noise may occur", UserWarning)
        ares_amplitude = 10 ** ((ares - self.full_scale_dbm) / 20)
        
        await self.d.clear_channels(module = self.module_index)
        
        nco = await self.d.get_nco_frequency(self.d.UNITS.HZ, module=self.module_index)
        async with self.d.tuber_context() as ctx:
            for ch, (fr, ar) in enumerate(zip(fres_nyq,ares_amplitude)):
                ## To be wrapped in context_manager
                ctx.set_frequency(fr-nco, self.d.UNITS.HZ, ch+1, module=self.module_index)
                ctx.set_amplitude(ar, self.d.UNITS.NORMALIZED, target=self.d.TARGET.DAC, channel=ch+1, module=self.module_index)
            await ctx()

    async def sweep_sequential(self,  ncos, frequencies, ares, nsamps = 10, verbose = True, 
                               pbar_description = 'Sweeping'):
        frequencies, ares = np.asarray(frequencies).copy(), np.asarray(ares).copy()
        self.nco_freq_dict = {key: nco for key, nco in enumerate(ncos)}
        # Split frequencies and ares into dictionaries 
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set")  

        channel_indices = list(range(len(frequencies))) 
        self.frequencies_dict = {key: [] for key in range(len(ncos))}
        self.ares_dict = {key: [] for key in range(len(ncos))}
        self.ch_ix_dict = {key: [] for key in range(len(ncos))}
        for ch_ix, freqs, ar in zip(channel_indices, frequencies, ares):
            # nco_index = min(self.nco_freq_dict, key = lambda k: max([np.abs(self.nco_freq_dict[k] - fr) for fr in [max(freqs), min(freqs)]])) 
            nco_index = min(self.nco_freq_dict, key = lambda k: np.abs(self.nco_freq_dict[k] - np.mean(freqs))) 
            self.frequencies_dict[nco_index].append(freqs) 
            self.ares_dict[nco_index].append(ar) 
            self.ch_ix_dict[nco_index].append(ch_ix)
        self.frequencies_dict = {key: np.array(value).copy() for key, value in self.frequencies_dict.items()}
        self.ares_dict = {key: np.array(value).copy() for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.array(value).copy() for key, value in self.ch_ix_dict.items()}
        for nco_index in self.nco_freq_dict.keys():
            if any(np.abs(self.frequencies_dict[nco_index] - self.nco_freq_dict[nco_index]).flatten() > 300e6):
                raise ValueError('All of frequencies must be within 300 MHz of an NCO frequency') 
        
        f, z = np.empty(frequencies.shape, dtype = float), np.empty(frequencies.shape, dtype = complex)
        for nco_index, nco_freq in self.nco_freq_dict.items():
            await self.set_nco(self.module_index, nco_freq, verbose = False)
            sleep(1)
            desc = pbar_description + f' NCO {nco_index + 1} / {len(self.nco_freq_dict)}'
            fi, zi = await self.sweep(self.frequencies_dict[nco_index], 
                             self.ares_dict[nco_index], nsamps = nsamps, 
                            verbose = verbose, pbar_description = desc,
                            return_raw = False)
            for ch_ix, fii, zii in zip(self.ch_ix_dict[nco_index], fi, zi):
                f[ch_ix] = fii 
                z[ch_ix] = zii
        return f, z
    
    # async def sweep_sequential(self,  ncos, frequencies, ares, nsamps = 10, verbose = True, 
    #                            pbar_description = 'Sweeping'):
    #     frequencies, ares = np.asarray(frequencies).copy(), np.asarray(ares).copy()
    #     self.nco_freq_dict = {key: nco for key, nco in enumerate(ncos)}
    #     # Split frequencies and ares into dictionaries 
    #     if not len(self.nco_freq_dict):
    #         raise Exception("NCO frequencies are not set")  

    #     channel_indices = list(range(len(frequencies))) 
    #     self.frequencies_dict = {key: [] for key in range(len(ncos))}
    #     self.ares_dict = {key: [] for key in range(len(ncos))}
    #     self.ch_ix_dict = {key: [] for key in range(len(ncos))}
    #     for ch_ix, freqs, ar in zip(channel_indices, frequencies, ares):
    #         # nco_index = min(self.nco_freq_dict, key = lambda k: max([np.abs(self.nco_freq_dict[k] - fr) for fr in [max(freqs), min(freqs)]])) 
    #         nco_index = min(self.nco_freq_dict, key = lambda k: np.abs(self.nco_freq_dict[k] - np.mean(freqs))) 
    #         self.frequencies_dict[nco_index].append(freqs) 
    #         self.ares_dict[nco_index].append(ar) 
    #         self.ch_ix_dict[nco_index].append(ch_ix)
    #     self.frequencies_dict = {key: np.array(value).copy() for key, value in self.frequencies_dict.items()}
    #     self.ares_dict = {key: np.array(value).copy() for key, value in self.ares_dict.items()}
    #     self.ch_ix_dict = {key: np.array(value).copy() for key, value in self.ch_ix_dict.items()}
    #     for nco_index in self.nco_freq_dict.keys():
    #         if any(np.abs(self.frequencies_dict[nco_index] - self.nco_freq_dict[nco_index]).flatten() > 300e6):
    #             raise ValueError('All of frequencies must be within 300 MHz of an NCO frequency') 
        
    #     f, z = np.empty(frequencies.shape, dtype = float), np.empty(frequencies.shape, dtype = complex)
    #     nco_index = 0 
    #     nco_freq = ncos[0]
    #     await self.set_nco(self.module_index, nco_freq, verbose = False)
    #     sleep(1)
    #     desc = pbar_description + f' NCO {nco_index + 1} / {len(self.nco_freq_dict)}'
    #     fi, zi = await self.sweep(frequencies, 
    #                         ares, nsamps = nsamps, 
    #                     verbose = verbose, pbar_description = desc,
    #                     return_raw = False)
    #     f, z = fi, zi
    #     for nco_index, nco_freq in self.nco_freq_dict.items():
    #         await self.set_nco(self.module_index, nco_freq, verbose = False)
    #         sleep(1)
    #         desc = pbar_description + f' NCO {nco_index + 1} / {len(self.nco_freq_dict)}'
    #         fi, zi = await self.sweep(self.frequencies_dict[nco_index], 
    #                          self.ares_dict[nco_index], nsamps = nsamps, 
    #                         verbose = verbose, pbar_description = desc,
    #                         return_raw = False)
    #         for ch_ix, fii, zii in zip(self.ch_ix_dict[nco_index], fi, zi):
    #             f[ch_ix] = fii 
    #             z[ch_ix] = zii
    #     return f, z

    async def sweep(self, frequencies, ares, nsamps = 10, verbose = True, pbar_description = 'Sweeping',
                    return_raw = False, nstd_thresh = 1):
        """
        Performs a frequency sweep and returns the complex S21 value at each frequency. Performs sweeps over 
        axis 0 of frequencies simultaneously 

        Parameters:
        frequencies (M X N array-like float): the first index M is the channel index (max len 1024) 
            and the second index N is the frequency in Hz for a single point in the sweep
        ares (M array-like float): list of amplitudes in dBm for each channel 
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        z (M X N array-like complex): complex S21 data in V for each frequency in f 

        """
        frequencies, ares = np.asarray(frequencies).copy(), np.asarray(ares).copy()
        # Randomize frequencies a little
        frequencies += np.random.uniform(-50, 50, frequencies.shape)
        comb_sampling_freq = transferfunctions.get_comb_sampling_freq()
        threshold = 101.
        frequencies[frequencies%(comb_sampling_freq/512) < threshold] += threshold
        frequencies_nyq = convert_freq_to_nyq(frequencies, self.nyquist_zones[self.module_index], 
                                              adc_sampling_rate=5e9)

        n_channels, n_points = frequencies.shape
        if len(ares) != n_channels:
            raise ValueError('ares and frequencies are not the same length')
        # Set the fir stage 
        fir_stage = 6 
        await self.d.set_fir_stage(fir_stage) 
        sample_frequency = 625e6 / (256 * 64 * 2**fir_stage)

        # Write amplitudes 
        fres = [fi[0] for fi in frequencies]
        await self.write_tones(fres, ares)
        # Initialize z array 
        z = np.empty((n_channels, n_points), dtype = complex)
        zcal = np.empty((n_channels, n_points), dtype = complex) 
        zraw = np.empty((n_channels, n_points), dtype = complex)

        pbar = range(n_points) 
        if verbose:
            pbar = tqdm(pbar, total = n_points, leave = False)
            pbar.set_description(pbar_description)
        
        for sweep_index in pbar:
            # Write frequencies 
            async with self.d.tuber_context() as ctx:
                for ch in range(n_channels):
                    f = frequencies_nyq[ch, sweep_index]
                    ctx.set_frequency(f - self.nco_freq, self.d.UNITS.HZ, ch + 1, module = self.module_index)
                await ctx()

            # take data and loopback calibration data
            await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, self.module_index)

            samples_cal = await self.d.py_get_samples(20,module=self.module_index)
            await self.d.set_dmfd_routing(self.d.ROUTING.ADC, self.module_index)
            samples = await self.d.py_get_samples(nsamps,module=self.module_index)
            # format and average data 
            zi = np.asarray(samples.i) + 1j * np.asarray(samples.q)
            # Remove cosmic rays 
            for di in range(len(zi)):
                zii = zi[di] 
                ix = list(range(len(zii)))
                nstd_thresh0 = nstd_thresh
                while not (len(zii) - len(ix)) > nsamps / 10:
                    zii_abs = np.abs(zii - np.mean(zii))
                    ix = np.where(zii_abs > np.std(zii_abs) * nstd_thresh0)[0]
                    nstd_thresh0 += 0.1
                zi[di][ix] = np.nan
            #average 
            zi = np.nanmean(zi[:n_channels, :] , axis = 1) 
            zical = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
            zical = np.mean(zical[:n_channels, :], axis = 1)
            # adjust for loopback calibration
            zcal[:, sweep_index] = zical 
            zraw[:, sweep_index] = zi * self.volts_per_roc 
            zi = remove_internal_phaseshift(frequencies[:, sweep_index], zi, zical) 
            z[:, sweep_index] = zi * self.volts_per_roc 
        # Turn off channels 
        await self.d.clear_channels(module = self.module_index)
        z /= 10 ** (ares[:, np.newaxis] / 20)
        z *= np.exp(1j * self.theta_offset)
        if return_raw:
            return frequencies, z, zcal, zraw 
        return frequencies, z
    
    async def sweep_linear(self, fres, ares, bw = 20e3, npoints = 10, 
                           nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where each channel is swept over the same range in 
        frequencies

        Parameters:
        fres (array-like): center frequencies in Hz 
        ares (array-like): amplitudes in dBm
        bw (float): span around each frequency to sweep in Hz 
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index and 
            N is the index of each point in the sweep 
        z (M X N np.array): array of complex S21 data corresponding to f 
        """ 
        fres, ares = np.asarray(fres), np.asarray(ares)
        f = np.linspace(fres - bw / 2, fres + bw / 2, npoints).T
        f, z = await self.sweep_sequential(self.ncos, f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
        return f, z
    
    async def sweep_qres(self, fres, ares, qres, npoints = 10, nsamps = 10,
                         verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where the span around each frequency is set 
        equal to fres / qres 

        Parameters:
        fres (array-like): center frequencies in Hz 
        ares (array-like): amplitudes in dBm
        qres (array-like): spans of each sweep given as a quality-factor-like 
            value: spans = fres / qres 
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        f (M X N np.array): array of frequencies where M is the channel index and 
            N is the index of each point in the sweep 
        z (M X N np.array): array of complex S21 data corresponding to f 
        """ 
        fres, qres = np.asarray(fres), np.asarray(qres)
        spans = fres / qres 
        f = np.linspace(fres - spans / 2, fres + spans / 2, npoints).T
        f, z = await self.sweep_sequential(self.ncos, f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
        return f, z
    
    async def sweep_full(self, amplitude, npoints = 10, 
                         nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep over the full 600 MHz bandwidth around the NCO 
        frequency

        Parameters:
        amplitude (float): amplitude in dBm
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        f (np.array): array of frequencies in Hz 
        z (np.array): array of complex S21 data corresponding to f 
        """  
        bw = 600e6 / 1024 + 200
        fres = np.linspace(np.min(self.ncos) - 300e6 + bw / 2, np.max(self.ncos) + 300e6 - bw / 2, 1024) 
        ares = amplitude * np.ones(len(fres))
        f, z = await self.sweep_linear(fres, ares, bw = bw, npoints = npoints,
                                      nsamps = nsamps, verbose = verbose, 
                                      pbar_description = pbar_description)
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_noise(self, fres, ares, noise_time, fir_stage = 6,
                            parser_loc='/home/daq1/github/citkid/citkid/crs/parser',
                            interface='enp2s0', delete_parser_data = False,
                            verbose = True):
        fres, ares = np.asarray(fres).copy(), np.asarray(ares).copy()
        self.nco_freq_dict = {key: nco for key, nco in enumerate(self.ncos)}
        # Split frequencies and ares into dictionaries 
        if not len(self.nco_freq_dict):
            raise Exception("NCO frequencies are not set")  

        channel_indices = list(range(len(fres))) 
        self.fres_dict = {key: [] for key in range(len(self.ncos))}
        self.ares_dict = {key: [] for key in range(len(self.ncos))}
        self.ch_ix_dict = {key: [] for key in range(len(self.ncos))}
        for ch_ix, fr, ar in zip(channel_indices, fres, ares):
            nco_index = min(self.nco_freq_dict, key = lambda k: np.abs(self.nco_freq_dict[k] - fr))
            self.fres_dict[nco_index].append(fr) 
            self.ares_dict[nco_index].append(ar) 
            self.ch_ix_dict[nco_index].append(ch_ix)
        self.fres_dict = {key: np.asarray(value) for key, value in self.fres_dict.items()}
        self.ares_dict = {key: np.asarray(value) for key, value in self.ares_dict.items()}
        self.ch_ix_dict = {key: np.asarray(value) for key, value in self.ch_ix_dict.items()}
        for nco_index in self.nco_freq_dict.keys():
            if any(np.abs(self.fres_dict[nco_index] - self.nco_freq_dict[nco_index]).flatten() > 300e6):
                raise ValueError('All of frequencies must be within 300 MHz of an NCO frequency') 
        
        z = None
        for nco_index, nco_freq in self.nco_freq_dict.items():
            await self.set_nco(self.module_index, nco_freq, verbose = False)
            sleep(1) 
            zi = await self.capture_noise_single(self.fres_dict[nco_index], self.ares_dict[nco_index], 
                                                 noise_time, fir_stage = fir_stage,
                               parser_loc=parser_loc, interface=interface,
                               delete_parser_data = delete_parser_data,
                               verbose = verbose, return_raw = False)
            if z is None:
                z = np.empty((len(fres), len(zi[0])), dtype = complex)
            # Make sure z arrays are all the same size
            len_diff = len(zi[0]) - len(z[0])
            if len_diff > 0:
                zi = zi[:, :-len_diff]
            if len_diff < 0:
                z = z[:, :len_diff]
            for ch_ix, zii in zip(self.ch_ix_dict[nco_index], zi):
                z[ch_ix] = zii
        z *= np.exp(1j * self.theta_offset)
        return z
        

    async def capture_noise_single(self, fres, ares, noise_time, fir_stage = 6,
                            parser_loc='/home/daq1/github/citkid/citkid/crs/parser',
                            interface='enp2s0', delete_parser_data = False,
                            verbose = True, return_raw = False):
        """
        Captures a noise timestream using the parser.
        
        Parameters:
        fres (array-like): tone frequencies in Hz 
        ares (array-like): tone amplitudes in dBm 
        fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz 
            5 -> 1,192.09 Hz 
            4 -> 2,384.19 Hz, will drop some packets
        parser_loc (str): path to the parser file 
        data_path (str): path to the data output file for the parser 
        interface (str): Ethernet interface identifier 
        delete_parser_data (bool): If True, deletes the parser data files 
            after importing the data 
        
        Returns:
        """ 
        if fir_stage <= 4:
            warning (f"packets will drop if fir_stage < 5", UserWarning)
        fres, ares = np.asarray(fres), np.asarray(ares) 
        os.makedirs('tmp/', exist_ok = True)
        data_path = 'tmp/parser_data_00/'
        if os.path.exists(data_path):
            raise FileExistsError(f'{data_path} already exists')
        # set fir stage
        await self.d.set_fir_stage(fir_stage) # Probably will drop packets after 4
        # get_samples will error if fir_stage is too low, but parser will not error
        self.sample_frequency = 625e6 / (256 * 64 * 2 ** fir_stage) 
        # print(f'fir stage is {await self.d.get_fir_stage()}')

        
        # set the tones
        await self.write_tones(fres, ares)
        sleep(1)
        # Get calibration data
        await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, self.module_index) 
        samples_cal = await self.d.py_get_samples(20, module=self.module_index)
        zcal = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
        zcal = np.mean(zcal[:len(fres)], axis = 1)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, self.module_index)
        np.save('tmp/zcal.npy', [np.real(zcal), np.imag(zcal)]) # Save in case it crashes
        sleep(0.1)
        # Collect the data 
        num_samps = int(self.sample_frequency*(noise_time + 10))
        parser = subprocess.Popen([parser_loc, '-d', data_path, '-i', interface, '-s', 
                                   f'{self.crs_sn:04d}', '-m', str(self.module_index), '-n', str(num_samps)], 
                                   shell=False)
        pbar = list(range(int(noise_time * 1.5)))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        for i in pbar:
            sleep(1) 
        # Set fir stage back
        await self.d.set_fir_stage(6)
        # read the data and convert to z
        zraw = convert_parser_to_z(data_path, self.crs_sn, self.module_index, ntones = len(fres))
        z = remove_internal_phaseshift(fres[:, np.newaxis], zraw, zcal[:, np.newaxis])
        if delete_parser_data:
            shutil.rmtree('tmp/')
        z /= 10 ** (ares[:, np.newaxis] / 20)
        if return_raw:
            return z, zcal, zraw
        return z

    async def capture_fast_noise(self, frequency, amplitude, nsamps):
        """ 
        Captures noise with a 2.44 MHz sample rate on a single channel. Turns on only 
        a single channel to avoid noise spikes from neighboring channels. Note that 
        the output will have to be corrected for the nonlinear PFB bin after taking a 
        PSD. It is harder to correct the timestream so single-photon events and 
        cosmic rays will not have the correct shape. 
        
        Parameters:
        frequency (float): tone frequency in Hz 
        amplitude (float): tone amplitude in dBm 
        nsamps (int): number of samples. Max is 1e6
        """
        fsample = 625e6 / 256
        await self.write_tones([frequency], [amplitude])
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, self.module_index)
        sleep(1)
        pfb_samples = await self.d.get_pfb_samples(int(nsamps), 'RAW', 1, self.module_index) # May have 20% exact difference 
        pfb_samples = np.array([complex(*sample) for sample in pfb_samples])
        await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, self.module_index)
        
        fraw, fft_corr_raw, builtin_gain_factor, pfb_sample_len =\
            apply_pfb_correction(pfb_samples, self.nco_freq, frequency, binlim=0.6e6, trim=True)
        cal_samples = await self.d.get_pfb_samples(2100, 'RAW', 1, self.module_index)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, self.module_index)
        cal_samples = np.array([complex(*sample) for sample in cal_samples][100:])
        fcal, fft_corr_cal, builtin_gain_factor_cal, pfb_sample_len_cal =\
            apply_pfb_correction(cal_samples, self.nco_freq, frequency, binlim=0.6e6, trim=True)

        ifft_raw, qfft_raw = [np.fft.fftshift(x) for x in separate_iq_fft_to_i_and_q(np.fft.fftshift(fft_corr_raw))]
        zraw = ifft_raw + 1j * qfft_raw
        ifft_cal, qfft_cal = [np.fft.fftshift(x) for x in separate_iq_fft_to_i_and_q(np.fft.fftshift(fft_corr_cal))]
        zcal = ifft_cal + 1j * qfft_cal
        zcal = np.mean(zcal)

        # Max of nsamps is 1e5 
        z = remove_internal_phaseshift(frequency, zraw, zcal) * self.volts_per_roc
        z /= 10 ** (ares[:, np.newaxis] / 20)
        return fraw, z