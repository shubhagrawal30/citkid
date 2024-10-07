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
from .util import volts_per_roc, remove_internal_phaseshift_noise
from hidfmux.core.utils.transferfunctions import apply_pfb_correction
from hidfmux.analysis.noise_processing import separate_iq_fft_to_i_and_q


class CRS:
    def __init__(self, crs_sn = 27): 
        """ 
        Initializes the crs object d. Not that the system must be 
        configured using CRS.configure_system before measurements 

        Parameters:
        crs_sn (int): CRS serial number 
        """ 
        self.crs_sn = crs_sn
        s = rfmux.load_session('!HardwareMap [ !CRS { ' + f'serial: "{crs_sn:04d}"' + ' } ]')
        self.d = s.query(rfmux.CRS).one()
        self.volts_per_roc = volts_per_roc
        
    async def configure_system(self, clock_source="SMA", full_scale_dbm = 1, verbose = True):
        """
        Resolves the system, sets the timestamp port, sets the clock source, and sets the full scale
        in dBm 
        
        Parameters:
        clock_source (str): clock source specification. 'VCXO' for the internal voltage controlled
            crystal oscillator or 'SMA' for the external 10 MHz reference (5 Vpp I think) 
        full_scale_dbm (int): full scale power in dBm. range is [???, 7?]
        verbose (bool): If True, gets and prints the clocking source 
        """
        await self.d.resolve()

        just_booted = await self.d.get_timestamp_port() != 'TEST'
        if just_booted: 
            await self.d.set_timestamp_port(self.d.TIMESTAMP_PORT.TEST)  
        
        await self.d.set_clock_source(clock_source)
        for module in range(1, 5):
            await self.d.set_dac_scale(full_scale_dbm, self.d.UNITS.DBM, module)
        self.full_scale_dbm = full_scale_dbm
        
        sleep(1)
        if verbose:
            print('System configured')
            print("Clocking source is", await self.d.get_clock_source())
        
    async def set_nco(self, module, nco_freq, verbose = True):
        """Set the NCO frequency
        
        Parameters:
        module (int): module index
        nco_freq (float): NCO frequency in Hz. This should not be a round number
        verbose (bool): If True, gets and prints the NCO frequency
        """
        await self.d.set_nco_frequency(nco_freq, self.d.UNITS.HZ, module = module)
        self.nco_freq = await self.d.get_nco_frequency(self.d.UNITS.HZ, module=module)
        if verbose:
            print(f'NCO is {self.nco_freq}')
            
    async def write_tones(self, module, fres, ares):
        """
        Writes an array of tones given frequencies and amplitudes 

        Parameters:
        module (int): module index 
        fres (array-like): list of frequencies in Hz 
        ares (array-like): list of powers in dBm (+- 2 dBm precision for now) 
        """
        try:
            nco = self.nco_freq
        except:
            raise Exception('NCO frequency has not been set')
        fres = np.asarray(fres) 
        ares = np.asarray(ares)
        if any(ares > self.full_scale_dbm):
            raise ValueError(f'ares must not exceed {self.full_scale_dbm} dBm: raise full_scale_dbm or lower powers')
        if any(ares < -60) and len(ares) < 100:
            warnings.warn(f"values in ares are < 60 dBm: digitization noise may occur", UserWarning)
        ares_amplitude = 10 ** ((ares - self.full_scale_dbm) / 20)
        
        await self.d.clear_channels()
        
        nco = await self.d.get_nco_frequency(self.d.UNITS.HZ, module=module)
        async with self.d.tuber_context() as ctx:
            for ch, (fr, ar) in enumerate(zip(fres,ares_amplitude)):
                ## To be wrapped in context_manager
                ctx.set_frequency(fr-nco, self.d.UNITS.HZ, ch+1, module=module)
                ctx.set_amplitude(ar, self.d.UNITS.NORMALIZED, target=self.d.TARGET.DAC, channel=ch+1, module=module)
            await ctx()

    async def sweep(self, module, frequencies, ares, nsamps = 10, verbose = True, pbar_description = 'Sweeping',
                    return_raw = False):
        """
        Performs a frequency sweep and returns the complex S21 value at each frequency. Performs sweeps over 
        axis 0 of frequencies simultaneously 

        Parameters:
        module (int): module index 
        frequencies (M X N array-like float): the first index M is the channel index (max len 1024) 
            and the second index N is the frequency in Hz for a single point in the sweep
        ares (M array-like float): list of amplitudes in dBm for each channel 
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        z (M X N array-like complex): complex S21 data in V for each frequency in f 

        """
        frequencies, ares = np.asarray(frequencies), np.asarray(ares)
        n_channels, n_points = frequencies.shape
        if len(ares) != n_channels:
            raise ValueError('ares and frequencies are not the same length')
        # Set the fir stage 
        fir_stage = 6 
        await self.d.set_fir_stage(fir_stage) 
        sample_frequency = 625e6 / (256 * 64 * 2**fir_stage)

        # Write amplitudes 
        fres = [fi[0] for fi in frequencies]
        await self.write_tones(module, fres, ares)
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
                    f = frequencies[ch, sweep_index]
                    ctx.set_frequency(f - self.nco_freq, self.d.UNITS.HZ, ch + 1, module = module)
                await ctx()

            # take data and loopback calibration data
            await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, module)
            samples_cal = await self.d.get_samples(21,module=module)
            await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module)
            samples = await self.d.get_samples(nsamps + 1,module=module)
            # format and average data 
            zi = np.asarray(samples.i) + 1j * np.asarray(samples.q)
            zi = np.mean(zi[:n_channels, 1:] , axis = 1)
            zical = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
            zical = np.mean(zical[:n_channels, 1:], axis = 1)
            # adjust for loopback calibration
            zcal[:, sweep_index] = zical 
            zraw[:, sweep_index] = zi 
            zi = remove_internal_phaseshift(frequencies[:, sweep_index], zi, zical) 
            z[:, sweep_index] = zi * self.volts_per_roc 
        # Turn off channels 
        await self.d.clear_channels()
        if return_raw:
            return z, zcal, zraw 
        return z

    async def sweep_linear(self, module, fres, ares, bw = 20e3, npoints = 10, 
                           nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where each channel is swept over the same range in 
        frequencies

        Parameters:
        module (int): module index 
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
        z = await self.sweep(module, f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
        return f, z

    async def sweep_qres(self, module, fres, ares, qres, npoints = 10, nsamps = 10,
                         verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep where the span around each frequency is set 
        equal to fres / qres 

        Parameters:
        module (int): module index 
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
        z = await self.sweep(module, f, ares, nsamps = nsamps, 
                            verbose = verbose, pbar_description = pbar_description)
        return f, z

    async def sweep_full(self, module, amplitude, npoints = 10, 
                         nsamps = 10, verbose = True, pbar_description = 'Sweeping'):
        """
        Performs a frequency sweep over the full 600 MHz bandwidth around the NCO 
        frequency

        Parameters:
        module (int): module index 
        amplitude (float): amplitude in dBm
        npoints (int): number of sweep points per channel
        nsamps (int): number of samples to average per point 
        verbose (bool): If True, displays a progress bar while sweeping 
        pbar_description (str): description for the progress bar 

        Returns:
        f (np.array): array of frequencies in Hz 
        z (np.array): array of complex S21 data corresponding to f 
        """ 
        fres = np.linspace(self.nco_freq - 300e6, self.nco_freq + 300e6, 1024) 
        fres += np.random.uniform(-100, 100, len(fres))
        ares = amplitude * np.ones(len(fres))
        bw = 600e6 / 1024 + 200
        f, z = await self.sweep_linear(module, fres, ares, bw = bw, npoints = npoints,
                                      nsamps = nsamps, verbose = verbose, 
                                      pbar_description = pbar_description)
        f, z = f.flatten(), z.flatten()
        ix = np.argsort(f)
        f, z = f[ix], z[ix]
        return f, z

    async def capture_noise(self, module, fres, ares, noise_time, fir_stage = 6,
                            parser_loc='/home/daq1/github/citkid/citkid/crs/parser',
                            interface='enp2s0', delete_parser_data = False,
                            verbose = True, return_raw = False):
        """
        Captures a noise timestream using the parser.
        
        Parameters:
        module (int): module index 
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
        print(f'fir stage is {await self.d.get_fir_stage()}')

        
        # set the tones
        await self.write_tones(module, fres, ares)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module)
        sleep(1)
        # Get calibration data
        await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, module) 
        samples_cal = await self.d.get_samples(21, module=module)
        zcal = np.asarray(samples_cal.i) + 1j * np.asarray(samples_cal.q) 
        zcal = np.mean(zcal[:len(fres), 1:], axis = 1)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module)
        np.save('tmp/zcal.npy', [np.real(zcal), np.imag(zcal)]) # Save in case it crashes
        sleep(0.1)
        # Collect the data 
        num_samps = int(self.sample_frequency*noise_time)
        parser = subprocess.Popen([parser_loc, '-d', data_path, '-i', interface, '-s', 
                                   f'{self.crs_sn:04d}', '-m', str(module), '-n', str(num_samps)], 
                                   shell=False)
        pbar = list(range(int(noise_time) + 2))
        if verbose:
            pbar = tqdm(pbar, leave = False)
        for i in pbar:
            sleep(1) 
        # read the data and convert to z
        zraw = convert_parser_to_z(data_path, self.crs_sn, module, ntones = len(fres))
        z = remove_internal_phaseshift(fres[:, np.newaxis], zraw, zcal[:, np.newaxis])
        # z = remove_internal_phaseshift_noise(fres[:, np.newaxis], z, ffine, zfine) 
        # z *= np.exp(1j * 0.8394967866987446)
        if delete_parser_data:
            shutil.rmtree('tmp/')
        if return_raw:
            return z, zcal, zraw
        return z

    async def capture_fast_noise(self, module, frequency, amplitude, nsamps):
        """ 
        Captures noise with a 2.44 MHz sample rate on a single channel. Turns on only 
        a single channel to avoid noise spikes from neighboring channels. Note that 
        the output will have to be corrected for the nonlinear PFB bin after taking a 
        PSD. It is harder to correct the timestream so single-photon events and 
        cosmic rays will not have the correct shape. 
        
        Parameters:
        module (int): module index 
        frequency (float): tone frequency in Hz 
        amplitude (float): tone amplitude in dBm 
        nsamps (int): number of samples. Max is 1e6
        """
        fsample = 625e6 / 256
        await self.write_tones(module, [frequency], [amplitude])
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module)
        sleep(1)
        pfb_samples = await self.d.get_pfb_samples(int(nsamps), 'RAW', 1, module) # May have 20% exact difference 
        pfb_samples = np.array([complex(*sample) for sample in pfb_samples])
        await self.d.set_dmfd_routing(self.d.ROUTING.CARRIER, module)
        
        fraw, fft_corr_raw, builtin_gain_factor, pfb_sample_len =\
            apply_pfb_correction(pfb_samples, self.nco_freq, frequency, binlim=0.6e6, trim=True)
        cal_samples = await self.d.get_pfb_samples(2100, 'RAW', 1, module)
        await self.d.set_dmfd_routing(self.d.ROUTING.ADC, module)
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
        return fraw, z