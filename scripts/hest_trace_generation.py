"""
hest_trace_generation.py — Convert HeST simulate_spectrum.py H5 output into
realistic 24-channel TES waveform traces (noise + signal pulses) in pytesdaq
HDF5 format.

Usage:
    python hest_trace_generation.py \
        --hest-file sims/Spectrum_Flat_ER_...h5 \
        --filter-file herald_filter.hdf5 \
        --save-path ./traces_out/ \
        --event-length-sec 10 \
        --uniform-channel channel_00

The HeST H5 file must have been produced with --save_raw, containing:
    events/{i}/{channel}/{sensor}/arrival_times  (µs)
    events/{i}/{channel}/{sensor}/energies       (eV)
where channel is one of: singlet, triplet, ir, qp
and sensor is an integer index 0..n_sensors-1.

"""

import argparse
import datetime as dt
import gc
import os
import sys

import h5py
import numpy as np
import qetpy as qp
import vaex as vx
from detprocess import FilterData, YamlConfig, Salting, RawData
from detprocess.utils import create_series_name
from pytesdaq.io import H5Writer, H5Reader
from pytesio import convert_length_msec_to_samples
from qetpy.utils import convert_channel_name_to_list
from tqdm import tqdm

from pytessim import Noise_Factory, Background_Manager
from pytessim.utils import gen_metadata, time_to_string, string_to_time


CHANNEL_KEYS = ['singlet', 'triplet', 'ir', 'qp']
N_CHANNELS = 24


def load_scaled_templates(filter_data, channels, template_tag, dpdi_tag,
                          dpdi_poles, uniform_channel=None):
    """
    Load templates from the filter file and pre-scale them to
    current-per-eV units for fast pulse injection.

    Returns
    -------
    templates : np.ndarray, shape (n_channels, n_template_samples)
    """
    templates = []

    if uniform_channel is not None:
        src_channels = [uniform_channel]
    else:
        src_channels = channels

    for chan in src_channels:
        tag = template_tag if template_tag is not None else chan
        dt_tag = dpdi_tag if dpdi_tag is not None else chan

        template, time_array, _ = filter_data.get_template(
            chan, tag=tag, return_metadata=True
        )
        if template.ndim > 1:
            template = np.squeeze(template)
            if template.ndim > 1:
                template = template[0]

        dpdi, _ = filter_data.get_dpdi(channel=chan, poles=dpdi_poles, tag=dt_tag)
        energy_norm = qp.get_energy_normalization(
            time_array, template, dpdi=dpdi, lgc_ev=True
        )
        templates.append(template * energy_norm)

    templates = np.array(templates)

    if uniform_channel is not None:
        templates = np.tile(templates, (N_CHANNELS, 1))

    return templates


def inject_pulses(traces, hest_event_grp, scaled_templates, trigger_offset,
                  fs, n_sensors):
    """
    Read per-quanta arrival times and energies from a HeST event group
    and add scaled template pulses to the trace array.
    """
    n_samples = traces.shape[1]

    for ch_key in CHANNEL_KEYS:
        if ch_key not in hest_event_grp:
            continue
        ch_grp = hest_event_grp[ch_key]
        weight = ch_grp.attrs.get('weight', 1.0)

        for sensor_idx in range(n_sensors):
            s_key = str(sensor_idx)
            if s_key not in ch_grp:
                continue
            if sensor_idx >= N_CHANNELS:
                continue

            s_grp = ch_grp[s_key]
            times_us = s_grp['arrival_times'][:]
            energies_eV = s_grp['energies'][:] * weight

            template = scaled_templates[sensor_idx]
            tlen = len(template)

            for q in range(len(times_us)):
                idx = int(fs * times_us[q] * 1e-6) + trigger_offset
                end_idx = idx + tlen

                if idx >= n_samples:
                    continue
                elif idx >= 0 and end_idx <= n_samples:
                    traces[sensor_idx][idx:end_idx] += template * energies_eV[q]
                elif idx < 0:
                    start_clip = abs(idx)
                    traces[sensor_idx][:tlen - start_clip] += (
                        template[start_clip:] * energies_eV[q]
                    )
                else:
                    traces[sensor_idx][idx:] += (
                        template[:n_samples - idx] * energies_eV[q]
                    )


def traces_to_adc(traces):
    """Convert TES current traces to int16 ADC counts."""
    traces *= 1e6
    traces /= 10
    traces *= 2**16
    return np.clip(traces, -32768, 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(
        description='Generate TES traces from HeST simulation H5 output')

    parser.add_argument('--hest-file', '--hest_file',
                        dest='hest_file', type=str, required=True,
                        help='Path to HeST H5 file (from simulate_spectrum.py --save_raw)')
    parser.add_argument('--filter-file', '--filter_file',
                        dest='filter_file', type=str, required=True,
                        help='Path to filter HDF5 file with templates and PSDs')
    parser.add_argument('--save-path', '--save_path',
                        dest='save_path', type=str, required=True,
                        help='Output directory for pytesdaq HDF5 files')
    parser.add_argument('--event-length-sec', '--event_length_sec',
                        dest='event_length_sec', type=float, default=10,
                        help='Trace length in seconds (default: 10)')
    parser.add_argument('--series-length-sec', '--series_length_sec',
                        dest='series_length_sec', type=float, default=600,
                        help='Series length in seconds (default: 600)')
    parser.add_argument('--fs',
                        dest='fs', type=float, default=1.25e6,
                        help='Sampling frequency in Hz (default: 1.25e6)')
    parser.add_argument('--comment', '-c',
                        dest='comment', type=str, default='HeST trace generation',
                        help='Metadata comment string')
    parser.add_argument('--filesize-lim-GB', '--filesize_lim_GB',
                        dest='filesize_lim_GB', type=float, default=1,
                        help='Max output file size in GB (default: 1)')
    parser.add_argument('--no-noise', '--no_noise',
                        dest='no_noise', action='store_true',
                        help='Skip noise generation (signal-only traces)')
    parser.add_argument('--trigger-offset-msec', '--trigger_offset_msec',
                        dest='trigger_offset_msec', type=float, default=5000,
                        help='Time in trace (ms) where HeST t=0 maps to (default: 5000)')
    parser.add_argument('--template-tag', '--template_tag',
                        dest='template_tag', type=str, default=None,
                        help='Template tag in filter file (default: channel name)')
    parser.add_argument('--noise-tag', '--noise_tag',
                        dest='noise_tag', type=str, default=None,
                        help='PSD/CSD tag in filter file (default: channel name)')
    parser.add_argument('--uniform-channel', '--uniform_channel',
                        dest='uniform_channel', type=str, default=None,
                        help='Use this single channel template+PSD for all 24 sensors')
    parser.add_argument('--dpdi-tag', '--dpdi_tag',
                        dest='dpdi_tag', type=str, default=None,
                        help='dP/dI tag in filter file (default: channel name)')
    parser.add_argument('--dpdi-poles', '--dpdi_poles',
                        dest='dpdi_poles', type=int, default=1,
                        help='Number of poles for dP/dI lookup (default: 1)')
    parser.add_argument('--enable-LEE', '--enable_LEE',
                        dest='enable_LEE', action='store_true',
                        help='Inject low-energy excess pulses after trace generation')
    parser.add_argument('--enable-background', '--enable_background',
                        dest='enable_background', action='store_true',
                        help='Inject Geant4 backgrounds after trace generation')
    parser.add_argument('--processing-setup', '--processing_setup',
                        dest='processing_setup', type=str, default=None,
                        help='Processing YAML file path (required for --enable-LEE)')
    parser.add_argument('--topology-setup', '--topology_setup',
                        dest='topology_setup', type=str, default=None,
                        help='Topology YAML file path (required for --enable-background)')

    args = parser.parse_args()

    if args.enable_LEE and args.processing_setup is None:
        raise ValueError('--processing-setup is required when --enable-LEE is set')
    if args.enable_background and args.topology_setup is None:
        raise ValueError('--topology-setup is required when --enable-background is set')

    # --- Derived parameters ---
    n_samples = int(args.fs * args.event_length_sec)
    trigger_offset = int(args.fs * args.trigger_offset_msec * 1e-3)

    channel_names = [f'channel_{i:02d}' for i in range(N_CHANNELS)]

    date_time_string = time_to_string(dt.datetime.now())
    save_path = args.save_path.rstrip('/') + '/continuous_I9_' + date_time_string

    if args.series_length_sec % args.event_length_sec != 0:
        raise ValueError(
            f'Event length ({args.event_length_sec}s) does not evenly divide '
            f'series length ({args.series_length_sec}s)')
    n_events_per_series = int(args.series_length_sec / args.event_length_sec)

    # --- Open HeST H5 file ---
    print(f'Opening HeST file: {args.hest_file}')
    hest_h5 = h5py.File(args.hest_file, 'r')
    n_events = hest_h5.attrs['n_events']
    n_sensors = hest_h5.attrs['n_sensors']
    print(f'  {n_events} events, {n_sensors} sensors')

    if n_sensors > N_CHANNELS:
        print(f'  WARNING: HeST has {n_sensors} sensors but only first '
              f'{N_CHANNELS} will be used')

    # --- Load templates ---
    print('Loading templates and computing energy normalization...')
    filter_data = FilterData()
    filter_data.load_hdf5(args.filter_file)

    scaled_templates = load_scaled_templates(
        filter_data, channel_names, args.template_tag, args.dpdi_tag,
        args.dpdi_poles, uniform_channel=args.uniform_channel
    )
    print(f'  Template length: {scaled_templates.shape[1]} samples')

    # --- Initialize noise ---
    enable_noise = not args.no_noise
    if enable_noise:
        print('Initializing noise factory...')
        noise_factory = Noise_Factory(
            args.filter_file, event_length_sec=args.event_length_sec,
            fs=args.fs
        )
        if args.uniform_channel is not None:
            noise_chan = args.uniform_channel
            noise_factory.initialize_factory(noise_chan)
            print(f'  Uniform noise from channel: {noise_chan}')
        else:
            for chan in channel_names:
                noise_factory.initialize_factory(chan)
            print(f'  Initialized {N_CHANNELS} independent noise channels')

    # --- Set up file writing ---
    start_time = string_to_time(date_time_string) + dt.timedelta(seconds=1)
    start_epoch = start_time.timestamp()
    time_counter = 0.0
    event_idx = 0

    print(f'Generating traces for {n_events} events...')
    print(f'  Event length: {args.event_length_sec}s ({n_samples} samples)')
    print(f'  Trigger offset: {trigger_offset} samples '
          f'({args.trigger_offset_msec} ms)')
    print(f'  Noise: {"enabled" if enable_noise else "disabled"}')

    writer = H5Writer()

    events_grp = hest_h5['events']

    for event_idx in tqdm(range(n_events)):
        # Start a new series if needed
        if event_idx % n_events_per_series == 0:
            if event_idx > 0:
                writer._close_file()

            series_string = time_to_string(
                start_time + dt.timedelta(seconds=time_counter)
            )
            writer.initialize(
                'I9_' + series_string, data_path=save_path
            )
            file_metadata, adc_metadata, det_metadata = gen_metadata(
                N_CHANNELS, comment=args.comment, fs=args.fs,
                sec_per_event=args.event_length_sec,
                sec_per_series=args.series_length_sec,
                dataset_start_epoch=start_epoch,
                series_string=series_string
            )
            file_metadata['timestamp'] = start_epoch + time_counter
            file_metadata['series_start'] = start_epoch + time_counter
            writer.set_metadata(
                file_metadata=file_metadata, adc_config=adc_metadata,
                detector_config=det_metadata
            )
            writer._open_file(prefix='cont')

        # Check filesize limit
        current_filename = writer._current_file_name
        if os.path.getsize(current_filename) / 1e9 > args.filesize_lim_GB:
            writer._close_file()
            file_metadata['timestamp'] = start_epoch + time_counter
            file_metadata['dump_number'] += 1
            writer.set_metadata(
                file_metadata=file_metadata, adc_config=adc_metadata,
                detector_config=det_metadata
            )
            writer._open_file(prefix='cont')

        # Build trace
        traces = np.zeros((N_CHANNELS, n_samples))

        # Add noise
        if enable_noise:
            if args.uniform_channel is not None:
                for i in range(N_CHANNELS):
                    traces[i] = noise_factory.sample_noise(args.uniform_channel)
            else:
                for i, chan in enumerate(channel_names):
                    traces[i] = noise_factory.sample_noise(chan)

        # Add signal pulses from HeST
        evt_key = str(event_idx)
        if evt_key in events_grp:
            inject_pulses(
                traces, events_grp[evt_key], scaled_templates,
                trigger_offset, args.fs, n_sensors
            )

        # Convert to ADC and write
        data = traces_to_adc(traces)
        writer.write_event(data, start_epoch + time_counter)
        time_counter += args.event_length_sec

    # Close final file
    writer._close_file()
    hest_h5.close()

    total_time_sec = n_events * args.event_length_sec
    channel_list = channel_names

    #########################################
    # Setting up the Salting class for LEE  #
    #########################################

    if args.enable_LEE:
        print('INFO: Setting up LEE injection...')

        config = YamlConfig(args.processing_setup, channel_list)
        LEE_config = config.get_config('salting')

        LEE_factory = Salting(args.filter_file)
        rawdata_obj = RawData(save_path, data_type='cont')
        LEE_factory.set_raw_data(rawdata_obj)

        if 'nsalt' in LEE_config['overall']:
            print('Warning: Ignoring argument nsalt; LEE counts should'
                  ' be defined as a rate for each channel.')
        if 'energies' in LEE_config['overall']:
            print('Warning: Ignoring argument energies; LEE distributions'
                  ' should be defined with a pdf tag/file for each channel.')

        LEE_PDF_file = None
        if 'LEE_PDF_file' in LEE_config['overall']:
            LEE_PDF_file = LEE_config['overall']['LEE_PDF_file']
        else:
            raise ValueError('ERROR: if enable_LEE is set, user must'
                             ' give a path to the LEE_PDF_file in the .yaml')

        chan_dataframe_list = []
        for salt, salt_config in LEE_config['channels'].items():
            print(f'INFO: Generating LEE for {salt}')

            template_tag = salt_config['template_tag']
            pdf_tag = salt_config['pdf_tag']
            pdf_bounds = salt_config['pdf_bounds']
            rate = salt_config['rate']
            dpdi_poles_lee = salt_config.get('dpdi_poles', None)
            dpdi_tag_lee = salt_config.get('dpdi_tag', None)
            chan = salt_config.get('channel', salt)
            chan_list_lee = convert_channel_name_to_list(chan)

            if (dpdi_tag_lee is None) != (dpdi_poles_lee is None):
                raise ValueError("Both 'dpdi_tag' and 'dpdi_poles' must be "
                                 "either set or None.")

            pce = 1
            if 'collection_efficiency' in salt_config:
                pce = salt_config['collection_efficiency']
            elif len(chan_list_lee) >= 2:
                pce = [pce] * len(chan_list_lee)

            LEE_factory.generate_salt(chan,
                                      energies=None,
                                      template_tag=template_tag,
                                      dpdi_tag=dpdi_tag_lee,
                                      dpdi_poles=dpdi_poles_lee,
                                      pdf_file=LEE_PDF_file,
                                      pdf_tag=pdf_tag,
                                      pdf_bounds=pdf_bounds,
                                      PCE=pce,
                                      rate=rate,
                                      poisson=True,
                                      do_salt_deadtime=True)
            LEE_dataframe = LEE_factory.get_dataframe()
            chan_dataframe_list.append(LEE_dataframe)
            LEE_factory.clear_dataframe()

        final_dataframe = chan_dataframe_list[0]
        if len(chan_dataframe_list) > 1:
            final_dataframe = vx.concat(chan_dataframe_list)

        series_name = create_series_name(9)
        file_name = f'LEE_pdf_{series_name}_F0001.hdf5'
        if not os.path.exists(f'{save_path}/salting'):
            os.mkdir(f'{save_path}/salting')
            print(f'INFO: Creating Salting Directory at {save_path}/salting')
        salting_file_path = f'{save_path}/salting/{file_name}'
        final_dataframe.export_hdf5(salting_file_path, mode='w')
        final_dataframe.close()
        salting_dataframe_list = [salting_file_path]

        del final_dataframe, LEE_factory
        gc.collect()

        ######################
        # Adding LEE to file #
        ######################
        print('INFO: Adding LEE to the generated data files')
        LEE_factory = Salting(args.filter_file)
        LEE_factory.set_dataframe(salting_dataframe_list)
        file_editor = H5Reader(edit_mode=True)
        file_editor.set_files(save_path)
        while True:
            old_traces, metadata = file_editor.read_next_event(
                include_metadata=True, adctoamp=True)
            if metadata['error_msg'] == 'No more files available':
                break
            new_traces = LEE_factory.inject_raw_salt(
                channel_list, old_traces,
                metadata['series_num'], metadata['event_num'])
            new_traces = traces_to_adc(new_traces)
            file_editor.set_this_event(new_traces, detector_chans=channel_list)
        file_editor.close()
        print('INFO: Finished LEE injection.')

    ###################################
    # Adding Backgrounds to Data      #
    ###################################

    if args.enable_background:
        print('INFO: Setting up background injection...')

        BG_Manager = Background_Manager(
            args.topology_setup,
            args.event_length_sec,
            args.series_length_sec,
            total_time_sec,
            args.fs)

        file_editor = H5Reader(edit_mode=True)
        file_editor.set_files(save_path)
        while True:
            old_traces, metadata = file_editor.read_next_event(
                include_metadata=True, adctoamp=True)
            if metadata['error_msg'] == 'No more files available':
                break
            new_traces = BG_Manager.inject_raw_salt(
                channel_list, old_traces,
                metadata['series_num'], metadata['event_num'])
            new_traces = traces_to_adc(new_traces)
            file_editor.set_this_event(new_traces, detector_chans=channel_list)
        file_editor.close()
        print('INFO: Finished background injection.')

    print(f'Done. Output written to: {save_path}')


if __name__ == '__main__':
    main()
