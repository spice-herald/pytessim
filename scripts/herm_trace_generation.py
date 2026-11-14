"""
herm_trace_generation.py — Convert HeRM simulate_spectrum.py H5 output into
realistic multichannel TES waveform traces with optional noise and signals,
in pytesdaq continuous-acquisition HDF5 format.

Required arguments
------------------
    --herm-file PATH    HeRM HDF5 file made with --save_raw.
    --filter-file PATH  Filter file with templates, dP/dI, and noise PSDs/CSDs.
    --save-path PATH    Parent for a new continuous_I9_* output directory.

Example using the provided inputs, including noise and native LEE
------------------------------------------------------------------
    python scripts/herm_trace_generation.py \
        --herm-file inputs/Spectrum_Mono_ER_energyMin_5900_energyMax_5900_nEvents_10000_seed_42069_HeraldLBNL_TopOnly.h5 \
        --filter-file inputs/herald_filter.hdf5 \
        --save-path ./outputs \
        --processing-setup inputs/lee.yaml \
        --event-length-sec 10 --series-length-sec 600 --fs 1.25e6 \
        --uniform-channel channel_00 --dpdi-poles 2

Noise
-----
Noise is enabled by default; no separate enable flag is needed. It requires
PSDs/CSDs in the filter file. For signal-only traces without noise, add:
    --no-noise

LEE
---
Native LEE is disabled by default. Each enabled channel gets independent
Poisson times and energies from dN/dE ∝ E^(-power_law_index) between positive
bounds in eV. The energy multiplies the corresponding energy-normalized filter
template. LEE is injected before ADC conversion, with times and energies saved
in <generated-save-path>/salting/lee_truth.hdf5.

Enable it using a YAML file with a complete top-level ``lee:`` section:
    --processing-setup inputs/lee.yaml

The example inputs/lee.yaml has enabled: true. Its other fields are rate_hz
(events/second/channel), energy_min_ev, energy_max_ev, power_law_index, and the
optional seed. It can be included in a larger processing YAML or used alone as
shown above. A top-level ``lee:`` section selects native LEE, not the legacy
``salting:`` PDF workflow.

Explicit CLI values override YAML values; the LEE seed does not seed noise:
    --lee-rate-hz 50 --lee-energy-min-ev 2 --lee-energy-max-ev 80 \
    --lee-power-law-index 1.5 --lee-seed 10

If the YAML has ``lee.enabled: false`` or no ``enabled`` field, add:
    --enable-LEE

Suppress LEE even when YAML enables it:
    --disable-LEE

Supplying ``--lee-*`` parameters alone does not enable injection. All native
parameters can also be supplied on the CLI without a processing YAML, in which
case use --enable-LEE. For backward compatibility, --enable-LEE without a native
``lee:`` section or native ``--lee-*`` parameters selects the legacy PDF workflow
and requires an appropriate --processing-setup file.

Other commonly used parameters
------------------------------
    --event-length-sec S   Duration of one waveform event (default: 10).
    --series-length-sec S  Series duration; an integer multiple of the event
                           length (default: 600).
    --fs HZ                Sampling frequency (default: 1.25e6).
    --max-events N         Limit the number of HeRM events processed.
    --trigger-offset-msec MS
                           Map HeRM arrival time zero to this time in the trace
                           (default: 5000). This is not applied to LEE.
    --template-tag TAG, --noise-tag TAG, --dpdi-tag TAG
                           Override filter lookup tags, which otherwise default
                           to each channel's name.
    --dpdi-poles N         dP/dI pole count (default: 1). The supplied
                           herald_filter.hdf5 requires --dpdi-poles 2.
    --uniform-channel CHANNEL
                           Reuse this channel's template and PSD for all
                           channels. For the supplied one-channel filter, use
                           --uniform-channel channel_00.
    --filesize-lim-GB GB   Raw-file size threshold for starting a new dump
                           within a series (default: 1).
    --comment TEXT         Store a comment in the generated file metadata.
    --enable-background --topology-setup PATH
                           Add Geant4 backgrounds after generation using the
                           corresponding topology YAML.

CLI options also accept underscores in place of dashes. Run --help for the
complete argument list.

The HeRM H5 file must have been produced with --save_raw, containing:
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
from pytessim.utils import gen_metadata, time_to_string, string_to_time, load_yaml_config


CHANNEL_KEYS = ['singlet', 'triplet', 'ir', 'qp']


def load_scaled_templates(filter_data, channels, template_tag, dpdi_tag,
                          dpdi_poles, fs, n_channels, uniform_channel=None):
    """
    Load templates from the filter file and pre-scale them to
    current-per-eV units for fast pulse injection.

    Returns
    -------
    templates : np.ndarray, shape (n_channels, n_template_samples)
    pretrigger_samples : int, number of baseline samples before pulse onset
    """
    templates = []
    pretrigger_samples = 0

    if uniform_channel is not None:
        src_channels = [uniform_channel]
    else:
        src_channels = channels

    for chan in src_channels:
        tag = template_tag if template_tag is not None else chan
        dt_tag = dpdi_tag if dpdi_tag is not None else chan

        template, time_array, metadata = filter_data.get_template(
            chan, tag=tag, return_metadata=True
        )
        if template.ndim > 1:
            template = np.squeeze(template)
            if template.ndim > 1:
                template = template[0]

        pretrigger_msec = metadata.get('pretrigger_length_msec', 0)
        pretrigger_samples = int(pretrigger_msec * 1e-3 * fs)

        dpdi, _ = filter_data.get_dpdi(channel=chan, poles=dpdi_poles, tag=dt_tag)
        energy_norm = qp.get_energy_normalization(
            time_array, template, dpdi=dpdi, lgc_ev=True
        )
        templates.append(template / energy_norm)

    templates = np.array(templates)

    if uniform_channel is not None:
        templates = np.tile(templates, (n_channels, 1))

    return templates, pretrigger_samples


def add_scaled_pulse(trace, template, start_sample, scale):
    """Add one scaled template to a trace, clipping at trace boundaries."""
    trace = np.asarray(trace)
    template = np.asarray(template)
    if trace.ndim != 1 or template.ndim != 1:
        raise ValueError('Trace and template must be one-dimensional arrays')
    if not isinstance(start_sample, (int, np.integer)):
        raise ValueError('start_sample must be an integer')

    start = int(start_sample)
    trace_start = max(start, 0)
    trace_end = min(start + len(template), len(trace))
    if trace_start >= trace_end:
        return

    template_start = trace_start - start
    template_end = template_start + trace_end - trace_start
    trace[trace_start:trace_end] += (
        template[template_start:template_end] * scale
    )


def inject_pulses(traces, herm_event_grp, scaled_templates, trigger_offset,
                  pretrigger_samples, fs, n_sensors):
    """
    Read per-quanta arrival times and energies from a HeRM event group
    and add scaled template pulses to the trace array.
    """
    for ch_key in CHANNEL_KEYS:
        if ch_key not in herm_event_grp:
            continue
        ch_grp = herm_event_grp[ch_key]
        weight = ch_grp.attrs.get('weight', 1.0)

        for sensor_idx in range(n_sensors):
            s_key = str(sensor_idx)
            if s_key not in ch_grp:
                continue
            if sensor_idx >= traces.shape[0]:
                continue

            s_grp = ch_grp[s_key]
            times_us = s_grp['arrival_times'][:]
            energies_eV = s_grp['energies'][:] * weight

            template = scaled_templates[sensor_idx]
            for q in range(len(times_us)):
                idx = (int(fs * times_us[q] * 1e-6) + trigger_offset
                       - pretrigger_samples)
                add_scaled_pulse(
                    traces[sensor_idx], template, idx, energies_eV[q]
                )


def sample_bounded_power_law(rng, size, energy_min_ev, energy_max_ev,
                             power_law_index):
    """Sample energies in eV from a bounded differential spectrum E^-alpha."""
    if (not isinstance(size, (int, np.integer)) or isinstance(size, bool)
            or size < 0):
        raise ValueError('size must be a nonnegative integer')

    energy_min_ev = _validate_finite_number(
        'energy_min_ev', energy_min_ev, strictly_positive=True
    )
    energy_max_ev = _validate_finite_number('energy_max_ev', energy_max_ev)
    power_law_index = _validate_finite_number(
        'power_law_index', power_law_index
    )
    if energy_max_ev <= energy_min_ev:
        raise ValueError('energy_max_ev must be greater than energy_min_ev')

    uniform = rng.random(int(size))
    if power_law_index == 1:
        return energy_min_ev * (energy_max_ev / energy_min_ev) ** uniform

    beta = 1.0 - power_law_index
    bounds = np.array([energy_min_ev ** beta, energy_max_ev ** beta])
    return (bounds[0] + uniform * (bounds[1] - bounds[0])) ** (1.0 / beta)


def generate_lee_events(n_channels, n_samples, fs, rate_hz, energy_min_ev,
                        energy_max_ev, power_law_index, pretrigger_samples,
                        rng):
    """Generate independent per-channel LEE times and energies for one trace."""
    for name, value in [('n_channels', n_channels), ('n_samples', n_samples),
                        ('pretrigger_samples', pretrigger_samples)]:
        if (not isinstance(value, (int, np.integer)) or isinstance(value, bool)
                or value < 0 or (name != 'pretrigger_samples' and value == 0)):
            raise ValueError(f'{name} must be a nonnegative integer'
                             + ('' if name == 'pretrigger_samples'
                                else ' greater than zero'))
    _validate_finite_number('fs', fs, strictly_positive=True)
    _validate_finite_number('rate_hz', rate_hz)
    if rate_hz < 0:
        raise ValueError('rate_hz must be nonnegative')

    n_samples = int(n_samples)
    duration_s = n_samples / fs
    events = []
    for _ in range(int(n_channels)):
        count = int(rng.poisson(rate_hz * duration_s))
        times_s = rng.uniform(0.0, duration_s, size=count)
        onset_samples = np.minimum(
            np.floor(times_s * fs).astype(np.int64), n_samples - 1
        )
        energies_eV = sample_bounded_power_law(
            rng, count, energy_min_ev, energy_max_ev, power_law_index
        )
        events.append({
            'onset_samples': onset_samples,
            'start_samples': onset_samples - int(pretrigger_samples),
            'energies_eV': energies_eV,
        })
    return events


def inject_lee_events(traces, scaled_templates, events):
    """Inject generated LEE events in floating-point current units."""
    if traces.ndim != 2 or scaled_templates.ndim != 2:
        raise ValueError('Traces and templates must be two-dimensional arrays')
    if len(events) != traces.shape[0] or scaled_templates.shape[0] != traces.shape[0]:
        raise ValueError('Events and templates must match the channel count')

    for channel_idx, channel_events in enumerate(events):
        start_samples = channel_events['start_samples']
        energies_eV = channel_events['energies_eV']
        if len(start_samples) != len(energies_eV):
            raise ValueError('Each LEE start sample must have an energy')
        for start_sample, energy_eV in zip(start_samples, energies_eV):
            add_scaled_pulse(
                traces[channel_idx], scaled_templates[channel_idx],
                int(start_sample), energy_eV
            )


LEE_YAML_TO_CLI = {
    'rate_hz': 'lee_rate_hz',
    'energy_min_ev': 'lee_energy_min_ev',
    'energy_max_ev': 'lee_energy_max_ev',
    'power_law_index': 'lee_power_law_index',
    'seed': 'lee_seed',
}
LEE_YAML_FIELDS = set(LEE_YAML_TO_CLI) | {'enabled'}


def _validate_finite_number(name, value, strictly_positive=False):
    """Return a finite float, with a descriptive configuration error."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a real number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a real number') from exc
    if not np.isfinite(number):
        raise ValueError(f'{name} must be finite')
    if strictly_positive and number <= 0:
        raise ValueError(f'{name} must be greater than zero')
    return number


def resolve_lee_settings(args):
    """Resolve native/legacy selection and CLI-over-YAML LEE settings."""
    yaml_config = {}
    if args.processing_setup is not None:
        yaml_config = load_yaml_config(args.processing_setup)
        if yaml_config is None:
            yaml_config = {}
        if not isinstance(yaml_config, dict):
            raise ValueError('Processing setup must contain a YAML mapping')

    native_indicated = 'lee' in yaml_config or any(
        getattr(args, cli_name, None) is not None
        for cli_name in LEE_YAML_TO_CLI.values()
    )
    yaml_lee = yaml_config.get('lee', {})
    if not isinstance(yaml_lee, dict):
        raise ValueError("The processing YAML 'lee' section must be a mapping")
    unknown_keys = set(yaml_lee) - LEE_YAML_FIELDS
    if unknown_keys:
        raise ValueError(
            'Unknown native LEE YAML parameter(s): '
            + ', '.join(sorted(unknown_keys))
        )
    if 'enabled' in yaml_lee and not isinstance(yaml_lee['enabled'], bool):
        raise ValueError('lee.enabled must be true or false')

    enabled = getattr(args, 'enable_LEE', None)
    if enabled is None:
        enabled = yaml_lee.get('enabled', False)
    native_enabled = native_indicated and enabled
    legacy_enabled = not native_indicated and enabled
    if legacy_enabled and args.processing_setup is None:
        raise ValueError(
            '--processing-setup is required for legacy PDF LEE injection'
        )

    settings = {}
    for yaml_name, cli_name in LEE_YAML_TO_CLI.items():
        value = getattr(args, cli_name, None)
        if value is None:
            value = yaml_lee.get(yaml_name)
        settings[yaml_name] = value

    for name in ('rate_hz', 'energy_min_ev', 'energy_max_ev', 'power_law_index'):
        if settings[name] is not None:
            settings[name] = _validate_finite_number(
                f'lee.{name}', settings[name],
                strictly_positive=name.startswith('energy_')
            )
    if settings['rate_hz'] is not None and settings['rate_hz'] < 0:
        raise ValueError('lee.rate_hz must be nonnegative')
    if (settings['energy_min_ev'] is not None
            and settings['energy_max_ev'] is not None
            and settings['energy_max_ev'] <= settings['energy_min_ev']):
        raise ValueError('lee.energy_max_ev must be greater than lee.energy_min_ev')

    seed = settings['seed']
    if seed is not None:
        if (not isinstance(seed, (int, np.integer))
                or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError('lee.seed must be a nonnegative integer')
        settings['seed'] = int(seed)

    if native_enabled:
        missing = [name for name in LEE_YAML_TO_CLI
                   if name != 'seed' and settings[name] is None]
        if missing:
            raise ValueError(
                'Native LEE is enabled but required parameter(s) are missing: '
                + ', '.join(missing)
            )

    return {
        **settings,
        'native_indicated': native_indicated,
        'native_enabled': native_enabled,
        'legacy_enabled': legacy_enabled,
    }


LEE_TRUTH_DTYPES = {
    'global_event_idx': 'u8',
    'series_num': 'u8',
    'event_in_series': 'u4',
    'channel_idx': 'u2',
    'onset_sample': 'i8',
    'energy_eV': 'f8',
    'event_start_epoch_s': 'f8',
}


class LEETruthWriter:
    """Write one extensible, per-pulse LEE truth table during generation."""

    def __init__(self, save_path, channel_names, fs, n_samples,
                 n_events_per_series, pretrigger_samples):
        salting_path = os.path.join(save_path, 'salting')
        os.makedirs(salting_path, exist_ok=True)
        self.path = os.path.join(salting_path, 'lee_truth.hdf5')
        self._file = h5py.File(self.path, 'x')
        self._group = self._file.create_group('lee_truth')
        self._n_rows = 0

        self._file.attrs['schema_version'] = 1
        self._file.attrs['fs_Hz'] = float(fs)
        self._file.attrs['event_length_samples'] = int(n_samples)
        self._file.attrs['n_events_per_series'] = int(n_events_per_series)
        self._file.attrs['pretrigger_samples'] = int(pretrigger_samples)
        self._file.attrs['onset_convention'] = (
            'sampled onset sample before template pretrigger subtraction'
        )
        self._file.attrs.create(
            'channel_names', np.asarray(channel_names, dtype=object),
            dtype=h5py.string_dtype('utf-8')
        )
        for name, dtype in LEE_TRUTH_DTYPES.items():
            self._group.create_dataset(
                name, shape=(0,), maxshape=(None,), chunks=True, dtype=dtype
            )

    @property
    def n_rows(self):
        return self._n_rows

    def append(self, global_event_idx, series_num, event_in_series,
               event_start_epoch_s, events):
        """Append all per-channel events for one generated waveform event."""
        channel_data = []
        for channel_idx, channel_events in enumerate(events):
            onsets = np.asarray(
                channel_events['onset_samples'], dtype=np.int64
            )
            energies = np.asarray(
                channel_events['energies_eV'], dtype=np.float64
            )
            if onsets.ndim != 1 or energies.shape != onsets.shape:
                raise ValueError('LEE onset and energy arrays must be aligned 1D arrays')
            channel_data.append((channel_idx, onsets, energies))

        counts = np.array([len(onsets) for _, onsets, _ in channel_data])
        n_rows = int(np.sum(counts))
        if n_rows == 0:
            return 0

        channel_indices = np.concatenate([
            np.full(count, channel_idx, dtype=np.uint16)
            for (channel_idx, _, _), count in zip(channel_data, counts)
        ])
        row_data = {
            'global_event_idx': np.full(n_rows, global_event_idx, dtype=np.uint64),
            'series_num': np.full(n_rows, series_num, dtype=np.uint64),
            'event_in_series': np.full(n_rows, event_in_series, dtype=np.uint32),
            'channel_idx': channel_indices,
            'onset_sample': np.concatenate([
                onsets for _, onsets, _ in channel_data
            ]),
            'energy_eV': np.concatenate([
                energies for _, _, energies in channel_data
            ]),
            'event_start_epoch_s': np.full(
                n_rows, event_start_epoch_s, dtype=np.float64
            ),
        }
        first = self._n_rows
        stop = first + n_rows
        for name, values in row_data.items():
            self._group[name].resize(stop, axis=0)
            self._group[name][first:stop] = values
        self._n_rows = stop
        return n_rows

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def traces_to_adc(traces):
    """Convert TES current traces to int16 ADC counts."""
    traces *= 1e6
    traces /= 10
    traces *= 2**16
    return np.clip(traces, -32768, 32767).astype(np.int16)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate TES traces from HeRM simulation H5 output')

    parser.add_argument('--herm-file', '--herm_file',
                        dest='herm_file', type=str, required=True,
                        help='Path to HeRM H5 file (from simulate_spectrum.py --save_raw)')
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
                        dest='comment', type=str, default='HeRM trace generation',
                        help='Metadata comment string')
    parser.add_argument('--filesize-lim-GB', '--filesize_lim_GB',
                        dest='filesize_lim_GB', type=float, default=1,
                        help='Max output file size in GB (default: 1)')
    parser.add_argument('--no-noise', '--no_noise',
                        dest='no_noise', action='store_true',
                        help='Skip noise generation (signal-only traces)')
    parser.add_argument('--trigger-offset-msec', '--trigger_offset_msec',
                        dest='trigger_offset_msec', type=float, default=5000,
                        help='Time in trace (ms) where HeRM t=0 maps to (default: 5000)')
    parser.add_argument('--template-tag', '--template_tag',
                        dest='template_tag', type=str, default=None,
                        help='Template tag in filter file (default: channel name)')
    parser.add_argument('--noise-tag', '--noise_tag',
                        dest='noise_tag', type=str, default=None,
                        help='PSD/CSD tag in filter file (default: channel name)')
    parser.add_argument('--uniform-channel', '--uniform_channel',
                        dest='uniform_channel', type=str, default=None,
                        help='Use this single channel template+PSD for all sensors')
    parser.add_argument('--dpdi-tag', '--dpdi_tag',
                        dest='dpdi_tag', type=str, default=None,
                        help='dP/dI tag in filter file (default: channel name)')
    parser.add_argument('--dpdi-poles', '--dpdi_poles',
                        dest='dpdi_poles', type=int, default=1,
                        help='Number of poles for dP/dI lookup (default: 1)')
    parser.add_argument('--max-events', '--max_events',
                        dest='max_events', type=int, default=None,
                        help='Maximum number of events to process (default: all)')
    lee_toggle = parser.add_mutually_exclusive_group()
    lee_toggle.add_argument(
        '--enable-LEE', '--enable_LEE', dest='enable_LEE',
        action='store_true', default=None,
        help='Enable native power-law or legacy PDF LEE injection')
    lee_toggle.add_argument(
        '--disable-LEE', '--disable_LEE', dest='enable_LEE',
        action='store_false', default=None,
        help='Disable LEE, including when enabled in the processing YAML')
    for cli_name, dest, yaml_name, help_text in [
        ('--lee-rate-hz', 'lee_rate_hz', 'rate_hz',
         'Native LEE rate in events/second/channel'),
        ('--lee-energy-min-ev', 'lee_energy_min_ev', 'energy_min_ev',
         'Native LEE minimum energy in eV'),
        ('--lee-energy-max-ev', 'lee_energy_max_ev', 'energy_max_ev',
         'Native LEE maximum energy in eV'),
        ('--lee-power-law-index', 'lee_power_law_index', 'power_law_index',
         'Native LEE alpha in dN/dE proportional to E^-alpha'),
    ]:
        parser.add_argument(
            cli_name, f'--{dest}', dest=dest,
            type=float, default=None,
            help=f'{help_text} (overrides YAML lee.{yaml_name})'
        )
    parser.add_argument(
        '--lee-seed', '--lee_seed', dest='lee_seed', type=int, default=None,
        help='Nonnegative seed for native LEE only (overrides YAML lee.seed)')
    parser.add_argument('--enable-background', '--enable_background',
                        dest='enable_background', action='store_true',
                        help='Inject Geant4 backgrounds after trace generation')
    parser.add_argument('--processing-setup', '--processing_setup',
                        dest='processing_setup', type=str, default=None,
                        help='Processing YAML with native lee or legacy salting config')
    parser.add_argument('--topology-setup', '--topology_setup',
                        dest='topology_setup', type=str, default=None,
                        help='Topology YAML file path (required for --enable-background)')

    return parser


def main():
    args = build_parser().parse_args()
    lee_settings = resolve_lee_settings(args)

    if args.enable_background and args.topology_setup is None:
        raise ValueError('--topology-setup is required when --enable-background is set')

    # --- Derived parameters ---
    n_samples = int(args.fs * args.event_length_sec)
    trigger_offset = int(args.fs * args.trigger_offset_msec * 1e-3)

    # n_channels derived from HeRM H5 after opening (see below)
    # channel_names set after n_channels is known

    date_time_string = time_to_string(dt.datetime.now())
    save_path = args.save_path.rstrip('/') + '/continuous_I9_' + date_time_string
    os.makedirs(save_path, exist_ok=True)

    events_per_series = args.series_length_sec / args.event_length_sec
    if not np.isclose(events_per_series, round(events_per_series)):
        raise ValueError(
            f'Event length ({args.event_length_sec}s) does not evenly divide '
            f'series length ({args.series_length_sec}s)')
    n_events_per_series = round(events_per_series)

    # --- Open HeRM H5 file ---
    print(f'Opening HeRM file: {args.herm_file}')
    herm_h5 = h5py.File(args.herm_file, 'r')
    events_grp = herm_h5['events']
    n_events = int(herm_h5.attrs['n_events'])
    n_channels = int(herm_h5.attrs['n_sensors'])
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f'  {n_events} events to process, {n_channels} sensors/channels')

    channel_names = [f'channel_{i:02d}' for i in range(n_channels)]

    # --- Load templates ---
    print('Loading templates and computing energy normalization...')
    filter_data = FilterData()
    filter_data.load_hdf5(args.filter_file)

    scaled_templates, pretrigger_samples = load_scaled_templates(
        filter_data, channel_names, args.template_tag, args.dpdi_tag,
        args.dpdi_poles, args.fs, n_channels, uniform_channel=args.uniform_channel
    )
    print(f'  Template length: {scaled_templates.shape[1]} samples')
    print(f'  Pretrigger: {pretrigger_samples} samples ({pretrigger_samples/args.fs*1000:.1f} ms)')

    lee_rng = None
    if lee_settings['native_enabled']:
        lee_rng = np.random.default_rng(lee_settings['seed'])
        print(
            'Native LEE: '
            f'{lee_settings["rate_hz"]:g} events/s/channel, '
            f'E in [{lee_settings["energy_min_ev"]:g}, '
            f'{lee_settings["energy_max_ev"]:g}] eV, '
            f'alpha={lee_settings["power_law_index"]:g}, '
            f'seed={lee_settings["seed"]}'
        )

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
            print(f'  Initialized {n_channels} independent noise channels')

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

    lee_truth_writer = None
    if lee_settings['native_enabled']:
        lee_truth_writer = LEETruthWriter(
            save_path, channel_names, args.fs, n_samples,
            n_events_per_series, pretrigger_samples
        )
    total_lee_pulses = 0

    try:
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
                    n_channels, comment=args.comment, fs=args.fs,
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
                file_metadata['dump_num'] += 1
                writer.set_metadata(
                    file_metadata=file_metadata, adc_config=adc_metadata,
                    detector_config=det_metadata
                )
                writer._open_file(prefix='cont')

            # Build trace
            traces = np.zeros((n_channels, n_samples))

            # Add noise
            if enable_noise:
                if args.uniform_channel is not None:
                    for i in range(n_channels):
                        traces[i] = noise_factory.sample_noise(args.uniform_channel)
                else:
                    for i, chan in enumerate(channel_names):
                        traces[i] = noise_factory.sample_noise(chan)

            # Add signal pulses from HeRM
            evt_key = str(event_idx)
            if evt_key in events_grp:
                inject_pulses(
                    traces, events_grp[evt_key], scaled_templates,
                    trigger_offset, pretrigger_samples, args.fs, n_channels
                )

            if lee_settings['native_enabled']:
                lee_events = generate_lee_events(
                    n_channels, n_samples, args.fs,
                    lee_settings['rate_hz'], lee_settings['energy_min_ev'],
                    lee_settings['energy_max_ev'],
                    lee_settings['power_law_index'], pretrigger_samples, lee_rng
                )
                inject_lee_events(traces, scaled_templates, lee_events)

            # Convert to ADC and write once, after all floating-point injection
            data = traces_to_adc(traces)
            writer.write_event(data, start_epoch + time_counter)
            if lee_truth_writer is not None:
                total_lee_pulses += lee_truth_writer.append(
                    global_event_idx=event_idx,
                    series_num=int(file_metadata['series_num']),
                    event_in_series=event_idx % n_events_per_series,
                    event_start_epoch_s=start_epoch + time_counter,
                    events=lee_events
                )
            time_counter += args.event_length_sec
    finally:
        if lee_truth_writer is not None:
            lee_truth_writer.close()

    # Close final file
    writer._close_file()
    herm_h5.close()

    if lee_truth_writer is not None:
        print(f'  Injected LEE pulses: {total_lee_pulses}')
        print(f'  LEE truth: {lee_truth_writer.path}')

    total_time_sec = n_events * args.event_length_sec
    channel_list = channel_names

    #########################################
    # Setting up the Salting class for LEE  #
    #########################################

    if lee_settings['legacy_enabled']:
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
