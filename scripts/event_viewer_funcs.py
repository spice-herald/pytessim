import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import h5py
from pytesdaq.io import H5Reader

matplotlib.rcParams['figure.figsize'] = 8.5,6
matplotlib.rcParams['figure.subplot.left'] = 0.15
matplotlib.rcParams['figure.subplot.right'] = 0.88
matplotlib.rcParams['figure.subplot.bottom'] = 0.15
matplotlib.rcParams['figure.subplot.top'] = 0.88
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['font.weight'] = 'normal' 
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['xtick.minor.visible'] = True 
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
matplotlib.rcParams['xtick.top'] = True 
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.major.pad'] = 6
matplotlib.rcParams['image.cmap'] = 'viridis'


LEE_TRUTH_FIELDS = (
    'global_event_idx', 'series_num', 'event_in_series', 'channel_idx',
    'onset_sample', 'energy_eV', 'event_start_epoch_s',
)
LEE_MARKER_COLOR = 'C5'
LEE_MARKER_STYLE = '--'


class EventLoader:
    """Load events on demand with caching and optional LEE truth overlays."""

    def __init__(self, trace_path, hest_h5=None, fs=1.25e6, max_cache=5,
                 lee_h5=None, show_lee=True):
        """
        trace_path: path (or glob pattern) to the trace H5 files.
        hest_h5: Optional HeRM/HeST truth H5 path or open h5py.File.
        fs: sample rate in Hz (used for time-axis conversion).
        lee_h5: native LEE truth path. None auto-discovers
            <trace_path>/salting/lee_truth.hdf5 for a directory trace_path.
        show_lee: display LEE onset markers and tables when truth is available.
        """
        self.trace_path = os.fspath(trace_path)
        self.max_cache = max_cache
        self.fs = fs
        self.show_lee = bool(show_lee)
        self._cache = {}
        self._event_series = []

        if lee_h5 is None and isinstance(self.trace_path, (str, bytes)):
            from pathlib import Path
            trace_dir = Path(self.trace_path)
            if trace_dir.is_dir():
                candidate = trace_dir / 'salting' / 'lee_truth.hdf5'
                if candidate.is_file():
                    lee_h5 = candidate
        self.lee_h5 = str(lee_h5) if lee_h5 is not None else None
        self.lee_data = None
        if lee_h5 is not None:
            try:
                self.lee_data = (lee_h5 if isinstance(lee_h5, h5py.File)
                                 else h5py.File(lee_h5, 'r'))
            except (OSError, TypeError) as exc:
                raise ValueError(f'Cannot open LEE truth file: {lee_h5}') from exc

        self.hest_data = None
        if hest_h5 is not None:
            self.hest_data = hest_h5 if isinstance(hest_h5, h5py.File) else h5py.File(hest_h5, 'r')
        # Count events (read but don't store) and retain series identity.
        reader = H5Reader()
        reader.set_files(self.trace_path)
        self.n_events = 0
        self.n_channels = None
        while True:
            traces, meta = reader.read_next_event(include_metadata=True, adctoamp=False)
            if meta['error_msg'] == 'No more files available':
                break
            if self.n_channels is None:
                self.n_channels = traces.shape[0]
            if self.lee_data is not None:
                self._event_series.append(int(meta['series_num']))
            self.n_events += 1
        reader.close()
        if self.lee_data is not None:
            self._validate_lee_truth()

    def _validate_lee_truth(self):
        """Validate native LEE schema, sample rate, and channel mapping."""
        attrs = self.lee_data.attrs
        if attrs.get('schema_version') != 1:
            raise ValueError('LEE truth must have schema_version = 1')
        if 'lee_truth' not in self.lee_data:
            raise ValueError('LEE truth file is missing the /lee_truth group')
        group = self.lee_data['lee_truth']
        missing = set(LEE_TRUTH_FIELDS) - set(group)
        if missing:
            raise ValueError('LEE truth is missing field(s): '
                             + ', '.join(sorted(missing)))
        lengths = set()
        for name in LEE_TRUTH_FIELDS:
            if group[name].ndim != 1:
                raise ValueError(f'LEE truth field {name} must be one-dimensional')
            lengths.add(len(group[name]))
        if len(lengths) != 1:
            raise ValueError('LEE truth datasets must have equal lengths')
        if not np.isclose(attrs.get('fs_Hz', np.nan), self.fs, rtol=0, atol=0):
            raise ValueError(
                f'LEE truth fs_Hz ({attrs.get("fs_Hz")}) does not match '
                f'viewer fs ({self.fs})'
            )
        names = attrs.get('channel_names', [])
        if len(names) != self.n_channels:
            raise ValueError('LEE truth channel count does not match raw traces')

    def _empty_lee_truth(self):
        if self.lee_data is None:
            return {name: np.array([]) for name in LEE_TRUTH_FIELDS}
        group = self.lee_data['lee_truth']
        return {
            name: np.array([], dtype=group[name].dtype)
            for name in LEE_TRUTH_FIELDS
        }

    def get_lee_truth(self, event_idx, channel_idx=None):
        """Return stored LEE pulses for one global event and optional channel."""
        if self.lee_data is None:
            return self._empty_lee_truth()
        if not isinstance(event_idx, (int, np.integer)) or event_idx < 0:
            raise IndexError('event_idx must be a nonnegative integer')
        if event_idx >= self.n_events:
            raise IndexError(f'Event {event_idx} out of range')
        group = self.lee_data['lee_truth']
        event_mask = group['global_event_idx'][:] == event_idx
        selected_series = np.unique(group['series_num'][:][event_mask])
        if len(selected_series) > 1:
            raise ValueError(f'LEE truth event {event_idx} spans multiple raw series')
        if len(selected_series) == 1:
            raw_series = self._event_series[event_idx]
            if int(selected_series[0]) != raw_series:
                raise ValueError(
                    f'LEE truth series {int(selected_series[0])} does not match '
                    f'raw event {event_idx} series {raw_series}'
                )
        if channel_idx is not None:
            if (not isinstance(channel_idx, (int, np.integer))
                    or not 0 <= channel_idx < self.n_channels):
                raise ValueError(f'Invalid channel index: {channel_idx}')
            event_mask &= group['channel_idx'][:] == channel_idx
        result = {name: group[name][:][event_mask] for name in LEE_TRUTH_FIELDS}
        order = np.argsort(result['onset_sample'], kind='stable')
        return {name: values[order] for name, values in result.items()}

    def _add_lee_markers(self, ax, event_idx, channel_idx, max_markers=100):
        """Draw exact LEE onset times, independently of downsampling/HeRM offsets."""
        if self.lee_data is None or not self.show_lee:
            return None
        truth = self.get_lee_truth(event_idx, channel_idx)
        n_pulses = len(truth['onset_sample'])
        if n_pulses == 0:
            return truth
        times_ms = truth['onset_sample'] / self.fs * 1e3
        if n_pulses > max_markers:
            print(f'LEE channel {channel_idx}: drawing {max_markers} of '
                  f'{n_pulses} markers; pulse count includes all pulses')
        for pulse_idx, time_ms in enumerate(times_ms[:max_markers]):
            ax.axvline(
                time_ms, color=LEE_MARKER_COLOR, linestyle=LEE_MARKER_STYLE,
                alpha=0.8, lw=1.2,
                label=(f'LEE ({n_pulses} pulses, '
                       f'{np.sum(truth["energy_eV"]):.3g} eV)')
                if pulse_idx == 0 else None
            )
        return truth

    def _print_lee_truth(self, event_idx, channel_idx, max_rows=50):
        """Print exact injected onset and energy values for a selected channel."""
        if self.lee_data is None or not self.show_lee:
            return
        truth = self.get_lee_truth(event_idx, channel_idx)
        n_pulses = len(truth['onset_sample'])
        print(f'LEE event {event_idx}, channel {channel_idx}: {n_pulses} pulses')
        if n_pulses == 0:
            return
        if n_pulses > max_rows:
            print(f'  Showing first {max_rows} of {n_pulses} pulses')
        print(f'  {"Pulse":>5} {"Onset sample":>13} {"Time (ms)":>13} {"Energy (eV)":>14}')
        print('  ' + '-' * 50)
        for pulse_idx in range(min(n_pulses, max_rows)):
            onset = truth['onset_sample'][pulse_idx]
            time_ms = onset / self.fs * 1e3
            print(f'  {pulse_idx + 1:>5} {onset:>13d} {time_ms:>13.6f} '
                  f'{truth["energy_eV"][pulse_idx]:>14.6f}')


    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        reader = H5Reader()
        reader.set_files(self.trace_path)
        for _ in range(idx + 1):
            traces, meta = reader.read_next_event(include_metadata=True, adctoamp=True)
            if meta['error_msg'] == 'No more files available':
                reader.close()
                raise IndexError(f'Event {idx} out of range')
        reader.close()
        if len(self._cache) >= self.max_cache:
            del self._cache[min(self._cache)]
        self._cache[idx] = (traces, meta)
        return traces, meta

    def downsample(self, trace, factor=None):
        """
        Downsample a trace.
        factor=None: no downsampling (full resolution).
        factor=N: take every Nth sample.
        """
        if factor is None or factor <= 1:
            return np.arange(len(trace)), trace
        idx = np.arange(0, len(trace), factor)
        return idx, trace[idx]


    def get_hest_truth(self, event_idx):
        """Get per-sensor energy and recoil info from HeST H5."""
        if self.hest_data is None:
            return None
        evt_key = str(event_idx)
        if evt_key not in self.hest_data['events']:
            return None
        evt = self.hest_data['events'][evt_key]
        sensor_energies = np.zeros(self.n_channels)
        for ch_key in ['singlet', 'triplet', 'ir', 'qp']:
            if ch_key not in evt:
                continue
            weight = evt[ch_key].attrs.get('weight', 1.0)
            for s in range(self.n_channels):
                if str(s) in evt[ch_key]:
                    sensor_energies[s] += np.sum(evt[ch_key][str(s)]['energies'][:]) * weight
        return {
            'recoil_energy': evt.attrs['recoil_energy'],
            'recoil_type': evt.attrs['recoil_type'],
            'sensor_energies': sensor_energies,
        }

    def plot_all_channels(self, event_idx=0, ds_factor=10, xlim=None, ylim=None, ncols=4):
        """
        Plot all channel traces for one event.
        ds_factor: downsample factor (None=full res, 2=every 2nd sample, etc.)
        xlim: (xmin, xmax) in ms, e.g. (4990, 5010). None=full trace.
        ylim: (ymin, ymax) in µA. None=auto.
        ncols: number of columns in the subplot grid (1 = stacked vertically).
        """
        traces, _ = self[event_idx]
        n_ch = traces.shape[0]

        truth = self.get_hest_truth(event_idx)
        title = f'Event {event_idx}'
        if truth:
            title += f' — {truth["recoil_type"]}, E = {truth["recoil_energy"]:.1f} eV'

        ncols = min(ncols, n_ch)
        nrows = int(np.ceil(n_ch / ncols))
        figw = 4 * ncols if ncols > 1 else 10
        fig, axes = plt.subplots(nrows, ncols, figsize=(figw, 3 * nrows), sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=14)

        for ch in range(n_ch):
            ax = axes[ch // ncols, ch % ncols]
            idx, vals = self.downsample(traces[ch], factor=ds_factor)
            ax.plot(idx / self.fs * 1e3, vals * 1e6, linewidth=2, color='C0')
            lee_truth = self._add_lee_markers(ax, event_idx, ch)
            if lee_truth is not None and len(lee_truth['onset_sample']):
                ax.legend(fontsize=7, loc='upper right')
            ax.set_title(f'ch {ch:02d}', fontsize=8)
            ax.tick_params(labelsize=7)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

        for i in range(n_ch, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel('Time (ms)')
        fig.text(0.04, 0.5, 'Current (µA)', va='center', rotation='vertical', fontsize=12)
        plt.tight_layout(rect=[0.05, 0, 1, 0.97])
        plt.show()

    def plot_sensor_with_truth(self, event_idx=0, sensor_idx=0, trigger_offset_ms=5000,
                           ds_factor=None, xlim=None, ylim=None):
        """
        Plot one sensor's trace with HeST truth overlay.
        ds_factor: downsample factor (None=full res, 2=every 2nd, etc.)
        xlim: (xmin, xmax) in ms, e.g. (4999, 5005). None=full trace.
        ylim: (ymin, ymax) in µA. None=auto.
        """
        traces, _ = self[event_idx]
        trace = traces[sensor_idx]
        self._print_lee_truth(event_idx, sensor_idx)

        fig, ax = plt.subplots(figsize=(14, 5))

        idx, vals = self.downsample(trace, factor=ds_factor)
        ax.plot(idx / self.fs * 1e3, vals * 1e6, linewidth=2, color='C0')
        self._add_lee_markers(ax, event_idx, sensor_idx)

        # HeST/HeRM truth overlay (LEE uses its stored onset, without this offset)
        colors = {'singlet': 'C1', 'triplet': 'C2', 'ir': 'C3', 'qp': 'C4'}
        if self.hest_data is not None:
            evt_key = str(event_idx)
            if evt_key in self.hest_data['events']:
                evt = self.hest_data['events'][evt_key]
                print(f'Event {event_idx}: {evt.attrs["recoil_type"]} recoil, '
                    f'E = {evt.attrs["recoil_energy"]:.2f} eV')
                print(f'Sensor {sensor_idx}:')
                print(f'  {"Channel":<10} {"#hits":>6} {"E_tot (eV)":>11} '
                    f'{"t_mean (µs)":>12} {"t_min (µs)":>11} {"t_max (µs)":>11}')
                print('  ' + '-' * 66)

                for ch_key in ['singlet', 'triplet', 'ir', 'qp']:
                    if ch_key not in evt:
                        continue
                    ch_grp = evt[ch_key]
                    s_key = str(sensor_idx)
                    if s_key not in ch_grp:
                        continue
                    weight = ch_grp.attrs.get('weight', 1.0)
                    times_us = ch_grp[s_key]['arrival_times'][:]
                    energies = ch_grp[s_key]['energies'][:] * weight
                    n_hits = len(times_us)
                    total_e = np.sum(energies)

                    print(f'  {ch_key:<10} {n_hits:>6} {total_e:>11.4f} '
                        f'{np.mean(times_us):>12.2f} {np.min(times_us):>11.2f} '
                        f'{np.max(times_us):>11.2f}')

                    hit_ms = times_us * 1e-3 + trigger_offset_ms
                    for ht in hit_ms[:50]:
                        ax.axvline(ht, color=colors[ch_key], alpha=0.3, lw=0.5)
                    ax.axvline(np.nan, color=colors[ch_key], alpha=0.7, lw=1.5,
                            label=f'{ch_key} ({n_hits}, {total_e:.2f} eV)')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (µA)')
        ax.set_title(f'Event {event_idx} — Sensor {sensor_idx}')
        ax.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_all_sensors_with_truth(self, event_idx=0, ds_factor=10, xlim=None, ylim=None,
                                    ncols=1, trigger_offset_ms=5000):
        """
        Plot all sensor traces for one event with HeST truth hits overlaid.
        Each panel shows one sensor's trace with vertical lines at hit arrival
        times, colored by channel type (singlet/triplet/ir/qp). Panels where
        the sensor has no hits show only the trace. A shared legend lists
        channel types present in the event, with hit count and total energy.

        ds_factor: downsample factor (None=full res, 2=every 2nd sample, etc.)
        xlim: (xmin, xmax) in ms, e.g. (4999, 5002). None=full trace.
        ylim: (ymin, ymax) in µA. None=auto.
        ncols: number of columns in the subplot grid (1 = stacked vertically).
        trigger_offset_ms: offset added to truth arrival times to align with
            the trace time axis (same convention as plot_sensor_with_truth).
        """
        traces, _ = self[event_idx]
        n_ch = traces.shape[0]

        colors = {'singlet': 'C1', 'triplet': 'C2', 'ir': 'C3', 'qp': 'C4'}

        # Gather truth hits per sensor: {sensor: {ch_key: (times_us, energies)}}
        truth_hits = {s: {} for s in range(n_ch)}
        title = f'Event {event_idx}'
        if self.hest_data is not None:
            evt_key = str(event_idx)
            if evt_key in self.hest_data['events']:
                evt = self.hest_data['events'][evt_key]
                title += (f' — {evt.attrs["recoil_type"]}, '
                          f'E = {evt.attrs["recoil_energy"]:.1f} eV')
                for ch_key in colors:
                    if ch_key not in evt:
                        continue
                    weight = evt[ch_key].attrs.get('weight', 1.0)
                    for s in range(n_ch):
                        s_key = str(s)
                        if s_key in evt[ch_key]:
                            times_us = evt[ch_key][s_key]['arrival_times'][:]
                            energies = evt[ch_key][s_key]['energies'][:] * weight
                            truth_hits[s][ch_key] = (times_us, energies)

        ncols = min(ncols, n_ch)
        nrows = int(np.ceil(n_ch / ncols))
        figw = 4 * ncols if ncols > 1 else 10
        fig, axes = plt.subplots(nrows, ncols, figsize=(figw, 3 * nrows),
                                 sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=14)

        for ch in range(n_ch):
            ax = axes[ch // ncols, ch % ncols]
            idx, vals = self.downsample(traces[ch], factor=ds_factor)
            ax.plot(idx / self.fs * 1e3, vals * 1e6, linewidth=2, color='C0')
            self._add_lee_markers(ax, event_idx, ch)

            panel_title = f'ch {ch:02d}'
            for ch_key, (times_us, energies) in truth_hits[ch].items():
                hit_ms = times_us * 1e-3 + trigger_offset_ms
                for ht in hit_ms[:100]:
                    ax.axvline(ht, color=colors[ch_key], alpha=0.4, lw=0.7)
                panel_title += f' {ch_key[0]}:{np.sum(energies):.2f}'
            ax.set_title(panel_title, fontsize=8)
            ax.tick_params(labelsize=7)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

        for i in range(n_ch, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        # Shared legend: one entry per channel type present in the event
        handles = []
        if self.lee_data is not None and self.show_lee:
            lee_truth = self.get_lee_truth(event_idx)
            n_lee = len(lee_truth['onset_sample'])
            if n_lee:
                handles.append(plt.Line2D(
                    [], [], color=LEE_MARKER_COLOR, linestyle=LEE_MARKER_STYLE,
                    lw=1.5, label=f'LEE ({n_lee} pulses, '
                                  f'{np.sum(lee_truth["energy_eV"]):.3g} eV)'
                ))
        for ch_key in colors:
            n_hits = sum(len(v[ch_key][0]) for v in truth_hits.values()
                         if ch_key in v)
            if n_hits:
                tot_e = sum(np.sum(v[ch_key][1]) for v in truth_hits.values()
                            if ch_key in v)
                handles.append(plt.Line2D([], [], color=colors[ch_key], lw=2,
                                          label=f'{ch_key} ({n_hits} hits, '
                                                f'{tot_e:.2f} eV)'))
        if handles:
            fig.legend(handles=handles, fontsize=12, loc='upper right',
                       bbox_to_anchor=(0.99, 0.97))

        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel('Time (ms)')
        fig.text(0.04, 0.5, 'Current (µA)', va='center', rotation='vertical',
                 fontsize=12)
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        plt.show()

    def plot_heatmap(self, event_idx=0, use_truth=False):
        """Bar plot of per-sensor amplitude (µA) or truth energy (eV)."""
        if use_truth:
            truth = self.get_hest_truth(event_idx)
            if truth is None:
                print('No HeST truth data')
                return
            values = truth['sensor_energies']
            label = 'Energy (eV)'
            extra = f' [TRUTH: {truth["recoil_type"]}, E={truth["recoil_energy"]:.1f} eV]'
        else:
            traces, _ = self[event_idx]
            n_ch = traces.shape[0]
            values = np.zeros(n_ch)
            for i in range(n_ch):
                bl = np.median(traces[i][:len(traces[i]) // 10])
                values[i] = np.max(np.abs(traces[i] - bl)) * 1e6
            label = 'Amplitude (µA)'
            extra = ''

        n_ch = len(values)
        fig, ax = plt.subplots(figsize=(max(6, n_ch * 0.8), 3))
        vmax = max(np.max(values), 1e-12)

        bar_colors = plt.cm.inferno(values / vmax)
        ax.bar(range(n_ch), values, color=bar_colors, edgecolor='gray', linewidth=0.5)
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels([f'{i}' for i in range(n_ch)])
        ax.set_xlabel('Sensor')
        ax.set_ylabel(label)
        ax.set_title(f'Event {event_idx}{extra}')
        plt.tight_layout()
        plt.show()