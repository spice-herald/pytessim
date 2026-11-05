import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import h5py
from pytesdaq.io import H5Reader

class EventLoader:
    """Load events on demand with caching. No bulk loading."""

    def __init__(self, trace_path, hest_h5=None, fs=1.25e6, max_cache=5):
        """
        trace_path: path (or glob pattern) to the trace H5 files.
        hest_h5: HeST truth H5 path or an already-open h5py.File. None = no truth.
        fs: sample rate in Hz (used for time-axis conversion).
        """
        self.trace_path = trace_path
        self.max_cache = max_cache
        self.fs = fs
        self._cache = {}

        self.hest_data = None
        if hest_h5 is not None:
            self.hest_data = hest_h5 if isinstance(hest_h5, h5py.File) else h5py.File(hest_h5, 'r')
        # Count events (read but don't store)
        reader = H5Reader()
        reader.set_files(trace_path)
        self.n_events = 0
        self.n_channels = None
        while True:
            traces, meta = reader.read_next_event(include_metadata=True, adctoamp=False)
            if meta['error_msg'] == 'No more files available':
                break
            if self.n_channels is None:
                self.n_channels = traces.shape[0]
            self.n_events += 1
        reader.close()

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

    def plot_all_channels(self, event_idx=0, ds_factor=10, xlim=None, ylim=None):
        """
        Plot all channel traces for one event.
        ds_factor: downsample factor (None=full res, 2=every 2nd sample, etc.)
        xlim: (xmin, xmax) in ms, e.g. (4990, 5010). None=full trace.
        ylim: (ymin, ymax) in µA. None=auto.
        """
        traces, _ = self[event_idx]
        n_ch = traces.shape[0]

        truth = self.get_hest_truth(event_idx)
        title = f'Event {event_idx}'
        if truth:
            title += f' — {truth["recoil_type"]}, E = {truth["recoil_energy"]:.1f} eV'

        ncols = min(4, n_ch)
        nrows = int(np.ceil(n_ch / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=14)

        for ch in range(n_ch):
            ax = axes[ch // ncols, ch % ncols]
            idx, vals = self.downsample(traces[ch], factor=ds_factor)
            ax.plot(idx / self.fs * 1e3, vals * 1e6, linewidth=0.3, color='C0')
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

        fig, ax = plt.subplots(figsize=(14, 5))

        idx, vals = self.downsample(trace, factor=ds_factor)
        ax.plot(idx / self.fs * 1e3, vals * 1e6, linewidth=0.5, color='C0')

        # Truth overlay
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
        ax.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
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