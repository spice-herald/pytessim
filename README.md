# pytessim

Simulation of TES (Transition Edge Sensor) datastreams for TESSERACT dark matter sensitivity projections. Generates realistic fake TES data by layering noise (from PSDs/CSDs), low-energy excess (LEE) pulses, and Geant4 signal backgrounds, then writes continuous-acquisition HDF5 files in pytesdaq format compatible with `detprocess` and `qetpy`.

## Installation

Requires TESSERACT-internal packages: [detprocess](https://github.com/spice-herald/detprocess), [pytesdaq](https://github.com/spice-herald/pytesdaq), [pytesio](https://github.com/spice-herald/pytesio), [qetpy](https://github.com/spice-herald/qetpy). Also needs `numpy`, `scipy`, `pandas`, `pyyaml`, `uproot`, `vaex`, `matplotlib`, `tqdm`, `cloudpickle`.

```bash
cd pytessim
pip install -e .
```

Pure Python — no compiled code.

## Detector Topologies

| Topology | Channels | Description |
|---|---|---|
| `PPD_1ch` | 1 | Single-channel phonon detector |
| `PPD_2ch` | 2 | Two-channel PPD with correlated noise |
| `GaAs_1ch` | 3 | GaAs target + 2 Si light detectors |
| `HeRALD_Modane` | 24 | HeRALD helium detector |

## Generic Data Generation (`file_generation.py`)

Generates noise + LEE + Geant4 background traces for any supported topology. Requires a filter file (template, PSD, dP/dI) and processing/topology YAML configs.

```bash
# 1. Create filter file + LEE distributions: run examples/<topology>/ffgen.ipynb

# 2. Generate data
python scripts/file_generation.py \
    --time 180 --nchannels 1 \
    --event-length-sec 10 --series-length-sec 60 \
    --processing-setup examples/1ch_PPD/PPD_1ch.yaml \
    --template-file 1channel_ppd.hdf5 \
    --save-path ./output/ --comment test \
    --enable-noise --enable-LEE --enable-background \
    --topology-setup examples/1ch_PPD/PPD_1ch_topo.yaml
```

Three worked examples in `examples/`: 1-channel PPD, 2-channel PPD, and GaAs sandwich.

### Configuration Files

**Processing YAML** — controls noise and LEE injection:

```yaml
filter_file: 1channel_ppd.hdf5
LEE_PDF_file: LEE_distributions.pkl
noise:
  channel_00:
    noise_tag: channel_00
salting:
  channel_00:
    template_tag: channel_00
    collection_efficiency: 1.0
    pdf_tag: singles
    pdf_bounds: [0, 100]
    dpdi_tag: channel_00
    dpdi_poles: 1
    rate: 0.5
```

**Topology YAML** — maps Geant4 volumes to detector factories:

```yaml
filter_file: 1channel_ppd.hdf5
background_file: Copper_Co57_germanium_reduced.root
topology:
  target_0:
    target_type: PPD_1ch
    G4ID: 0
    channel:
      channel_name: channel_00
      template_tag: channel_00
      dpdi_tag: channel_00
      dpdi_npoles: 1
      collection_efficiency: 1.0
```

## HeST Trace Generation (`hest_trace_generation.py`)

Converts raw HeST simulation output (per-quanta arrival times + energies on 24 sensors) into realistic 24-channel TES traces. Reads a HeST H5 file (from `simulate_spectrum.py --save_raw`), convolves hits with a TES pulse template, optionally adds noise, and writes pytesdaq-format HDF5.

### Quick Start

**1. Create a filter file** with `scripts/create_filterFile.py` (or provide your own):

```bash
python scripts/create_filterFile.py   # writes inputs/herald_filter.hdf5
```

The dP/dI prefactor (`A0`) controls pulse height. Verify with:

```python
import qetpy as qp
energy_norm = qp.get_energy_normalization(time_array, template, dpdi=dpdi, lgc_ev=True)
print(f'16 eV singlet peak: {16 / energy_norm * 1e6:.4f} µA')  # target 0.5–2 µA; ADC clips at ±5 µA
```

**2. Run HeST** (in the HeST repository):

```bash
python simulate_spectrum.py \
    --spectrum flat --energy-min 10 --energy-max 10000 \
    --n-events 100 --save_raw --output sims/my_simulation
```

The `--save_raw` flag writes `events/{i}/{channel}/{sensor}/arrival_times` (µs) and `energies` (eV), where `channel` is `singlet`, `triplet`, `ir`, or `qp`.

**3. Generate traces:**

```bash
python scripts/hest_trace_generation.py \
    --hest-file sims/Spectrum_Flat_ER_...h5 \
    --filter-file inputs/herald_filter.hdf5 \
    --save-path ./traces_out/ \
    --event-length-sec 10 \
    --uniform-channel channel_00 \
    --dpdi-poles 3
```

Key options: `--no-noise` for signal-only traces, `--max-events N` to limit events, and `--uniform-channel channel_00` to apply one channel's template/PSD to all sensors. HeRM and HeST refer to the same per-quanta input format used by `herm_trace_generation.py`.

### Native LEE injection

Native LEE pulses are injected in floating-point current before the single ADC conversion. Each channel uses the energy-normalized impulse template selected for that channel from `--filter-file`, including the existing `--template-tag`, `--dpdi-tag`, `--dpdi-poles`, and `--uniform-channel` settings. For an event of energy **E eV**, the added current is **E × (current/eV template)**. HeRM quantum weights and legacy collection efficiency are not applied to LEE.

Add a top-level `lee:` section to the YAML supplied with `--processing-setup`:

```yaml
lee:
  enabled: true
  rate_hz: 100.0
  energy_min_ev: 1.0
  energy_max_ev: 100.0
  power_law_index: 2.0
  seed: 42
```

These are example values, not built-in physics defaults. The rate is in **events/second/channel**, the energy bounds are in **eV**, and the spectrum is bounded with **dN/dE ∝ E^(-power_law_index)**. An independent Poisson stream is generated for every channel. Onset times are uniform over each trace's duration; pulses may overlap and are clipped at trace edges.

The same run can be configured entirely on the command line, without a processing YAML:

```bash
python scripts/herm_trace_generation.py \
    --herm-file inputs/my_herm.h5 \
    --filter-file inputs/herald_filter.hdf5 \
    --save-path ./traces_out/ \
    --enable-LEE --lee-rate-hz 100 \
    --lee-energy-min-ev 1 --lee-energy-max-ev 100 \
    --lee-power-law-index 2 --lee-seed 42
```

Explicit CLI values override YAML values, including zero-valued rates and spectral indices. `--disable-LEE` overrides `enabled: true` in YAML, while `--enable-LEE` can enable a configured YAML section. Supplying `--lee-*` parameters alone does not enable injection. Without any LEE options or native YAML section, injection is disabled. Only `enabled: false` and a nondeterministic absent seed have defaults; all four rate/spectrum parameters are required when native injection is enabled. The LEE seed reproducibly controls LEE across traces and series, but does not seed or otherwise alter the existing noise generator.

If `--enable-LEE` is used without a native `lee:` section or native `--lee-*` parameters, the existing legacy PDF `salting:` workflow is selected instead and still requires `--processing-setup`. A present `lee:` section selects native mode, including when it is disabled, so it never falls back to legacy salting.

Native injection writes per-pulse truth to `<generated-save-path>/salting/lee_truth.hdf5`, with one row per pulse and channel. `global_event_idx` is the zero-based viewer event index; `series_num` and `event_in_series` identify the raw series and logical position within it. `onset_sample` is the sampled pulse onset **before** subtracting the template pretrigger for insertion. Energy, channel index, and the intended event-start epoch are included. A zero-pulse run still creates a valid empty truth file; disabled injection does not create this sidecar.

## Event Viewer (`scripts/event_viewer.ipynb`)

Interactive Jupyter notebook for inspecting generated traces alongside HeST truth data:

- **All-channels grid** — 24-panel trace overview with recoil info, interactive event slider
- **Single sensor with truth** — zoomed trace with truth hit table (channel, count, energy, timing) and arrival-time markers
- **Sensor heatmap** — pulse amplitude or truth energy on the physical HeRALD v1 hex-packed layout

Set `TRACE_PATH` and `HEST_H5_PATH` (optional, for truth overlay) in the config cell. Set `LEE_H5_PATH = None` to auto-discover `<TRACE_PATH>/salting/lee_truth.hdf5`, or provide an explicit native LEE truth path. `SHOW_LEE = True` adds dashed magenta LEE onset markers with pulse counts to all three waveform views. The selected-channel view also prints a table of onset sample, time in ms, and energy in eV. Markers use the exact stored onset, independently of waveform downsampling and **without** the HeRM trigger offset; `SHOW_LEE = False` suppresses only LEE overlays. The energy heat map remains HeRM/HeST truth only and does not include LEE energy.

The `LEE_H5_PATH`/`SHOW_LEE` settings correspond to `EventLoader(..., lee_h5=..., show_lee=...)`. A mismatched sampling rate, invalid schema, or inconsistent raw-series identity produces an error rather than showing incorrectly aligned markers.

## Project Structure

```
pytessim/
  pytessim/
    core/
      backgrounds/           # Background_Manager, Background_Factory, per-topology factories
      noise/Noise_Factory.py  # Noise generation from PSDs/CSDs
    utils/                    # Metadata generation, YAML config, time conversion
  scripts/
    file_generation.py        # Generic data generation (noise + LEE + backgrounds)
    hest_trace_generation.py  # HeST → TES traces pipeline
    create_filterFile.py      # Synthetic filter file for HeRALD
    event_viewer.ipynb        # Interactive trace + truth viewer
  examples/
    1ch_PPD/                  # 1-channel PPD example (ffgen.ipynb + configs)
    2ch_PPD/                  # 2-channel PPD example
    1ch_GaAs/                 # GaAs sandwich example
```
