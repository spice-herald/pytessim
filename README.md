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

Key options: `--no-noise` for signal-only traces, `--max-events N` to limit events, `--uniform-channel channel_00` to apply one channel's template/PSD to all 24 sensors, `--enable-LEE` / `--enable-background` for post-processing injection passes.

## Event Viewer (`scripts/event_viewer.ipynb`)

Interactive Jupyter notebook for inspecting generated traces alongside HeST truth data:

- **All-channels grid** — 24-panel trace overview with recoil info, interactive event slider
- **Single sensor with truth** — zoomed trace with truth hit table (channel, count, energy, timing) and arrival-time markers
- **Sensor heatmap** — pulse amplitude or truth energy on the physical HeRALD v1 hex-packed layout

Set `TRACE_PATH`, `HEST_H5_PATH` (optional, for truth overlay), and `FILTER_PATH` in the config cell.

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
