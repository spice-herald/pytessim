import numpy as np
from detprocess import FilterData

# Sampling and trace parameters (matching ffgen.ipynb)
fs = 1.25e6
trace_length_sec = 0.020       # 20 ms
pretrigger_length_sec = 0.005  # 5 ms
nsamp = int(fs * trace_length_sec)  # 25000 samples

tdata = np.arange(nsamp) / fs
fdata = np.fft.fftfreq(nsamp, 1 / fs)

# Template — two-pole pulse, onset at t0 = pretrigger
def twopole(t, trise, tfall, A, t0):
    pulse = A * (np.exp(-(t - t0) / tfall) - np.exp(-(t - t0) / trise))
    return np.where(pulse > 0, pulse, 0)

template = twopole(tdata, 10e-6, 100e-6, 1, pretrigger_length_sec)
template /= np.max(template)

# PSD — band-limited noise
def power(f, f0, n, A0):
    return A0 * A0 / (1 + (f / f0) ** n) ** 2

psd = power(fdata, 1e4, 2, 1e-10)
psd[0] = 0

# dP/dI — real-valued, rising with frequency
def dpdi_func(f, f0, n, A0):
    return A0 * (1 + (np.abs(f) / f0) ** n)

dpdi = dpdi_func(fdata, 5e4, 2.5, 1e-7).astype(np.complex128)
dpdi_err = 0.1 * dpdi

# Build the filter file
fd = FilterData()
channel = 'channel_00'

fd.set_template(channel, template=template, tag=channel,
                sample_rate=fs,
                pretrigger_length_msec=pretrigger_length_sec * 1000)
fd.set_psd(channel, psd, fdata, tag=channel)
fd.set_dpdi(channel, dpdi, dpdi_err, fdata, poles=3, tag=channel)

fd.save_hdf5('../inputs/herald_filter.hdf5')