import numpy as np

from qetpy.utils import fft, ifft, fftfreq, rfftfreq
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from detprocess import FilterData
from numba import njit

class Noise_Factory(FilterData):
    """
    Class for interpolating user-defined PSDs and generating statistically representative 
    noise samples therefrom. Generation from PSDs is relatively simple; generation from
    CSDs is more complicated, utilizing cholesky decomposition to properly sample noise 
    shared between samples

    Paremeters
    ----------

    filter_file : string
        path to filter file storing the noise templates
    event_length_sec : float
        desired event length for generated data in seconds
    fs : float
        sampling frequency; must be shared by the user-defined
        PSD and the data to be generated
    verbose : bool
        Verbosity flag
    """

    def __init__(self, filter_file, event_length_sec, fs, verbose=True):

        self._verbose = verbose
        self._filter_data = None
        self._event_length_sec = event_length_sec
        self._fs = fs
        self._nsamples = int(self._fs * self._event_length_sec)

        self._csd_norm = self._fs * self._nsamples


        super().__init__(verbose=verbose)

        self.load_hdf5(filter_file, overwrite=False)
    
        self.noise_interp = {}
        self.L_matrices = {}
        self.channel_list = []

    def initialize_factory(self, channel):
        """
        Performs necessary pre-calculations to generating data, including interpolating the 
        PSD/CSD and calculating Cholesky decomposition for CSDs

        Parameters
        ----------
        channel : string
            string defining channel(s) in the filter file 
            (e.g. "channel0", "channel1", "channel0|channel1")

        Returns
        -------
        None

        """
        self.channel_list.append(channel)
        #interpolate and store CSD
        self._interp_csd(channel)

        #Pre-calculate L matrices
        self._precalc_L_matrices(channel)


    
    def _interp_csd(self, channel):

        """"
        Reads a CSD from the filterfile, interpolates it, and stores the 
        interpolated CSD under the channel name for later use.

        Parameters
        ----------
        channel : string
            string defining channel(s) in the filter file 
            (e.g. "channel0", "channel1", "channel0|channel1")

        Returns
        -------
        None        
        """

        csd_orig, freqs_orig = self.get_csd(channel)


        freqs_interp = fftfreq(self._nsamples, self._fs)
        zero_freqs = ( (0 < np.abs(freqs_interp)) & (np.abs(freqs_interp) < freqs_orig[1]) )  | ( (freqs_interp) > np.max(freqs_orig)) | ( (freqs_interp) < np.min(freqs_orig))

        if csd_orig.ndim == 3:

            csd_orig_real = np.real(csd_orig)
            csd_orig_imag = np.imag(csd_orig)
            csd_interp = np.zeros((2,2,self._nsamples), dtype = 'complex128')

            csd_interp[0,0] += np.where(zero_freqs, 0, interp1d(freqs_orig, csd_orig_real[0,0], bounds_error=False, fill_value = 0)(freqs_interp))
            csd_interp[1,1] += np.where(zero_freqs, 0, interp1d(freqs_orig, csd_orig_real[1,1], bounds_error=False, fill_value = 0)(freqs_interp))

            csd_interp[0,1] += np.where(zero_freqs, 0, interp1d(freqs_orig, csd_orig_real[0,1], bounds_error=False, fill_value = 0)(freqs_interp))
            csd_interp[0,1] += (0+1j)*np.where(zero_freqs, 0, interp1d(freqs_orig, csd_orig_imag[0,1], bounds_error=False, fill_value = 0)(freqs_interp))

            csd_interp[1,0] = np.conjugate(csd_interp[0,1])


        elif csd_orig.ndim == 1:

            csd_interp = np.where(zero_freqs, 0, interp1d(freqs_orig, csd_orig, bounds_error=False, fill_value = 0)(freqs_interp))

        self.noise_interp[channel] = (csd_interp, freqs_interp)
            

    def _precalc_L_matrices(self, channel):

        """"
        Reads a noise template; calculates and stores cholesky decomposition
        if a CSD. If a PSD, do nothing.

        Parameters
        ----------
        channel : string
            string defining channel(s) in the filter file 
            (e.g. "channel0", "channel1", "channel0|channel1")

        Returns
        -------
        None        

        """
        
        csd_interp, freqs_interp = self.noise_interp[channel]
        _, freqs_orig = self.get_csd(channel)
        
        if csd_interp.ndim == 1:

            self.L_matrices[channel] = None
            return 
        
        elif csd_interp.ndim == 3:

            Ls = np.full_like(csd_interp, (0+0j))

            #Skip any frequencies that won't play nice with a Cholesky decomposition (any with zero as an eigenvalue)
            zero_freqs = ( (0 < np.abs(freqs_interp)) & (np.abs(freqs_interp) < freqs_orig[1]) )  | ( (freqs_interp) > np.max(freqs_orig)) | ( (freqs_interp) < np.min(freqs_orig))

            Ls[:,:,~zero_freqs] = (np.linalg.cholesky((csd_interp.transpose(2,0,1))[~zero_freqs])).transpose(1,2,0)

            #Store
            self.L_matrices[channel] = Ls

    def sample_noise(self, channel):

        if self.L_matrices[channel] is not None:

            zs = self.gen_white_fourier_coeffs_2chan()

            fourier_amps = np.einsum('ijn,jn->in',self.L_matrices[channel], zs) * np.sqrt(self._csd_norm)

            return np.real(ifft(fourier_amps))
        else:
            
            zs = self.gen_white_fourier_coeffs()

            psd, _ = self.noise_interp[channel]
            fourier_amps =  zs * np.sqrt(psd * self._csd_norm)

            return np.real(ifft(fourier_amps))


    def gen_white_fourier_coeffs(self):
        """
        Generates fourier coefficients corresponding to white noise with a (two-sided)
        PSD equal to one at all non-zero frequencies. Phases are generated to ensure
        that the imaginary component is odd in frequency (i.e. to ensure that the 
        inverse fft will be real)
        """
        return _gen_white_fourier_coeffs(self._nsamples)
    
    def gen_white_fourier_coeffs_2chan(self):
        """
        Does the same, but for a pair of sensors with no correlated noise
        """

        return np.array([_gen_white_fourier_coeffs(self._nsamples),
                         _gen_white_fourier_coeffs(self._nsamples)])
    
def _gen_white_fourier_coeffs(n):

    half = (n - 1) // 2
    even = (n % 2 == 0)

    zs = (np.random.normal(size=half) + 1j * np.random.normal(size=half)) / np.sqrt(2)

    if even:
        zs_final = np.empty(n, dtype=np.complex128)
        zs_final[0] = 0.0 # Baseline offset
        zs_final[1:half+1] = zs
        zs_final[half+1] = np.random.normal()  # Nyquist frequency term
        zs_final[half+2:] = np.conjugate(zs[::-1])
    else:
        zs_final = np.empty(n, dtype=np.complex128)
        zs_final[0] = 0.0 # Baseline offset
        zs_final[1:half+1] = zs
        zs_final[half+1:] = np.conjugate(zs[::-1])

    return zs_final

        


    