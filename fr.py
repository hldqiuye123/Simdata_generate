import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def freq2fr(f, xgrid, kernel_type='gaussian', param=None, r=None,nfreq=None,theta_set=None):
    """
    Convert an array of frequencies to a frequency representation discretized on xgrid.
    """
    if kernel_type == 'gaussian':
        return gaussian_kernel(f, xgrid, param, r,nfreq,theta_set)
    elif kernel_type == 'triangle':
        return triangle(f, xgrid, param)

# def gaussian_kernel(f, xgrid, sigma, r,nfreq):
#     """
#     Create a frequency representation with a Gaussian kernel.
#     """
#     for i in range(f.shape[0]):
#         r[i,nfreq[i]:]=np.min(r[i,0:nfreq[i]])
#
#     fr = np.zeros((f.shape[0], xgrid.shape[0]))
#     # for i in range(f.shape[1]):
#     #     dist = np.abs(xgrid[None, :] - f[:, i][:, None])
#     #     rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
#     #     ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
#     #     dist = np.minimum(dist, rdist, ldist)
#     #     # fr += np.exp(- dist ** 2 / sigma ** 2)
#     #     fr += np.exp(- dist ** 2 / sigma ** 2)
#     # fr=20*np.log10(fr+1e-2)+40*(r[:,i][:,None])
#     # m1 = np.zeros((f.shape[0], xgrid.shape[0]), dtype='float32')
#
#     fr_ground = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32')
#     for ii in range(fr.shape[0]):
#
#         for i in range(f.shape[1]):
#             if f[ii, i] == -10:
#                 continue
#             idx0 = (f[ii, i] + 0.5) / (1 / (np.shape(xgrid)[0]))
#
#             ctr0 = int(np.round(idx0))
#             if ctr0 == (np.shape(xgrid)[0]):
#                 ctr0 = (np.shape(xgrid)[0]) - 1
#
#             # if ctr0 == 0:
#             #     ctr0_up = 1
#             # else:
#             #     ctr0_up = ctr0 - 1
#             # if ctr0 == np.shape(xgrid)[0] - 1:
#             #     ctr0_down = np.shape(xgrid)[0] - 2
#             # else:
#             #     ctr0_down = ctr0 + 1
#
#             FX=xgrid[ctr0]
#             dist = np.abs(xgrid - FX)
#             rdist = np.abs(xgrid - (FX + 1))
#             ldist = np.abs(xgrid - (FX - 1))
#             dist = np.minimum(dist, rdist, ldist)
#             fr[ii,:] += np.exp(- dist ** 2 / sigma ** 2)*20*np.log10(10*r[ii,i]+1)
#             cost = (np.power(20*np.log10(10*np.max(r[ii,:])+1),2)/ np.power(fr[ii,ctr0],2))
#             fr_ground[ii, ctr0] = cost
#
#
#
#     m2 = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32')
#     m1=m2
#     return fr, fr_ground.astype('float32'), m1, m2
import copy


def gaussian_kernel(f, xgrid,  sigma_phase, r, nfreq, theta_set):
    """
    Create a frequency representation with separate Gaussian kernels for amplitude and phase.

    Parameters:
    f : ndarray of shape (B, P)
        Instantaneous frequency positions for each signal component across batches.
    xgrid : ndarray of shape (N,)
        Discretized frequency grid for mapping the frequency response.
    sigma_phase : float
        Standard deviation controlling the spread of the Gaussian kernel.
    r : ndarray of shape (B, P)
        Amplitude values associated with each frequency component.
    nfreq : int
        Total number of frequency bins (unused in body but may be used externally for consistency).
    theta_set : ndarray of shape (B, P)
        Phase values (in radians) associated with each frequency component.

    Returns:
    fr_amp : ndarray of shape (B, N)
        Amplitude spectrogram constructed via Gaussian smoothing.
    fr_phase : ndarray of shape (B, N)
        Phase spectrogram extracted from the complex frequency representation.
    fr_ground : ndarray of shape (B, N)
        Ground-truth amplitude map placeholder (set to ones, modifiable externally).
    m1 : ndarray of shape (B, N)
        Mask matrix initialized to zeros (may be used for supervision or evaluation).
    m2 : ndarray of shape (B, N)
        Complementary mask to m1, initialized to ones.
    """
    # Initialize the amplitude response matrix (real-valued)
    fr_amp_only = np.zeros((f.shape[0], xgrid.shape[0]), dtype='float32')

    # Initialize the phase response matrix (complex-valued)
    fr_phase_only = np.zeros((f.shape[0], xgrid.shape[0]), dtype=complex)

    # Iterate over each frequency component
    for i in range(f.shape[1]):
        # Compute absolute distances between frequency components and grid points
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])

        # Handle frequency periodicity by considering wrap-around effects
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))  # right neighbor
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))  # left neighbor
        dist = np.minimum(dist, np.minimum(rdist, ldist))  # element-wise minimum

        # Compute phase term using complex exponential (Euler's identity)
        phase_term = np.exp(1j * theta_set[:, i][:, None])

        # Apply Gaussian kernel to amplitude channel
        gaussian_term_amp = np.exp(- dist ** 2 / sigma_phase ** 2)
        # Normalize amplitude using per-sample maximum, and accumulate
        fr_amp_only += gaussian_term_amp * (r[:, i][:, None] / np.max(r, axis=1)[:, None])

        # Apply a sharper Gaussian to the phase term for selective smoothing
        gaussian_term_phase = np.exp(- dist ** 2 / sigma_phase ** 3)
        # Accumulate the complex-valued phase contributions
        fr_phase_only += gaussian_term_phase * phase_term

    # Combine amplitude and phase to form final complex frequency representation
    # Phase is extracted via angle of the accumulated complex phase signal
    fr = fr_amp_only * np.exp(1j * np.angle(fr_phase_only))

    # Generate auxiliary outputs for further processing or loss computation
    m1 = np.zeros((f.shape[0], xgrid.shape[0]), dtype='float32')  # Zero mask
    fr_ground = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32')  # Ground truth placeholder
    m2 = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32') - m1  # Complementary mask

    # Extract amplitude and phase from final complex response
    fr_amp = abs(fr)
    fr_phase = np.angle(fr)

    return fr_amp, fr_phase, fr_ground, m1, m2



def triangle(f, xgrid, slope):
    """
    Create a frequency representation with a triangle kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.clip(1 - slope * dist, 0, 1)
    return fr


def find_freq_m(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((1, max_freq))
    for n in range(1):
        find_peaks_out = scipy.signal.find_peaks(fr, height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), nfreq)
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        ff[n, :num_spikes] = np.sort(xgrid[find_peaks_out[0][idx]])
    return ff

def find_freq_idx(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((1, max_freq))
    for n in range(1):
        find_peaks_out = scipy.signal.find_peaks(fr, height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), nfreq)
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        freq_idx=find_peaks_out[0][idx]
    return np.sort(freq_idx)


def find_freq(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((nfreq.shape[0], max_freq))
    for n in range(len(nfreq)):

        if nfreq[n] < 1:  # at least one frequency
            nf = 1
        else:
            nf = nfreq[n]

        find_peaks_out = scipy.signal.find_peaks(fr[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(nf))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        ff[n, :num_spikes] = np.sort(xgrid[find_peaks_out[0][idx]])
    return ff


def periodogram(signal, xgrid):
    """
    Compute periodogram.
    """
    js = np.arange(signal.shape[1])
    return (np.abs(np.exp(-2.j * np.pi * xgrid[:, None] * js).dot(signal.T) / signal.shape[1]) ** 2).T


def make_hankel(signal, m):
    """
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype='complex128')
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(-2.0j * np.pi * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1)))
        u = V[nfreq[n]:]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr
