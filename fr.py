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
    """
    fr_amp_only = np.zeros((f.shape[0], xgrid.shape[0]), dtype='float32')  # Initialize amplitude array
    fr_phase_only = np.zeros((f.shape[0], xgrid.shape[0]), dtype=complex)  # Initialize phase array

    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)

        # Apply phase term using Euler's formula
        phase_term = np.exp(1j * theta_set[:, i][:, None])

        # Apply Gaussian kernel to amplitude
        gaussian_term_amp = np.exp(- dist ** 2 / sigma_phase ** 2)
        fr_amp_only += gaussian_term_amp * (r[:, i][:, None] / np.max(r, axis=1)[:, None])

        # Apply separate Gaussian kernel to phase
        gaussian_term_phase = np.exp(- dist ** 2 / sigma_phase ** 3)
        fr_phase_only += gaussian_term_phase * phase_term

    # Combine the amplitude and phase information into the complex fr
    fr = fr_amp_only * np.exp(1j * np.angle(fr_phase_only))

    m1 = np.zeros((f.shape[0], xgrid.shape[0]), dtype='float32')
    fr_ground = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32')
    m2 = np.ones((f.shape[0], xgrid.shape[0]), dtype='float32') - m1
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
