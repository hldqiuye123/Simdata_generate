import numpy as np
import torch


def frequency_generator(f, nf, min_sep, dist_distribution):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)
        
def find_freq_idx(fr, batch):
    """
    Extract indices of the highest peaks from a frequency representation.
    """
    for n in range(batch):

        peaks, _ = scipy.signal.find_peaks(fr[n], height=0.1)

        # 获取峰值的值
        peak_values = fr[n][peaks]

        # 对峰值的值进行排序，获取排序后的索引（降序）
        sorted_indices = np.argsort(peak_values)[::-1]
        top_indices = peaks[sorted_indices[:min(10, len(peaks))]]
        top_values = peak_values[sorted_indices[:min(10, len(peaks))]]


    return top_indices,top_values

def rebuild_time_signal(args, fre_point, amp, phase, time_len, amplitude_threshold):
    # 创建频率数组，增加一个批处理维度
    frequencies = torch.linspace(0, 1, fre_point, dtype=torch.double, device=args.device).unsqueeze(0)

    # 创建时间点数组，增加一个批处理维度
    xgrid = torch.arange(0, time_len, dtype=torch.double, device=args.device).unsqueeze(0)

    # 应用幅度阈值
#     amp_mod = amp.clone()
#     amp_mod[amp_mod < amplitude_threshold] = 0

    # 生成正弦波和余弦波，使用广播
    sin_waves = amp.unsqueeze(2) * torch.sin(2 * np.pi * frequencies.unsqueeze(2) * xgrid + phase.unsqueeze(2))
    cos_waves = amp.unsqueeze(2) * torch.cos(2 * np.pi * frequencies.unsqueeze(2) * xgrid + phase.unsqueeze(2))

    # 累加复数信号并规范化
    s_complex_batch = torch.sum(sin_waves, dim=1) + 1j * torch.sum(cos_waves, dim=1)

    # 使用均方根进行归一化，类似于 gen_signal 的方法
    rms_values = torch.sqrt(torch.mean(torch.abs(s_complex_batch)**2+ 0.00001, dim=1, keepdim=True))  # 计算均方根
    s_complex_batch = (s_complex_batch+ 1e-7) / (rms_values + 0.00001)

    return s_complex_batch


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (15 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return 5*np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)



def gen_signal(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    theta_set = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        if n%50==0:
            print(n)

        frequency_generator(f[n], nfreq[n], d_sep, distance)
        if n == 0:
            nfreq[n] = 1
            f[n, 0] = 0
            r[n,0]=1
            theta_set[n,0] = 0
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            theta_set[n,i] =theta[n,i]
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / (np.sqrt(np.mean(np.power(s[n], 2)+ 0.000001))+1e-13)
    # f.sort(axis=1)
    # theta_set.sort(axis=1)
    f[f == float('inf')] = -10
    theta_set[theta_set == float('inf')] = 0
    # 创建一个复数数组
    s_complex = s[:, 0, :] + 1j * s[:, 1, :]
    s_complex = np.pad(s_complex, (0, 4096 - 64), 'constant')
    # 执行傅里叶变换
    s_Fourier = np.fft.fft(s_complex, axis=-1)
    # s_Fourier = np.fft.fftshift(s_Fourier)
    return s.astype('float32'), f.astype('float32'), nfreq,r,s_Fourier,theta_set.astype('float32')

def gen_signal_accuracy(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=1
        r[n,0]=1

        frequency_generator(f[n], nfreq[n], d_sep, distance)
        f[n,0]=0.200001
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r

def gen_signal_resolution(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False,dw=None):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=2
        r[n,0:2]=np.array([1,1])

        frequency_generator(f[n], nfreq[n], d_sep, distance)
        f[n,0:2]=np.array([0,dw/64])
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r

def gen_signal_weak(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    for n in range(num_samples):
        nfreq[n]=2

        r[n, 1] = np.abs(np.random.randn()) + 0.1
        r[n, 0] = 10 ** (20/ 20) * r[n, 1]



        frequency_generator(f[n], nfreq[n], d_sep, distance)

        #signal for testing the matched filter module
        if n==0:
            nfreq[0]=1
            r[0,0]=1
            f[0,0:10]='inf'
            f[0,0]=0.0

        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j*theta[n,i]+ 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    # f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq,r


def gen_signal_weak2(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
                    floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq

    amp_diff=np.array(range(8,35,2))
    for n in range(num_samples):
        nfreq[n] = 2

        r[n, 1] = np.abs(np.random.randn()) + 0.1
        r[n, 0] = 10 ** (amp_diff[n//1000] / 20) * r[n, 1]

        frequency_generator(f[n], nfreq[n], d_sep, distance)

        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float('inf')] = -10
    return s.astype('float32'), f.astype('float32'), nfreq, r

def compute_snr(clean, noisy):
    return np.linalg.norm(clean, axis=1) ** 2 / np.linalg.norm(clean - noisy, axis=1) ** 2


def compute_snr_torch(clean, noisy):
    return (torch.sum(clean.view(clean.size(0), -1) ** 2, dim=1) / torch.sum(
        ((clean - noisy).view(clean.size(0), -1)) ** 2, dim=1)).mean()
