import shutil
import tarfile
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from tqdm import tqdm
import matlab.engine
matlabeng = matlab.engine.start_matlab()   #启动matlab


def getIntDifMat(n: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """获取积分和求导矩阵

    Args:
        n (int): 矩阵阶数
        dt (float): 时间步长

    Returns:
        tuple[np.array, np.array]: 积分矩阵，求导矩阵
    """
    phi1 = np.concatenate([np.array([0, 0.5, 0]), np.zeros([n - 3, ])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([0, -0.5, 0])])
    Phi_dif = np.concatenate([np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
    Phi_int = np.linalg.inv(Phi_dif) * dt
    Phi_dif = Phi_dif / dt
    return Phi_int, Phi_dif


def getIntDifMat1(n: int, dt: float) -> tuple[np.array, np.array]:
    """获取积分和求导矩阵

    Args:
        n (int): 矩阵阶数
        dt (float): 时间步长

    Returns:
        tuple[np.array, np.array]: 积分矩阵，求导矩阵
    """
    Phi_int = np.triu(np.zeros((n, n)) + 1).transpose() - np.identity(n) / 2
    Phi_dif = np.linalg.inv(Phi_int) / dt
    Phi_int *= dt
    return Phi_int, Phi_dif


def getvel_dsp(acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """根据加速度时程获取速度和位移时程

    Args:
        acc (np.ndarray): 加速度时程矩阵，每一行表示一个时程
        dt (float): 时间步长

    Returns:
        tuple[np.ndarray, np.ndarray]: 速度时程矩阵，位移时程矩阵
    """
    n = acc.shape[-1]
    if n < 10000:
        Phi_int = np.triu(np.zeros((n, n)) + 1).transpose() - np.identity(n) / 2
        # Phi_dif = np.linalg.inv(Phi_int) / dt
        Phi_int *= dt

        vel = Phi_int.dot(acc.transpose()).transpose()
        if len(vel.shape) == 2:
            vel = vel - np.mean(vel, axis=1)[:, None]
        else:
            vel = vel - np.mean(vel)
        dsp = Phi_int.dot(vel.transpose()).transpose()
        if len(dsp.shape) == 2:
            dsp = dsp - np.mean(dsp, axis=1)[:, None]
        else:
            dsp = dsp - np.mean(dsp)
    else:
        vel = np.zeros_like(acc)
        vel[..., 0] = 0.5 * acc[..., 0] * dt
        for i in range(1, n):
            vel[..., i] = vel[..., i - 1] + 0.5 * (acc[..., i - 1] + acc[..., i]) * dt
        vel = vel - np.mean(vel)
        dsp = np.zeros_like(vel)
        dsp[..., 0] = 0.5 * vel[..., 0] * dt
        for i in range(1, n):
            dsp[..., i] = dsp[..., i - 1] + 0.5 * (vel[..., i - 1] + vel[..., i]) * dt
        dsp = dsp - np.mean(dsp)
    return vel, dsp


def baselineCorrection(acc: np.ndarray, dt: float, M: int) -> np.ndarray:
    """基线调整函数

    Args:
        acc (np.ndarray): 加速度时程
        dt (float): 时间步长
        M (int): 阶数，越高代表基线调整去除的高次项越多

    Returns:
        np.ndarray: 基线调整后的加速度时程
    """
    t = np.linspace(dt, dt * len(acc), len(acc))
    acc1 = acc
    vel, dsp = getvel_dsp(acc, dt)
    Gv = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gv[:, i] = t ** (M + 1 - i)
    polyv = np.dot(np.dot(np.linalg.inv(Gv.transpose().dot(Gv)), Gv.transpose()), vel)
    for i in range(M + 1):
        acc1 -= (M + 1 - i) * polyv[i] * t ** (M - i)
        
    acc_new = acc1
    vel1, dsp1 = getvel_dsp(acc1, dt)
    Gd = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gd[:, i] = t ** (M + 2 - i)
    polyd = np.dot(np.dot(np.linalg.inv(Gd.transpose().dot(Gd)), Gd.transpose()), dsp1)
    for i in range(M + 1):
        acc_new -= (M + 2 - i) * (M + 1 - i) * polyd[i] * t ** (M - i)
    return acc_new


def solve_sdof_eqwave_piecewise_exact(omg: np.ndarray, zeta: float, ag: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """求解单自由度响应

    Args:
        omg (np.ndarray): 单自由度圆频率数组
        zeta (float): 阻尼比
        ag (np.ndarray): 作用加速度时程
        dt (float): 时间步长

    Returns:
        tuple[np.ndarray, np.ndarray]: 位移响应时程，速度响应时程
    """
    omg_d = omg * np.sqrt(1.0 - zeta * zeta)
    m = len(omg)
    n = len(ag)
    u, v = np.zeros((m, n)), np.zeros((m, n))
    B1 = np.exp(-zeta * omg * dt) * np.cos(omg_d * dt)
    B2 = np.exp(-zeta * omg * dt) * np.sin(omg_d * dt)
    omg2 = 1.0 / omg / omg
    omg3 = 1.0 / omg / omg / omg
    for i in range(n - 1):
        alpha = (-ag[i + 1] + ag[i]) / dt
        A0 = -ag[i] * omg2 - 2.0 * zeta * alpha * omg3
        A1 = alpha * omg2
        A2 = u[:, i] - A0
        A3 = (v[:, i] + zeta * omg * A2 - A1) / omg_d
        u[:, i + 1] = A0 + A1 * dt + A2 * B1 + A3 * B2
        v[:, i + 1] = A1 + (omg_d * A3 - zeta * omg * A2) * B1 - (omg_d * A2 + zeta * omg * A3) * B2
    return u, v


def getResponseSpectrum(acc, dt, Period=np.logspace(-1.5, 0.5, 300), damp=0.05):
    sa = matlabeng.getResponseSpectrum(matlab.double(acc.tolist()), dt, matlab.double(Period.tolist()), damp)
    return np.array(sa).ravel()


def response_spectra_py(ag: np.ndarray, dt: float, T: np.ndarray=np.logspace(-1.5, 0.5, 300), zeta=0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        ag (np.ndarray): 加速度时程
        dt (float): 时间步长
        T (np.ndarray, optional): 反应谱周期数组. Defaults to np.logspace(-1.5, 0.5, 300).
        zeta (float, optional): 阻尼比. Defaults to 0.05.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 加速度谱，速度谱，位移谱
    """
    N = len(T)
    RSA = np.zeros(N)
    RSV = np.zeros(N)
    RSD = np.zeros(N)
    omg = 2.0 * np.pi / T
    u, v = solve_sdof_eqwave_piecewise_exact(omg, zeta, ag, dt)
    a = -2.0 * zeta * omg[:, None] * v - omg[:, None]  * omg[:, None]  * u
    RSA = np.max(np.abs(a), axis=1)
    RSV = np.max(np.abs(v), axis=1)
    RSD = np.max(np.abs(u), axis=1)
    return RSA, RSV, RSD


def getFourierSpectrum(acc: np.ndarray, dt: float, freq: np.ndarray=None, smooth: str=None, order: int=3, coef: int=None) -> tuple[np.ndarray, np.ndarray]:
    """求解傅里叶谱

    Args:
        acc (np.ndarray): 加速度时程
        dt (float): 时间步长
        freq (np.ndarray, optional): 傅里叶谱频率数组. Defaults to None.
        smooth (str, optional): 平滑方法. Defaults to None.
        order (int, optional): 's-g'平滑方法的平滑阶数. Defaults to 3.
        coef (int, optional): 平滑系数. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: 傅里叶谱频率数组，傅里叶谱值数组
    """
    num = len(acc)
    lent = dt * num
    newdt = 0.01
    newnum = np.floor(num * dt / newdt)
    acc = np.interp((np.arange(newnum) + 1) * newdt, (np.arange(num) + 1) * dt, acc)
    nfft = int(2 ** np.ceil(np.log2(newnum)))
    h = np.fft.fft(acc, n=nfft)
    f = np.fft.fftfreq(nfft, d=newdt)
    h = np.abs(h[1 : int(len(h) / 2)]) / lent
    f = f[1 : int(len(f) / 2)]
    if smooth == 'kohmachi':
        h = kohmachi(h, f, coef)
    if smooth == 'movemean':
        h = np.convolve(h, np.ones((coef,)) / coef, mode='same')
    if smooth == 's-g':
        h = signal.savgol_filter(h, coef, order, mode='nearest')
    if freq is not None:
        h = np.interp(freq, f, h)
        f = freq
    return f, h


def kohmachi(sign: np.ndarray, freq: np.ndarray, coef: int) -> np.ndarray:
    """信号频谱平滑函数

    Args:
        sign (np.ndarray): 信号频谱数组
        freq (np.ndarray): 信号频谱的频率数组
        coef (int): 平滑系数

    Returns:
        np.ndarray: 平滑后的频谱
    """
    f_shift = freq / (1 + 1e-4)
    f_shift = np.repeat(f_shift[:, None], len(f_shift), axis=1)
    z = f_shift / freq
    w = (np.sin(coef * np.log10(z)) / (coef * np.log10(z))) ** 4
    y = sign.dot(w) / np.sum(w, axis=0)
    return y


def EQprocess(acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GVDA和WLA数据库处理地震动的算法

    Args:
        acc (np.ndarray): 地震加速度时程数组
        dt (float): 时间步长

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 处理后的加速度、速度和位移时程
    """
    dt = float(dt)
    t = np.linspace(dt, dt * len(acc), len(acc))
    loc_uncer = 0.2

    # Remove mean
    acc = acc - np.mean(acc)

    # Copy acc
    acc_copy1 = np.copy(acc)
    acc_copy2 = np.copy(acc)

    # Remove linear trend
    coef1 = np.polyfit(t, acc_copy1, 1)
    acc_fit = coef1[0] * t + coef1[1]
    acc_copy1 = acc_copy1 - acc_fit

    # Acausal bandpass filter
    sos = signal.butter(4, [0.1, 20], 'bandpass', fs=int(1 / dt), output='sos')
    acc_filter = signal.sosfilt(sos, acc_copy1)

    # Find event onset
    loc, _ = matlabeng.PphasePicker(matlab.double(acc_filter.tolist()), dt, 'sm', nargout=2)

    # Initial baseline correction: remove pre-event mean
    acc_copy2 = acc_copy2 - np.mean(acc_copy2[:int((loc - loc_uncer) / dt)])
    # Integrate to velocity
    vel, _ = getvel_dsp(acc_copy2, dt)

    # Compute best fit trend in velocity
    vel_fit1_coef = np.polyfit(t, vel, 1)
    vel_fit2_coef = np.polyfit(t, vel, 2)
    vel_fit1 = vel_fit1_coef[0] * t + vel_fit1_coef[1]
    vel_fit2 = vel_fit2_coef[0] * t * t + vel_fit2_coef[1] * t + vel_fit2_coef[2]
    RMSD1 = np.sqrt(np.mean((vel_fit1 - vel) ** 2))
    RMSD2 = np.sqrt(np.mean((vel_fit2 - vel) ** 2))

    # Remove derivative of best fit trend from accelerationf
    if RMSD1 > RMSD2:
        acc_copy2 = acc_copy2 - (2 * vel_fit2_coef[0] * t + vel_fit2_coef[1])
    else:
        acc_copy2 = acc_copy2 - vel_fit1_coef[0]
        
    # Integrate acceleration to velocity
    vel, _ = getvel_dsp(acc_copy2, dt)

    # Quality check for velocity
    flc = 0.1
    fhc = np.min([40, 0.5 / dt - 5])
    win_len = np.max([loc - loc_uncer, 1 / flc])
    lead = np.abs(np.mean(vel[:int(win_len / dt)]))
    trail = np.abs(np.mean(vel[-int(win_len / dt):]))
    if lead > 0.01 or trail > 0.01:
        print('Quality check for velocity not pass!')

    # Tapering and padding
    N_begin = int((loc - loc_uncer) / dt / 2)
    N_end = int(3 / dt)
    taper_begin = 0.5 * (1 - np.cos(np.pi * np.linspace(0, N_begin - 1, N_begin) / N_begin))
    taper_end = 0.5 * (1 + np.cos(np.pi * np.linspace(0, N_end - 1, N_end) / N_end))
    acc_copy2[:N_begin] = acc_copy2[:N_begin] * taper_begin
    acc_copy2[-N_end:] = acc_copy2[-N_end:] * taper_end

    num_pad = int(6 / flc / dt)
    acc_copy2 = np.concatenate([np.zeros(int(num_pad / 2)), acc_copy2, np.zeros(int(num_pad / 2))])

    # Acausal bandpass filter acceleration
    sos = scipy.signal.butter(4, [flc, fhc], 'bandpass', fs=int(1 / dt), output='sos')
    acc_copy2 = scipy.signal.sosfilt(sos, acc_copy2)

    # Integrate acceleration to velocity and displacement
    vel, dsp = getvel_dsp(acc_copy2, dt)
    acc_copy2 = acc_copy2[int(num_pad / 2) + 2 : len(acc_copy2) - int(num_pad / 2) + 2]
    vel = vel[int(num_pad / 2) + 2 : len(vel) - int(num_pad / 2) + 2]
    dsp = dsp[int(num_pad / 2) + 2 : len(dsp) - int(num_pad / 2) + 2]

    # Quality check for final velocity and displacement
    win_len = np.max([loc - loc_uncer, 1 / flc])
    vel_lead = np.abs(np.mean(vel[:int(win_len / dt)]))
    vel_trail = np.abs(np.mean(vel[-int(win_len / dt):]))
    dsp_trail = np.abs(np.mean(dsp[-int(win_len / dt):]))
    if vel_lead > 0.01 or vel_trail > 0.01 or dsp_trail > 0.01:
        print('Quality check for velocity and displacement not pass!')

    return acc_copy2, vel, dsp


def getAccMsg(acc: np.ndarray, dt: float) -> list:
    """获取加速度时程的相关信息（PGA、PGV、PGD等）

    Args:
        acc (np.ndarray): 加速度时程
        dt (float): 时间步长

    Returns:
        list: 加速度时程的关键参数
    """
    accmsg = []
    acc2 = np.insert(acc, 0, [0])
    acc2 = acc2[0 : -1]
    accAvg = (acc + acc2) / 2
    vel = np.cumsum(accAvg) * dt
    velAvg = vel + (acc / 3 + acc2 / 6) * dt
    dsp = np.cumsum(velAvg) * dt
    accmsg.append(max(abs(acc)))
    accmsg.append(max(abs(vel)))
    accmsg.append(max(abs(dsp)))
    CAV = np.cumsum(abs(acc)) * dt
    accmsg.append(CAV[-1])
    Ia = np.cumsum(acc * acc) * dt * np.pi / (2 * 981)
    accmsg.append(Ia[-1])
    idx = np.where(Ia < 0.05 * Ia[-1])
    idx5 = idx[0][-1]
    idx = np.where(Ia > 0.75 * Ia[-1])
    idx75 = idx[0][0]
    idx = np.where(Ia > 0.95 * Ia[-1])
    idx95 = idx[0][0]
    accmsg.append(idx5 * dt)
    accmsg.append((idx75 - idx5) * dt)
    accmsg.append((idx95 - idx5) * dt)
    return accmsg