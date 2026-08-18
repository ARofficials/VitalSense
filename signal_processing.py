# signal_processing.py
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch, butter, filtfilt

def apply_bandpass(signal_data, fs):
    low = 0.7; high = 3.0; nyq = 0.5 * fs 
    if high >= nyq: high = nyq - 0.1
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal_data)

def clean_signal(wave, fs):
    wave = signal.detrend(wave)
    wave = apply_bandpass(wave, fs)
    win = int(5.0 * fs)
    if win < 1: win = 1
    pad = np.pad(wave, (win//2, win//2), mode='edge')
    rms = np.array([np.sqrt(np.mean(pad[i:i+win]**2)) for i in range(len(wave))])
    wave = wave / (rms + 1e-6)
    return (wave - np.mean(wave)) / (np.std(wave) + 1e-6)

def run_pos_algorithm(r, g, b, fs):
    win = int(1.6 * fs)
    l = len(r)
    h = np.zeros(l)
    for i in range(0, l - win, 1):
        seg_r = r[i:i+win]; seg_g = g[i:i+win]; seg_b = b[i:i+win]
        cn_r = seg_r / (np.mean(seg_r) + 1e-6)
        cn_g = seg_g / (np.mean(seg_g) + 1e-6)
        cn_b = seg_b / (np.mean(seg_b) + 1e-6)
        s1 = cn_g - cn_b
        s2 = cn_g + cn_b - 2 * cn_r
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        h_seg = s1 + alpha * s2
        h[i + win//2] = h_seg[win//2]
    return clean_signal(h, fs)

def get_dominant_freq(sig, fs):
    f, p = welch(sig, fs=fs, nfft=2048)
    mask = (f >= 0.7) & (f <= 2.5) 
    if np.any(mask):
        return f[mask][np.argmax(p[mask])] * 60.0
    return 0.0

def compute_hrv_features(ppg, fs):
    peaks, _ = find_peaks(ppg, distance=fs*0.5)
    if len(peaks) < 2: return 0.0, 0.0, 0.0, 0.0
    rr = np.diff(peaks) / fs * 1000 
    sdnn, rmssd = np.std(rr), np.sqrt(np.mean(np.diff(rr)**2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100
    n_seg = min(len(rr), 256)
    if n_seg < 2: return sdnn, rmssd, pnn50, 0.0
    f, p = welch(rr/1000, fs=4.0, nperseg=n_seg)
    lf = np.trapz(p[(f>=0.04)&(f<0.15)], f[(f>=0.04)&(f<0.15)])
    hf = np.trapz(p[(f>=0.15)&(f<0.4)], f[(f>=0.15)&(f<0.4)])
    return sdnn, rmssd, pnn50, lf/(hf+1e-6)

def predict_breathing_heuristic(ppg, fs):
    # Fallback if XGBoost model fails or isn't compatible
    if len(ppg) < 150: return 15.0
    sos = signal.butter(2, [0.1/(fs/2), 0.5/(fs/2)], btype='band', output='sos')
    resp_sig = signal.sosfiltfilt(sos, ppg)
    pks, _ = find_peaks(resp_sig, distance=fs*2.0)
    if len(pks) < 2: return 14.0
    br = (len(pks) / (len(ppg)/fs)) * 60.0
    return float(round(br, 1))