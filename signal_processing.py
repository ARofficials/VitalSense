import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch, butter, filtfilt, savgol_filter

def bandpass_filter(signal_data, fs, low=0.7, high=3.0, order=3):
    nyq = 0.5 * fs 
    if high >= nyq: high = nyq - 0.1
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal_data)

def clean_signal(wave, fs):
    wave = signal.detrend(wave)
    # Use wider band for general cleaning (up to 210 BPM)
    wave = bandpass_filter(wave, fs, low=0.7, high=3.5, order=4)
    
    # Amplitude normalization (Sliding RMS)
    win = int(5.0 * fs)
    if win < 1: win = 1
    pad = np.pad(wave, (win//2, win//2), mode='edge')
    rms = np.array([np.sqrt(np.mean(pad[i:i+win]**2)) for i in range(len(wave))])
    wave = wave / (rms + 1e-6)
    
    # Z-score normalization
    return (wave - np.mean(wave)) / (np.std(wave) + 1e-6)

def refine_peaks_interp(sig, peaks):
    refined_peaks = []
    for p in peaks:
        if p < 1 or p >= len(sig) - 1:
            refined_peaks.append(float(p))
            continue
            
        alpha = sig[p-1]
        beta = sig[p]
        gamma = sig[p+1]
        
        denom = (alpha - 2*beta + gamma)
        if denom == 0: 
            refined_peaks.append(float(p))
            continue
            
        delta = 0.5 * (alpha - gamma) / denom
        refined_peaks.append(p + delta)
        
    return np.array(refined_peaks)

def calculate_baevsky_stress(rr_ms):
    if len(rr_ms) < 10: return 0.0
    
    bin_size = 50 
    bins = np.arange(min(rr_ms), max(rr_ms) + bin_size, bin_size)
    hist, edges = np.histogram(rr_ms, bins=bins)
    
    if len(hist) == 0: return 0.0
    
    max_bin_idx = np.argmax(hist)
    Mo = (edges[max_bin_idx] + edges[max_bin_idx+1]) / 2 / 1000.0 
    
    AMo = (hist[max_bin_idx] / len(rr_ms)) * 100.0
    MxDM = (np.max(rr_ms) - np.min(rr_ms)) / 1000.0 
    
    if MxDM == 0 or Mo == 0: return 0.0
    
    SI = AMo / (2 * Mo * MxDM)
    return SI

def compute_hrv_features(ppg, fs):
    peaks, _ = find_peaks(ppg, distance=fs*0.5)
    
    if len(peaks) < 2: 
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    true_peaks = refine_peaks_interp(ppg, peaks)
    rr = np.diff(true_peaks) / fs * 1000 
    
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100
    
    n_seg = min(len(rr), 256)
    if n_seg < 2: return sdnn, rmssd, pnn50, 0.0, 0.0
    
    f, p = welch(rr/1000, fs=4.0, nperseg=n_seg)
    lf = np.trapz(p[(f>=0.04)&(f<0.15)], f[(f>=0.04)&(f<0.15)])
    hf = np.trapz(p[(f>=0.15)&(f<0.4)], f[(f>=0.15)&(f<0.4)])
    ratio = lf/(hf+1e-6)
    
    si = calculate_baevsky_stress(rr)
    
    return sdnn, rmssd, pnn50, ratio, si

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
    mask = (f >= 0.7) & (f <= 3.0) 
    if np.any(mask):
        return f[mask][np.argmax(p[mask])] * 60.0
    return 0.0

# --- NEW BREATHING LOGIC (AI Output Processing) ---
def extract_breathing_rate_from_envelope(envelope, fs):
    """
    Computes BR from the respiration envelope (AI Output) using Welch PSD.
    Band: 0.08 Hz (4.8 BPM) - 0.45 Hz (27 BPM)
    """
    if len(envelope) < fs * 2: return 15.0 # Not enough data
    
    # 1. Smooth the envelope
    try:
        win_len = 31
        if len(envelope) < win_len: win_len = len(envelope) // 2 * 2 + 1
        if win_len < 3: win_len = 3
        smooth_env = savgol_filter(envelope, window_length=win_len, polyorder=3)
    except:
        smooth_env = envelope

    # 2. Welch PSD
    f, pxx = welch(smooth_env, fs=fs, nperseg=len(smooth_env), nfft=4096)
    
    # 3. Mask for valid breathing range
    RESP_LOW = 0.08
    RESP_HIGH = 0.45
    mask = (f >= RESP_LOW) & (f <= RESP_HIGH)
    
    if not np.any(mask) or np.sum(pxx[mask]) == 0:
        return 14.5 # Default fallback
        
    # 4. Peak Finding
    best_freq = f[mask][np.argmax(pxx[mask])]
    br_bpm = best_freq * 60.0
    
    return float(round(br_bpm, 1))

# --- FALLBACK LOGIC (Missing Function Restored) ---
def predict_breathing_heuristic(ppg, fs):
    """
    Pure signal processing fallback if AI fails.
    Extracts breathing directly from the PPG baseline wander.
    """
    if len(ppg) < fs * 5: return 14.0 
    
    # 1. Filter to Respiratory Band (0.1 - 0.5 Hz)
    try:
        resp_sig = bandpass_filter(ppg, fs, low=0.1, high=0.5, order=2)
    except:
        return 15.0

    # 2. Welch PSD
    n_seg = min(len(resp_sig), 1024)
    f, p = welch(resp_sig, fs=fs, nperseg=n_seg, nfft=4096)
    
    # 3. Peak Finding
    mask = (f >= 0.1) & (f <= 0.5)
    
    if not np.any(mask) or np.sum(p[mask]) == 0:
        return 14.5 
    
    best_freq = f[mask][np.argmax(p[mask])]
    return float(round(best_freq * 60.0, 1))