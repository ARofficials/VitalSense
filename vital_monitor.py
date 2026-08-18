import sys
import os
import json
import time
import datetime
import cv2
import numpy as np
import mediapipe as mp
import torch
import xgboost as xgb
from scipy.stats import skew, kurtosis 

# --- SAFE PLOTTING IMPORTS ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt 

import config
import models_loading
import signal_processing as dsp

class ProductionVitalMonitor:
    def __init__(self):
        self.loader = models_loading.ModelLoader()
        
        # 1. Load AI Models
        self.tscan = self.loader.load_tscan()
        self.resnet = self.loader.load_resnet()
        self.resp_model = self.loader.load_breathing_model() 
        self.stress_model = self.loader.load_stress_model() 
        
        # 2. Face Detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.roi_state = None 
        self.output_dir = None
        
        # Smoothers
        self.br_history = []

    def create_output_directory(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        self.output_dir = os.path.join("outputs", date_str, time_str)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_report(self, data_dict, report_text):
        if not self.output_dir: return
        json_path = os.path.join(self.output_dir, "vital_data.json")
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
        txt_path = os.path.join(self.output_dir, "report.txt")
        with open(txt_path, 'w') as f:
            f.write(report_text)

    def extract_forehead_roi(self, frame, landmarks):
        h_img, w_img, _ = frame.shape
        fh_indices = [109, 10, 338, 299, 296, 336, 9]
        pts = np.array([[int(landmarks.landmark[p].x * w_img), int(landmarks.landmark[p].y * h_img)] for p in fh_indices])
        x, y, w, h = cv2.boundingRect(pts)
        
        if self.roi_state is None:
            self.roi_state = [float(x), float(y), float(w), float(h)]
        else:
            alpha = 0.9  
            self.roi_state[0] = alpha * self.roi_state[0] + (1-alpha) * x
            self.roi_state[1] = alpha * self.roi_state[1] + (1-alpha) * y
            self.roi_state[2] = alpha * self.roi_state[2] + (1-alpha) * w
            self.roi_state[3] = alpha * self.roi_state[3] + (1-alpha) * h
            
        return self._crop_from_state(frame)

    def get_last_roi(self, frame):
        if self.roi_state is None: return None
        return self._crop_from_state(frame)

    def _crop_from_state(self, frame):
        sx, sy, sw, sh = [int(v) for v in self.roi_state]
        sx = max(0, sx); sy = max(0, sy)
        if sw < 5 or sh < 5: return None
        crop = frame[sy:sy+sh, sx:sx+sw]
        if crop.size == 0: return None
        return cv2.resize(crop, (72, 72))

    def run_tscan_inference(self, buffer_slice):
        r_mean = np.mean(buffer_slice['fh'][:, :, :, 2], axis=(1, 2))
        g_mean = np.mean(buffer_slice['fh'][:, :, :, 1], axis=(1, 2))
        b_mean = np.mean(buffer_slice['fh'][:, :, :, 0], axis=(1, 2))
        
        # Crash Protection
        if self.tscan is None:
             return (g_mean - np.mean(g_mean)) * -1.0, r_mean, g_mean, b_mean

        raw = buffer_slice['fh'].astype(np.float32) / 255.0
        raw_padded = np.concatenate([raw[0:1], raw], axis=0)
        diff = (raw_padded[1:] - raw_padded[:-1]) / (raw_padded[1:] + raw_padded[:-1] + 1e-6)
        
        raw_t = torch.from_numpy(raw).float().unsqueeze(0).permute(0, 4, 1, 2, 3).to(self.loader.device)
        diff_t = torch.from_numpy(diff).float().unsqueeze(0).permute(0, 4, 1, 2, 3).to(self.loader.device)
        
        with torch.no_grad():
            ppg_out = self.tscan(diff_t, raw_t)
            
        return ppg_out.cpu().numpy().flatten(), r_mean, g_mean, b_mean

    def calculate_physio_score(self, segment_ppg, segment_red, segment_green, fs):
        n_seg = min(len(segment_ppg), 256)
        if n_seg < 10: return 0.0
        f, p = dsp.welch(segment_ppg, fs=fs, nperseg=n_seg, nfft=1024)
        
        mask_pulse = (f >= 0.7) & (f <= 2.0)
        mask_noise = (f < 0.7) | (f > 2.0)
        snr = np.sum(p[mask_pulse]) / (np.sum(p[mask_noise]) + 1e-6)
        base_score = snr 
        bpm_ppg = dsp.get_dominant_freq(segment_ppg, fs)
        bpm_red = dsp.get_dominant_freq(segment_red, fs)
        bpm_green = dsp.get_dominant_freq(segment_green, fs)
        multiplier = 1.0
        if bpm_ppg < 45 or bpm_ppg > 120: multiplier *= 0.1 
        if abs(bpm_red - bpm_green) > 12.0: multiplier *= 0.7 
        else: multiplier *= 1.2 
        return base_score * multiplier

    def generate_extended_features(self, ppg_signal, fs, hr, br, sdnn, rmssd, pnn50, lf_hf_ratio):
        peaks, _ = dsp.find_peaks(ppg_signal, distance=fs*0.5)
        true_peaks = dsp.refine_peaks_interp(ppg_signal, peaks)
        if len(true_peaks) < 2: return np.zeros((1, 18))
        rr_ms = np.diff(true_peaks) / fs * 1000.0 
        mean_rr = np.mean(rr_ms); median_rr = np.median(rr_ms)
        min_rr = np.min(rr_ms); max_rr = np.max(rr_ms); std_rr = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        if len(diff_rr) > 1:
            sd1 = np.sqrt(np.std(diff_rr, ddof=1)**2 * 0.5)
            sd2 = np.sqrt(2 * std_rr**2 - 0.5 * np.std(diff_rr, ddof=1)**2)
        else:
            sd1, sd2 = 0.0, 0.0

        skew_rr = skew(rr_ms) if len(rr_ms) > 2 else 0
        kurt_rr = kurtosis(rr_ms) if len(rr_ms) > 2 else 0
        
        n_seg = min(len(rr_ms), 256)
        if n_seg > 1:
            f, p = dsp.welch(rr_ms/1000.0, fs=4.0, nperseg=n_seg, nfft=256)
            vlf_pow = np.trapz(p[(f < 0.04)], f[(f < 0.04)])
            lf_pow = np.trapz(p[(f>=0.04)&(f<0.15)], f[(f>=0.04)&(f<0.15)])
            hf_pow = np.trapz(p[(f>=0.15)&(f<0.4)], f[(f>=0.15)&(f<0.4)])
            total_pow = vlf_pow + lf_pow + hf_pow
        else:
            lf_pow, hf_pow, vlf_pow, total_pow = 0,0,0,0

        features = np.array([
            hr, sdnn, rmssd, pnn50, lf_hf_ratio, br, 
            mean_rr, median_rr, min_rr, max_rr,
            lf_pow, hf_pow, vlf_pow, total_pow,
            sd1, sd2, skew_rr, kurt_rr
        ])
        return np.nan_to_num(features).reshape(1, -1)

    def process_full_recording(self, buffer, valid_frames):
        if valid_frames < 30:
            raise ValueError("Not enough data frames captured.")

        self.create_output_directory()
        times = buffer['time'][:valid_frames]
        duration = times[-1] - times[0]
        if duration <= 0: duration = 1.0
        real_fps = valid_frames / duration

        # Extraction
        full_ppg, full_r, full_g, full_b = [], [], [], []
        num_batches = valid_frames // config.BATCH_SIZE
        if num_batches == 0 and valid_frames > 0: pass 

        for i in range(num_batches):
            chunk = buffer[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
            ppg, r, g, b = self.run_tscan_inference(chunk)
            full_ppg.extend(ppg); full_r.extend(r); full_g.extend(g); full_b.extend(b)

        clean_ppg = dsp.clean_signal(np.array(full_ppg), fs=real_fps)
        clean_pos = dsp.run_pos_algorithm(np.array(full_r), np.array(full_g), np.array(full_b), fs=real_fps)
        clean_g = dsp.clean_signal(np.array(full_g), fs=real_fps)
        clean_r = dsp.clean_signal(np.array(full_r), fs=real_fps)

        # 2. Segment Selection (Physics Gated)
        win_frames = int(12.0 * real_fps)
        checkpoints = np.linspace(0, len(clean_ppg) - win_frames, 5, dtype=int)
        candidates = []
        for i in checkpoints:
            if i + win_frames >= len(clean_ppg): continue
            seg_ppg = clean_ppg[i : i+win_frames]
            seg_pos = clean_pos[i : i+win_frames]
            # Use Green/Red for verification (Physics Check)
            score = self.calculate_physio_score(seg_ppg, clean_r[i:i+win_frames], clean_g[i:i+win_frames], fs=real_fps)
            candidates.append({'segment': seg_ppg, 'segment_pos': seg_pos, 'score': score})
            
        # --- FIX: Ensure 'segment_pos' exists in fallback ---
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
        else:
            best = {
                'segment': clean_ppg[-win_frames:], 
                'segment_pos': clean_pos[-win_frames:], # <--- ADDED THIS LINE
                'score': 0
            }
            
        final_seg = best['segment']
        final_pos = best['segment_pos'] # Now guaranteed to exist

        # 3. Lock Window (256)
        TARGET_LEN = 256
        if len(final_seg) > TARGET_LEN:
            seg_input = final_seg[-TARGET_LEN:]
        else:
            seg_input = np.pad(final_seg, (TARGET_LEN - len(final_seg), 0), mode='edge')

        # --- HR & SNR PIPELINE ---
        ai_resnet = 70.0
        snr = 0.0
        seg_hr = None
        
        try:
            # 1. Bandpass Filter
            seg_hr = dsp.bandpass_filter(seg_input, fs=real_fps, low=0.7, high=3.0, order=3)
            
            # 2. SNR Check
            residual = seg_input - seg_hr
            residual_filt = dsp.bandpass_filter(residual, fs=real_fps, low=0.1, high=0.6, order=2)
            snr = np.var(seg_hr) / (np.var(residual_filt) + 1e-6)
            
            if np.std(seg_hr) < 1e-4: raise ValueError("Flatline")
            
            # 3. Inference
            if self.resnet is not None:
                seg_norm = (seg_hr - np.mean(seg_hr)) / (np.std(seg_hr) + 1e-6)
                t_seg = torch.tensor(seg_norm, dtype=torch.float32).view(1, 1, -1).to(self.loader.device)
                with torch.no_grad():
                    ai_resnet = self.resnet(t_seg).item()
            else:
                ai_resnet = dsp.get_dominant_freq(final_seg, real_fps)
                
            # 4. Gates
            hr_fft = dsp.get_dominant_freq(seg_hr, real_fps)
            
            if abs(ai_resnet - hr_fft) > 12.0:
                ai_resnet = hr_fft

            hr_tscan_base = dsp.get_dominant_freq(final_seg, real_fps)
            hr_pos_base = dsp.get_dominant_freq(final_pos, real_fps)
            
            if abs(ai_resnet - hr_tscan_base) > 20.0 and abs(ai_resnet - hr_pos_base) > 20.0:
                ai_resnet = np.mean([hr_tscan_base, hr_pos_base])
                
        except Exception:
            ai_resnet = dsp.get_dominant_freq(final_seg, real_fps)

        hr_tscan = dsp.get_dominant_freq(final_seg, real_fps)
        hr_pos = dsp.get_dominant_freq(final_pos, real_fps) 
        
        # Consensus
        diffs = [abs(hr_tscan - hr_pos), abs(hr_tscan - ai_resnet), abs(hr_pos - ai_resnet)]
        best_pair_idx = np.argmin(diffs)
        
        if diffs[best_pair_idx] > 15.0: 
            consensus_hr = np.mean([hr_tscan, hr_pos, ai_resnet])
        elif best_pair_idx == 0: 
            consensus_hr = (hr_tscan + hr_pos) / 2.0
        elif best_pair_idx == 1: 
            consensus_hr = (hr_tscan + ai_resnet) / 2.0
        else: 
            consensus_hr = (hr_pos + ai_resnet) / 2.0

        if snr < 1.0: final_hr = hr_tscan
        elif snr < 2.0: final_hr = 0.6*hr_tscan + 0.4*consensus_hr
        else: final_hr = consensus_hr
        
        final_hr = np.clip(final_hr, 40, 180)

        # HRV & Breathing
        sdnn, rmssd, pnn50, ratio, baevsky_si = dsp.compute_hrv_features(final_seg, real_fps)
        
        # --- BREATHING PIPELINE ---
        br = 15.0
        # 1. FFT Baseline
        br_fft = dsp.predict_breathing_heuristic(final_seg, real_fps)
        
        # 2. SNR Check
        resp_band = dsp.bandpass_filter(clean_pos, fs=real_fps, low=0.1, high=0.4, order=2)
        resp_snr = np.var(resp_band) / (np.var(clean_pos-resp_band) + 1e-6)
        
        # 3. CNN Estimate
        br_cnn = None
        if self.resp_model is not None:
            try:
                resp_input = clean_pos[-256:] 
                if len(resp_input) < 256: resp_input = np.pad(resp_input, (256-len(resp_input),0), mode='edge')
                resp_input = (resp_input - np.mean(resp_input)) / (np.std(resp_input) + 1e-8)
                t_resp = torch.tensor(resp_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.loader.device)
                with torch.no_grad():
                    cnn_out = self.resp_model(t_resp).squeeze().cpu().numpy()
                
                if cnn_out.ndim > 0 and cnn_out.size > 4:
                    br_cnn = dsp.extract_breathing_rate_from_envelope(cnn_out, real_fps)
                else:
                    br_cnn = 6.0 + float(cnn_out) * (35.0 - 6.0) 
            except: pass

        # 4. Consensus
        candidates_br = []
        if 6.0 < br_fft < 35.0: candidates_br.append(br_fft)
        if br_cnn is not None and 6.0 < br_cnn < 35.0: candidates_br.append(br_cnn)
        
        if not candidates_br: br = br_fft
        elif resp_snr > 1.5: br = np.mean(candidates_br)
        else: br = br_fft

        # EMA Smoothing
        br = np.clip(br, 6.0, 35.0)
        alpha = 0.6
        if not self.br_history: final_br = br
        else: final_br = alpha * br + (1 - alpha) * self.br_history[-1]
        self.br_history.append(final_br)
        if len(self.br_history) > 5: self.br_history.pop(0)

        # Stress
        ai_stress = 50.0
        if self.stress_model and sdnn > 0.1:
            try:
                feats = self.generate_extended_features(final_seg, real_fps, final_hr, final_br, sdnn, rmssd, pnn50, ratio)
                ai_stress = float(self.stress_model.predict(feats)[0])
                if ai_stress <= 1.0: ai_stress *= 100
            except: ai_stress = 50.0
            
        physio_stress = min(100, max(0, 20 * np.log(baevsky_si + 1) - 50))
        final_stress = 0.6 * ai_stress + 0.4 * physio_stress

        # Report
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_formatted = (
            "=============================================\n"
            f"🩺 VITALSENSE REPORT - {ts}\n"
            "=============================================\n"
            f"❤️  Heart Rate    : {final_hr:.1f} BPM\n"
            f"    ├─ AI (TSCAN) : {hr_tscan:.1f}\n"
            f"    ├─ Math (POS) : {hr_pos:.1f}\n"
            f"    └─ AI (ResNet): {ai_resnet:.1f}\n"
            f"    └─ SNR        : {snr:.2f}\n"
            f"🫁  Breathing Rate: {final_br:.1f} BrPM\n"
            "---------------------------------------------\n"
            f"📊 HRV (SDNN)    : {sdnn:.1f} ms\n"
            f"📊 HRV (RMSSD)   : {rmssd:.1f} ms\n"
            f"😰 Stress Level   : {final_stress:.1f} / 100\n"
            "=============================================\n"
        )
        print(report_formatted)

        data_packet = {
            "timestamp": time.time(),
            "heart_rate": final_hr,
            "breathing_rate": final_br,
            "hrv_sdnn": sdnn,
            "hrv_rmssd": rmssd,
            "baevsky_si": baevsky_si,
            "stress_final": final_stress,
            "components": {"tscan": hr_tscan, "pos": hr_pos, "resnet": ai_resnet, "snr": float(snr)}
        }
        self.save_report(data_packet, report_formatted)
        
        return {
            "hr": final_hr, 
            "hrv": rmssd, 
            "br": final_br, 
            "stress": final_stress, 
            "waveform": final_seg,
            "fps": real_fps,
            "score": best['score']
        }

    def save_plots(self, waveform, fs, score):
        if not self.output_dir: return
        fig = Figure(figsize=(10, 8), dpi=100); FigureCanvasAgg(fig) 
        fig.patch.set_facecolor('black')
        ax1 = fig.add_subplot(2, 1, 1); ax1.set_facecolor('black')
        ax1.plot(waveform, color='#00ff00', linewidth=2)
        ax1.set_title(f"Extracted Pulse Wave (Score: {score:.2f})", color='white')
        ax1.tick_params(colors='white'); ax1.grid(True, alpha=0.2)
        ax2 = fig.add_subplot(2, 1, 2); ax2.set_facecolor('black')
        n_seg = min(len(waveform), 256)
        if n_seg > 10:
            f, p = dsp.welch(waveform, fs=fs, nperseg=n_seg, nfft=2048)
            mask = (f >= 0.6) & (f <= 3.0)
            ax2.plot(f[mask]*60, p[mask], color='cyan')
        ax2.set_title("Frequency Spectrum", color='white'); ax2.tick_params(colors='white'); ax2.grid(True, alpha=0.2)
        fig.tight_layout(); fig.savefig(os.path.join(self.output_dir, "waveform_analysis.png"), facecolor=fig.get_facecolor())
        plt.close(fig)