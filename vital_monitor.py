# vital_monitor.py
import sys
import os
import json
import time
import datetime
import cv2
import numpy as np
import mediapipe as mp
import torch
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import skew, kurtosis 

import config
import models_loading
import signal_processing as dsp

class ProductionVitalMonitor:
    def __init__(self):
        print("🚀 Initializing Production Vital Monitor...")
        self.loader = models_loading.ModelLoader()
        
        # Load Models
        self.tscan = self.loader.load_tscan()
        self.resnet = self.loader.load_resnet()
        self.resp_model = self.loader.load_breathing_model() 
        self.stress_model = self.loader.load_stress_model() 
        
        # Face Detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.roi_state = None 

    def create_output_directory(self):
        """Creates outputs/YYYY-MM-DD/HH-MM-SS/ structure"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Build path: outputs/2026-01-06/14-30-00/
        self.output_dir = os.path.join("outputs", date_str, time_str)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 Output Directory Created: {self.output_dir}")
        return self.output_dir

    def save_report(self, data_dict, report_text):
        """Saves textual report and raw JSON data"""
        # 1. Save JSON Data
        json_path = os.path.join(self.output_dir, "vital_data.json")
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            
        # 2. Save Human Readable Report
        txt_path = os.path.join(self.output_dir, "report.txt")
        with open(txt_path, 'w') as f:
            f.write(report_text)
            
        print(f"💾 Data Saved to: {self.output_dir}")

    # ... [Keep extract_forehead_roi, run_tscan_inference, calculate_physio_score UNCHANGED] ...
    # (Copy them from the previous version, or I can include them if you prefer. 
    #  For brevity, I assume you kept the previous helper methods and just need the logic update below)
    
    def extract_forehead_roi(self, frame, landmarks):
        # ... (Same as before) ...
        h_img, w_img, _ = frame.shape
        fh_indices = [109, 10, 338, 299, 296, 336, 9]
        pts = np.array([[int(landmarks.landmark[p].x * w_img), int(landmarks.landmark[p].y * h_img)] for p in fh_indices])
        x, y, w, h = cv2.boundingRect(pts)
        if self.roi_state is None: self.roi_state = [float(x), float(y), float(w), float(h)]
        else:
            alpha = 0.9  
            self.roi_state[0] = alpha * self.roi_state[0] + (1-alpha) * x
            self.roi_state[1] = alpha * self.roi_state[1] + (1-alpha) * y
            self.roi_state[2] = alpha * self.roi_state[2] + (1-alpha) * w
            self.roi_state[3] = alpha * self.roi_state[3] + (1-alpha) * h
        sx, sy, sw, sh = [int(v) for v in self.roi_state]
        sx = max(0, sx); sy = max(0, sy)
        if sw < 5 or sh < 5: return None
        crop = frame[sy:sy+sh, sx:sx+sw]
        if crop.size == 0: return None
        return cv2.resize(crop, (72, 72))

    def run_tscan_inference(self, buffer_slice):
        # ... (Same as before) ...
        r_mean = np.mean(buffer_slice['fh'][:, :, :, 2], axis=(1, 2))
        g_mean = np.mean(buffer_slice['fh'][:, :, :, 1], axis=(1, 2))
        b_mean = np.mean(buffer_slice['fh'][:, :, :, 0], axis=(1, 2))
        raw = buffer_slice['fh'].astype(np.float32) / 255.0
        raw_padded = np.concatenate([raw[0:1], raw], axis=0)
        diff = (raw_padded[1:] - raw_padded[:-1]) / (raw_padded[1:] + raw_padded[:-1] + 1e-6)
        raw_t = torch.from_numpy(raw).float().unsqueeze(0).permute(0, 4, 1, 2, 3).to(self.loader.device)
        diff_t = torch.from_numpy(diff).float().unsqueeze(0).permute(0, 4, 1, 2, 3).to(self.loader.device)
        with torch.no_grad(): ppg_out = self.tscan(diff_t, raw_t)
        return ppg_out.cpu().numpy().flatten(), r_mean, g_mean, b_mean

    def calculate_physio_score(self, segment_ppg, segment_red, segment_green, fs):
        # ... (Same as before) ...
        f, p = dsp.welch(segment_ppg, fs=fs, nfft=1024)
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
        # ... (Same as before) ...
        peaks, _ = dsp.find_peaks(ppg_signal, distance=fs*0.5)
        if len(peaks) < 2: return np.zeros((1, 18))
        rr_ms = np.diff(peaks) / fs * 1000.0 
        mean_rr = np.mean(rr_ms); median_rr = np.median(rr_ms)
        min_rr = np.min(rr_ms); max_rr = np.max(rr_ms); std_rr = np.std(rr_ms)
        diff_rr = np.diff(rr_ms)
        sd1 = np.sqrt(np.std(diff_rr, ddof=1)**2 * 0.5)
        sd2 = np.sqrt(2 * std_rr**2 - 0.5 * np.std(diff_rr, ddof=1)**2)
        skew_rr = skew(rr_ms); kurt_rr = kurtosis(rr_ms)
        f, p = dsp.welch(rr_ms/1000.0, fs=4.0, nperseg=min(len(rr_ms), 256))
        vlf_pow = np.trapz(p[(f < 0.04)], f[(f < 0.04)])
        lf_pow = np.trapz(p[(f>=0.04)&(f<0.15)], f[(f>=0.04)&(f<0.15)])
        hf_pow = np.trapz(p[(f>=0.15)&(f<0.4)], f[(f>=0.15)&(f<0.4)])
        total_pow = vlf_pow + lf_pow + hf_pow
        features = np.array([
            hr, sdnn, rmssd, pnn50, lf_hf_ratio, br, 
            mean_rr, median_rr, min_rr, max_rr,
            lf_pow, hf_pow, vlf_pow, total_pow,
            sd1, sd2, skew_rr, kurt_rr
        ])
        return np.nan_to_num(features).reshape(1, -1)

    def process_full_recording(self, buffer, valid_frames):
        print("\n" + "="*40 + "\n🔄 PROCESSING DATA\n" + "="*40)
        
        # 1. Setup Logging
        self.create_output_directory()
        
        times = buffer['time'][:valid_frames]
        duration = times[-1] - times[0]
        real_fps = valid_frames / duration
        print(f"⏱️  True Duration: {duration:.2f}s | Real FPS: {real_fps:.2f}")

        # 2. Extract Signals
        full_ppg, full_r, full_g, full_b = [], [], [], []
        num_batches = valid_frames // config.BATCH_SIZE
        print(f"🔹 Extracting Signals ({num_batches} batches)...")
        for i in range(num_batches):
            chunk = buffer[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE]
            ppg, r, g, b = self.run_tscan_inference(chunk)
            full_ppg.extend(ppg); full_r.extend(r); full_g.extend(g); full_b.extend(b)
            sys.stdout.write(f"\r   Batch {i+1}/{num_batches} done...")
        print("\n")

        # 3. Clean & Window
        clean_ppg = dsp.clean_signal(np.array(full_ppg), fs=real_fps)
        clean_pos = dsp.run_pos_algorithm(np.array(full_r), np.array(full_g), np.array(full_b), fs=real_fps)
        clean_g = dsp.clean_signal(np.array(full_g), fs=real_fps)
        clean_r = dsp.clean_signal(np.array(full_r), fs=real_fps)

        win_frames = int(12.0 * real_fps)
        checkpoints = np.linspace(0, len(clean_ppg) - win_frames, 5, dtype=int)
        candidates = []
        for i in checkpoints:
            seg_ppg = clean_ppg[i : i+win_frames]
            seg_pos = clean_pos[i : i+win_frames]
            score = self.calculate_physio_score(seg_ppg, clean_r[i:i+win_frames], clean_g[i:i+win_frames], fs=real_fps)
            candidates.append({'segment': seg_ppg, 'segment_pos': seg_pos, 'score': score})
            
        best = max(candidates, key=lambda x: x['score'])
        final_seg = best['segment']
        final_pos = best['segment_pos']

        # 4. Consensus & Calculation
        seg_norm = (final_seg - np.mean(final_seg)) / (np.std(final_seg) + 1e-6)
        t_seg = torch.tensor(seg_norm, dtype=torch.float32).view(1, 1, -1).to(self.loader.device)
        with torch.no_grad(): ai_hr = self.resnet(t_seg).item()
        spec_hr = dsp.get_dominant_freq(final_seg, real_fps)
        pos_hr = dsp.get_dominant_freq(final_pos, real_fps)

        diffs = [abs(spec_hr - pos_hr), abs(spec_hr - ai_hr), abs(pos_hr - ai_hr)]
        best_pair = np.argmin(diffs)
        if diffs[best_pair] > 15.0: final_hr = np.mean([spec_hr, pos_hr, ai_hr])
        elif best_pair == 0: final_hr = (spec_hr + pos_hr) / 2.0
        elif best_pair == 1: final_hr = (spec_hr + ai_hr) / 2.0
        else: final_hr = (pos_hr + ai_hr) / 2.0

        sdnn, rmssd, pnn50, ratio = dsp.compute_hrv_features(final_seg, real_fps)
        
        # 5. ML Models
        if self.resp_model:
            try:
                dtest = xgb.DMatrix(np.array([final_hr, sdnn, rmssd]).reshape(1, -1))
                br = float(self.resp_model.predict(dtest)[0])
            except: br = dsp.predict_breathing_heuristic(final_seg, real_fps)
        else: br = dsp.predict_breathing_heuristic(final_seg, real_fps)

        stress_val = 50.0
        if self.stress_model:
            try:
                stress_feats = self.generate_extended_features(final_seg, real_fps, final_hr, br, sdnn, rmssd, pnn50, ratio)
                stress_val = float(self.stress_model.predict(stress_feats)[0])
                if stress_val <= 1.0: stress_val *= 100
            except Exception as e:
                stress_val = max(0, min(100, 50 + (final_hr/100 - sdnn/200)*20))
        else:
            stress_val = max(0, min(100, 50 + (final_hr/100 - sdnn/200)*20))

        # 6. REPORT GENERATION
        report = (
            f"=============================================\n"
            f"🩺 VITALSENSE REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=============================================\n"
            f"❤️  Heart Rate    : {final_hr:.1f} BPM\n"
            f"    ├─ AI (TSCAN) : {spec_hr:.1f}\n"
            f"    ├─ Math (POS) : {pos_hr:.1f}\n"
            f"    └─ AI (ResNet): {ai_hr:.1f}\n"
            f"🫁  Breathing Rate: {br:.1f} BrPM\n"
            f"---------------------------------------------\n"
            f"📊 HRV (SDNN)    : {sdnn:.1f} ms\n"
            f"📊 HRV (RMSSD)   : {rmssd:.1f} ms\n"
            f"😰 Stress Level   : {stress_val:.1f} / 100\n"
            f"=============================================\n"
        )
        print(report)
        
        # Save Text and JSON
        data_packet = {
            "timestamp": time.time(),
            "heart_rate": final_hr,
            "breathing_rate": br,
            "hrv_sdnn": sdnn,
            "hrv_rmssd": rmssd,
            "stress_level": stress_val,
            "components": {"tscan": spec_hr, "pos": pos_hr, "resnet": ai_hr}
        }
        self.save_report(data_packet, report)
        
        # 7. Plotting & Saving
        self.save_plots(final_seg, real_fps, best['score'])

    def save_plots(self, waveform, fs, score):
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(waveform, color='#00ff00', linewidth=2)
        ax1.set_title(f"Extracted Pulse Wave (Quality Score: {score:.2f})")
        ax1.grid(True, alpha=0.2)
        f, p = dsp.welch(waveform, fs=fs, nfft=2048)
        mask = (f >= 0.6) & (f <= 3.0)
        ax2.plot(f[mask]*60, p[mask], color='cyan')
        ax2.set_title("Frequency Spectrum")
        ax2.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # Save instead of just showing
        plot_path = os.path.join(self.output_dir, "waveform_analysis.png")
        plt.savefig(plot_path)
        print(f"📈 Graph Saved to: {plot_path}")
        
        plt.show() # Still show it on screen

    def run(self, source=0):
        # ... (Same as before) ...
        cap = cv2.VideoCapture(source)
        buffer = np.zeros(config.CAPTURE_FRAMES, dtype=config.DTYPE_BUFFER)
        frame_idx = 0
        last_fh = None
        print(f"\n🚀 Starting Capture on Source {source}")
        print(f"🎯 Target: {config.CAPTURE_FRAMES} frames (~30s)\n")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % 3 == 0:
                    res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if res.multi_face_landmarks:
                        fh = self.extract_forehead_roi(frame, res.multi_face_landmarks[0])
                        if fh is not None: last_fh = fh
                elif self.roi_state:
                    sx, sy, sw, sh = [int(v) for v in self.roi_state]
                    sx = max(0, sx); sy = max(0, sy)
                    if sw>5 and sh>5:
                        crop = frame[sy:sy+sh, sx:sx+sw]
                        if crop.size!=0: last_fh = cv2.resize(crop, (72, 72))
                if last_fh is not None:
                    buffer[frame_idx]['fh'] = last_fh
                    buffer[frame_idx]['time'] = time.time()
                    frame_idx += 1
                    sys.stdout.write(f"\r📸 Capturing: {frame_idx}/{config.CAPTURE_FRAMES}")
                    sys.stdout.flush()
                if frame_idx >= config.CAPTURE_FRAMES: break
        except KeyboardInterrupt: pass
        finally:
            cap.release()
            if frame_idx >= config.BATCH_SIZE: self.process_full_recording(buffer[:frame_idx], frame_idx)
            else: print(f"\n❌ Not enough data.")