import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Backend Imports
from vital_monitor import ProductionVitalMonitor
import config

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class CameraWorker:
    def __init__(self, monitor_instance):
        self.monitor = monitor_instance
        self.cap = None
        self.running = False
        self.recording = False
        self.previewing = False
        
        self.buffer = []
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.start_time = 0
        self.face_detected = False
        self.frame_count = 0

    def start(self):
        self.running = True
        self.previewing = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.recording = False
        self.previewing = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def start_recording(self):
        with self.lock:
            self.buffer = []
            self.frame_count = 0
            self.start_time = time.time()
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False

    def pause_preview(self):
        self.previewing = False

    def resume_preview(self):
        self.previewing = True

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame, self.face_detected

    def _run_loop(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_interval = 1.0 / 30.0
        
        while self.running:
            if not self.previewing:
                time.sleep(0.1)
                continue

            loop_start = time.time()
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: continue

                frame = cv2.flip(frame, 1)
                
                roi = None
                face_found = self.face_detected
                
                should_run_ai = (self.frame_count % 4 == 0) or (not self.face_detected)
                
                if should_run_ai:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = self.monitor.face_mesh.process(rgb)
                    if res.multi_face_landmarks:
                        face_found = True
                        if self.recording:
                            roi = self.monitor.extract_forehead_roi(frame, res.multi_face_landmarks[0])
                    else:
                        face_found = False
                else:
                    if self.recording and face_found:
                        roi = self.monitor.get_last_roi(frame)

                with self.lock:
                    self.latest_frame = frame
                    self.face_detected = face_found
                    
                    if self.recording and roi is not None:
                        self.buffer.append({'fh': roi, 'time': time.time()})
                        if len(self.buffer) >= config.CAPTURE_FRAMES:
                            self.recording = False

                self.frame_count += 1
                
                elapsed = time.time() - loop_start
                wait = frame_interval - elapsed
                if wait > 0: time.sleep(wait)

        if self.cap: self.cap.release()

class VitalSenseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VitalSense AI - Medical Grade Monitor")
        self.geometry("1400x900")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # START INVISIBLE FOR FADE-IN
        self.attributes("-alpha", 0.0)
        
        self.app_state = "LOADING" 
        self.monitor = None
        self.camera = None
        self.running = True
        
        self.grid_columnconfigure(0, weight=3) 
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_ui()
        
        # Animate Fade In
        self.after(100, self.animate_fade_in)
        # Load backend AFTER fade-in starts to keep UI responsive
        self.after(500, self.load_backend)

    def animate_fade_in(self):
        alpha = 0.0
        while alpha < 1.0:
            alpha += 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)

    def animate_fade_out(self):
        alpha = 1.0
        while alpha > 0.0:
            alpha -= 0.05
            self.attributes("-alpha", alpha)
            self.update()
            time.sleep(0.01)

    def setup_ui(self):
        self.cam_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#1a1a1a")
        self.cam_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.cam_frame.grid_rowconfigure(0, weight=1)
        self.cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(self.cam_frame, text="Initializing System...", text_color="gray")
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.cam_frame, height=15)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.progress_bar.set(0)

        self.data_panel = ctk.CTkFrame(self, corner_radius=15, fg_color="#2b2b2b")
        self.data_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        self.data_panel.grid_columnconfigure(0, weight=1)
        self.data_panel.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.data_panel, text="VITAL METRICS", font=("Roboto", 20, "bold")).grid(row=0, column=0, columnspan=2, pady=20)

        self.card_hr = self.create_card("HEART RATE", "--", "BPM", "#e74c3c", 1, 0)
        self.card_hrv = self.create_card("HRV (RMSSD)", "--", "ms", "#3498db", 1, 1)
        self.card_br = self.create_card("BREATHING", "--", "BrPM", "#2ecc71", 2, 0)
        self.card_stress = self.create_card("STRESS LVL", "--", "%", "#f1c40f", 2, 1)

        self.graph_container = ctk.CTkFrame(self.data_panel, fg_color="#1a1a1a")
        self.graph_container.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=20)
        self.data_panel.grid_rowconfigure(3, weight=1)

        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white', labelbottom=False, labelleft=True)
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.set_title("Waiting for Data...", color='gray', fontsize=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status_lbl = ctk.CTkLabel(self.data_panel, text="System Ready", font=("Roboto", 14), text_color="gray")
        self.status_lbl.grid(row=4, column=0, columnspan=2, pady=(10, 5))

        self.btn_action = ctk.CTkButton(self.data_panel, text="START MONITORING", height=50, 
                                      font=("Roboto", 16, "bold"), fg_color="#2980b9", 
                                      command=self.handle_button_click)
        self.btn_action.grid(row=5, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 10))
        self.btn_action.configure(state="disabled")

        self.btn_back = ctk.CTkButton(self.data_panel, text="⬅ BACK TO MENU", height=40, 
                                      font=("Roboto", 14, "bold"), fg_color="#7f8c8d", 
                                      hover_color="#95a5a6",
                                      command=self.return_to_menu)
        self.btn_back.grid(row=6, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 20))

    def create_card(self, title, val, unit, color, r, c):
        f = ctk.CTkFrame(self.data_panel, fg_color="#333333", border_color=color, border_width=2)
        f.grid(row=r, column=c, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(f, text=title, font=("Arial", 12, "bold"), text_color="#aaa").pack(pady=(10,0))
        lbl = ctk.CTkLabel(f, text=val, font=("Arial", 28, "bold"), text_color="white")
        lbl.pack(pady=0)
        ctk.CTkLabel(f, text=unit, font=("Arial", 12), text_color="#aaa").pack(pady=(0,10))
        f.value = lbl
        return f

    def load_backend(self):
        self.status_lbl.configure(text="Loading Models...")
        self.update()
        try:
            self.monitor = ProductionVitalMonitor()
            self.start_new_camera_session()
        except Exception as e:
            self.status_lbl.configure(text=f"Error: {e}", text_color="red")

    def start_new_camera_session(self):
        if self.camera:
            self.camera.stop()
        self.camera = CameraWorker(self.monitor)
        self.camera.start()
        self.app_state = "IDLE"
        self.update_gui_loop()

    def update_gui_loop(self):
        if not self.running or self.app_state in ["PROCESSING", "RESULTS"]:
            return

        frame, face_detected = self.camera.get_latest_frame()
        
        try:
            if not self.winfo_exists(): return
            
            if frame is not None:
                h, w = frame.shape[:2]
                disp_w = self.cam_label.winfo_width()
                disp_h = self.cam_label.winfo_height()
                
                if disp_w > 1 and disp_h > 1:
                    scale = min(disp_w/w, disp_h/h)
                    nw, nh = int(w*scale), int(h*scale)
                    frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(rgb)
                    imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
                    self.cam_label.configure(image=imgtk, text="")
                    self.cam_label.image = imgtk

            if self.app_state == "IDLE":
                if face_detected:
                    self.status_lbl.configure(text="Face Detected - Ready", text_color="#2ecc71")
                    self.btn_action.configure(state="normal", text="START MONITORING", fg_color="#2980b9")
                else:
                    self.status_lbl.configure(text="No Face Detected", text_color="orange")
                    self.btn_action.configure(state="disabled")

            elif self.app_state == "RECORDING":
                with self.camera.lock:
                    rec_count = len(self.camera.buffer)
                    is_recording = self.camera.recording
                
                prog = rec_count / config.CAPTURE_FRAMES
                self.progress_bar.set(prog)
                self.status_lbl.configure(text=f"Recording... {rec_count}/{config.CAPTURE_FRAMES}", text_color="#3498db")
                
                if not is_recording and rec_count >= 10:
                    self.finish_recording()

            self.after(33, self.update_gui_loop)
            
        except Exception:
            pass

    def handle_button_click(self):
        if self.app_state == "IDLE":
            self.app_state = "RECORDING"
            self.btn_action.configure(text="STOP RECORDING", fg_color="#c0392b")
            self.camera.start_recording()
            
        elif self.app_state == "RECORDING":
            self.camera.stop_recording()
            self.finish_recording()
            
        elif self.app_state == "RESULTS":
            self.app_state = "IDLE"
            self.progress_bar.set(0)
            self.ax.clear()
            self.ax.set_facecolor('#1a1a1a')
            self.ax.set_title("Waiting for Data...", color='gray')
            self.canvas.draw()
            for card in [self.card_hr, self.card_hrv, self.card_br, self.card_stress]:
                card.value.configure(text="--")
            self.start_new_camera_session()

    def finish_recording(self):
        self.app_state = "PROCESSING"
        self.camera.stop()
        
        self.status_lbl.configure(text="Processing Data... Please Wait", text_color="yellow")
        self.cam_label.configure(image=None, text="⚠️ PROCESSING DATA ⚠️")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.btn_action.configure(state="disabled")
        
        threading.Thread(target=self.run_analysis).start()

    def run_analysis(self):
        with self.camera.lock:
            raw_buffer = self.camera.buffer[:]
        
        valid_frames = len(raw_buffer)
        if valid_frames < 30:
            self.after(0, lambda: self.show_error("Not enough data. Try again."))
            return

        np_buffer = np.zeros(valid_frames, dtype=config.DTYPE_BUFFER)
        for i, item in enumerate(raw_buffer):
            np_buffer[i]['fh'] = item['fh']
            np_buffer[i]['time'] = item['time']

        try:
            results = self.monitor.process_full_recording(np_buffer, valid_frames)
            self.after(0, lambda: self.show_results(results))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.show_error(err_msg))

    def show_results(self, res):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1)
        
        if not res:
            self.show_error("No result returned from backend")
            return

        self.card_hr.value.configure(text=f"{res['hr']:.1f}")
        self.card_hrv.value.configure(text=f"{res['hrv']:.1f}")
        self.card_br.value.configure(text=f"{res['br']:.1f}")
        self.card_stress.value.configure(text=f"{res['stress']:.1f}")
        
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        self.ax.plot(res['waveform'], color='#00ff00', linewidth=1.5)
        self.ax.set_title("Pulse Waveform (30s)", color='white')
        self.ax.grid(True, color='#333')
        self.canvas.draw()
        
        self.monitor.save_plots(res['waveform'], res['fps'], res['score'])
        
        self.app_state = "RESULTS"
        self.status_lbl.configure(text="Analysis Complete", text_color="#2ecc71")
        self.btn_action.configure(state="normal", text="START NEW SCAN", fg_color="#2980b9")

    def show_error(self, msg):
        self.progress_bar.stop()
        self.app_state = "RESULTS"
        self.status_lbl.configure(text=f"Error: {msg}", text_color="red")
        self.btn_action.configure(state="normal", text="TRY AGAIN")

    def return_to_menu(self):
        self.running = False
        if self.camera:
            self.camera.stop()
        
        # Animate Fade Out first
        #self.animate_fade_out()
        self.destroy() 
        import launcher
        app = launcher.LauncherApp(skip_loading=True)
        app.mainloop()

    def on_close(self):
        self.running = False
        if self.camera:
            self.camera.stop()
        self.quit()
        self.destroy()
        import sys
        sys.exit()

if __name__ == "__main__":
    app = VitalSenseApp()
    app.mainloop()