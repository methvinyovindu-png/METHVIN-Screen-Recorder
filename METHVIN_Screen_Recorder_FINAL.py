import sys
import os
import time
import datetime
import threading
import queue
import subprocess
import wave
import json
import zipfile
import numpy as np

try:
    import mss
    import mss.tools
    import cv2
    import sounddevice as sd
    from pynput import keyboard
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QDateTimeEdit, QVBoxLayout, QDialog, QScrollArea, QDoubleSpinBox,
                                   QHBoxLayout, QPushButton, QLabel, QComboBox,
                                   QFileDialog, QMessageBox, QTabWidget, QCheckBox,
                                   QSystemTrayIcon, QMenu, QGroupBox, QSpinBox, QSlider, QSizePolicy, QListWidget, QListWidgetItem)
    from PySide6.QtCore import (Qt, QTimer, QThread, Signal, Slot, QPoint, QRect, QSize, QDateTime)
    from PySide6.QtGui import (QIcon, QPixmap, QImage, QPainter, QColor, QPen,
                               QBrush, QAction, QFont, QCursor)
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you have installed all dependencies: pip install -r requirements.txt")
    sys.exit(1)

APP_NAME = "METHVIN Screen Recorder"
DEFAULT_FPS = 30
AUDIO_SAMPLE_RATE = 44100
CHANNELS = 2

def _no_console_kwargs():
    """Prevent child console windows on Windows while keeping compatibility elsewhere."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return {"creationflags": flags}

def get_ffmpeg_path():
    """Verify if FFmpeg is installed and accessible via system PATH or current directory."""
    # Check local folder first (e.g. bundled alongside .exe)
    local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    
    # Check system PATH
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        check=True, **_no_console_kwargs())
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_audio_input_devices():
    """Detect available microphones and Windows WASAPI loopback audio devices."""
    mics = []
    system_devices = []
    
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        for idx, dev in enumerate(devices):
            # Microphone / Input devices
            if dev['max_input_channels'] > 0:
                mics.append((idx, dev['name']))
            
            # WASAPI output devices for system audio loopback on Windows
            host_api_idx = dev['hostapi']
            if host_api_idx < len(hostapis):
                host_api_name = hostapis[host_api_idx]['name']
                if 'WASAPI' in host_api_name and dev['max_output_channels'] > 0:
                    system_devices.append((idx, dev['name']))
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        
    return mics, system_devices


class VideoCaptureThread(QThread):
    """Real-time screen capture worker.

    Frames are timestamped so a slow capture loop cannot make the saved video
    play faster than real time. Optional camera frames are composited into the
    screen frame before it is handed to the recorder.
    """
    frame_ready = Signal(object, float)
    error_occurred = Signal(str)

    def __init__(self, monitor_rect, fps, camera_enabled=False,
                 camera_index=0, camera_scale=0.25, camera_position="Bottom Right", camera_track_path=None, cursor_highlight=False):
        super().__init__()
        self.monitor_rect = monitor_rect
        self.fps = max(1, int(fps))
        self.camera_enabled = camera_enabled
        self.camera_index = camera_index
        self.camera_scale = float(camera_scale)
        self.camera_position = camera_position
        self.camera_track_path = camera_track_path
        self.cursor_highlight = bool(cursor_highlight)
        self.camera_writer = None
        self.is_recording = False
        self.is_paused = False
        self.sct = mss.mss()
        self.frame_time = 1.0 / self.fps
        self.camera = None

    def _open_camera(self):
        if not self.camera_enabled:
            return
        try:
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                self.camera = None
                self.error_occurred.emit("Camera could not be opened. Screen recording will continue without camera.")
        except Exception as e:
            self.camera = None
            self.error_occurred.emit(f"Camera initialization failed: {e}")

    def _overlay_camera(self, frame):
        if self.camera is None:
            return frame
        ok, cam = self.camera.read()
        if not ok or cam is None:
            return frame

        if self.camera_track_path and self.camera_writer is None:
            ch0, cw0 = cam.shape[:2]
            fourcc=cv2.VideoWriter_fourcc(*'MJPG')
            self.camera_writer=cv2.VideoWriter(self.camera_track_path, fourcc, float(self.fps), (cw0 - cw0%2, ch0 - ch0%2))
            if not self.camera_writer.isOpened():
                self.camera_writer.release(); self.camera_writer=None
        if self.camera_writer is not None:
            try:
                ch1,cw1=cam.shape[:2]
                cam_for_write=cam[:ch1-ch1%2,:cw1-cw1%2]
                self.camera_writer.write(cam_for_write)
            except Exception:
                pass

        h, w = frame.shape[:2]
        ch, cw = cam.shape[:2]
        target_w = max(120, int(w * self.camera_scale))
        target_h = max(90, int(ch * target_w / max(1, cw)))
        if target_h >= h or target_w >= w:
            return frame

        cam = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_AREA)
        margin = max(10, int(min(w, h) * 0.02))
        if self.camera_position == "Bottom Left":
            x, y = margin, h - target_h - margin
        elif self.camera_position == "Top Left":
            x, y = margin, margin
        elif self.camera_position == "Top Right":
            x, y = w - target_w - margin, margin
        else:
            x, y = w - target_w - margin, h - target_h - margin

        # Rounded corners are approximated with a border; this keeps the
        # overlay compatible with OpenCV/FFmpeg without GUI dependencies.
        x = max(0, min(x, w - target_w))
        y = max(0, min(y, h - target_h))
        frame[y:y + target_h, x:x + target_w] = cam
        cv2.rectangle(frame, (x - 2, y - 2),
                      (x + target_w + 1, y + target_h + 1), (255, 255, 255), 2)
        return frame

    def _draw_cursor(self, frame):
        if not self.cursor_highlight or os.name != "nt": return frame
        try:
            import ctypes
            class POINT(ctypes.Structure): _fields_=[("x",ctypes.c_long),("y",ctypes.c_long)]
            pt=POINT(); ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            x=int(pt.x-self.monitor_rect.get("left",0)); y=int(pt.y-self.monitor_rect.get("top",0))
            if 0<=x<frame.shape[1] and 0<=y<frame.shape[0]:
                cv2.circle(frame,(x,y),18,(0,210,255),3,cv2.LINE_AA); cv2.circle(frame,(x,y),4,(255,255,255),-1,cv2.LINE_AA)
        except Exception: pass
        return frame

    def run(self):
        self.is_recording = True
        self._open_camera()
        next_frame_time = time.perf_counter()

        try:
            while self.is_recording:
                if self.is_paused:
                    time.sleep(0.02)
                    next_frame_time = time.perf_counter()
                    continue

                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                capture_ts = time.perf_counter()

                sct_img = self.sct.grab(self.monitor_rect)
                img = np.array(sct_img)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                frame = self._overlay_camera(frame)
                frame = self._draw_cursor(frame)

                self.frame_ready.emit(frame, capture_ts)

                # Schedule from the previous deadline instead of from the
                # end of capture. This prevents drift when MSS takes time.
                next_frame_time += self.frame_time
                if next_frame_time < time.perf_counter() - self.frame_time:
                    next_frame_time = time.perf_counter() + self.frame_time

        except Exception as e:
            self.error_occurred.emit(f"Video capture error: {str(e)}")
        finally:
            if self.camera_writer is not None:
                try: self.camera_writer.release()
                except Exception: pass
                self.camera_writer=None
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception:
                    pass
                self.camera = None

    def stop(self):
        self.is_recording = False
        self.wait()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


class AudioCaptureThread(QThread):
    """Thread for capturing microphone or system audio (WASAPI loopback) into uncompressed WAV files."""
    error_occurred = Signal(str)

    def __init__(self, temp_wav_path, device_index=None, is_loopback=False):
        super().__init__()
        self.temp_wav_path = temp_wav_path
        self.device_index = device_index
        self.is_loopback = is_loopback
        self.is_recording = False
        self.is_paused = False
        self.q = queue.Queue()
        self.stream = None
        self.wav_file = None

    def callback(self, indata, frames, time_info, status):
        """Callback invoked by sounddevice for each chunk of incoming PCM audio."""
        if status:
            print(f"Audio buffer status: {status}", file=sys.stderr)
        if not self.is_paused:
            self.q.put(indata.copy())

    def run(self):
        self.is_recording = True
        try:
            # Initialize PCM 16-bit stereo WAV writer
            self.wav_file = wave.open(self.temp_wav_path, 'wb')
            self.wav_file.setnchannels(CHANNELS)
            self.wav_file.setsampwidth(2)
            self.wav_file.setframerate(AUDIO_SAMPLE_RATE)

            extra_settings = None
            if self.is_loopback:
                try:
                    extra_settings = sd.WasapiSettings(loopback=True)
                except Exception as e:
                    print(f"WASAPI loopback not supported on device: {e}")

            # Start audio capture stream
            self.stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                device=self.device_index,
                channels=CHANNELS,
                dtype='int16',
                callback=self.callback,
                extra_settings=extra_settings
            )
            
            with self.stream:
                while self.is_recording:
                    try:
                        data = self.q.get(timeout=0.1)
                        if self.wav_file and not self.is_paused:
                            self.wav_file.writeframes(data.tobytes())
                    except queue.Empty:
                        continue
        except Exception as e:
            audio_type = "System Audio" if self.is_loopback else "Microphone"
            self.error_occurred.emit(f"{audio_type} capture error: {str(e)}")
        finally:
            if self.wav_file:
                self.wav_file.close()
                self.wav_file = None

    def stop(self):
        self.is_recording = False
        self.wait()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


class RecorderManager(QThread):
    """Main orchestrator thread: coordinates video/audio workers and executes FFmpeg encoding."""
    recording_finished = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)

    def __init__(self, output_path, monitor_rect, fps, quality, record_mic, mic_dev_idx, record_sys_audio, sys_dev_idx, mic_volume=1.0, camera_enabled=False, camera_index=0, camera_scale=0.25, camera_position='Bottom Right', output_resolution='Native', smart_cleanup=False, auto_enhance=False, system_volume=1.0, cursor_highlight=False):
        super().__init__()
        self.output_path = output_path
        self.monitor_rect = monitor_rect
        self.fps = fps
        self.quality = quality
        self.record_mic = record_mic
        self.mic_dev_idx = mic_dev_idx
        self.record_sys_audio = record_sys_audio
        self.sys_dev_idx = sys_dev_idx
        self.mic_volume = max(0.0, min(2.0, float(mic_volume)))
        self.camera_enabled = bool(camera_enabled)
        self.camera_index = int(camera_index)
        self.camera_scale = float(camera_scale)
        self.camera_position = camera_position
        self.output_resolution = output_resolution
        self.smart_cleanup = bool(smart_cleanup)
        self.auto_enhance = bool(auto_enhance)
        self.system_volume = max(0.0, min(2.0, float(system_volume)))
        self.cursor_highlight = bool(cursor_highlight)
        self.capture_start_time = None
        self.last_capture_time = None
        self.frames_written = 0
        self.is_recording = False
        self.is_paused = False
        
        # Temporary scratch files
        temp_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        self.temp_video = os.path.join(temp_dir, f"{base_name}_temp.avi")
        self.temp_camera = os.path.join(temp_dir, f"{base_name}_camera_temp.avi")
        self.temp_mic_wav = os.path.join(temp_dir, f"{base_name}_mic_temp.wav")
        self.temp_sys_wav = os.path.join(temp_dir, f"{base_name}_sys_temp.wav")
        
        self.video_thread = None
        self.mic_thread = None
        self.sys_thread = None
        self.video_writer = None

        self.ffmpeg_path = get_ffmpeg_path()

    @Slot(object, float)
    def on_frame_ready(self, frame, capture_ts):
        """Write frames according to elapsed wall-clock time.

        The old implementation wrote one video frame per callback while
        declaring a fixed FPS. If MSS captured slower than that FPS, the MP4
        duration became too short and playback looked sped up. We compensate
        by duplicating frames when necessary so frame count tracks real time.
        """
        if not self.is_recording or self.is_paused or self.video_writer is None:
            return

        try:
            if self.capture_start_time is None:
                self.capture_start_time = capture_ts
                self.last_capture_time = capture_ts
                self.frames_written = 0
                self.video_writer.write(frame)
                self.frames_written = 1
                return

            elapsed = max(0.0, capture_ts - self.capture_start_time)
            target_frames = max(1, int(round(elapsed * float(self.fps))) + 1)
            frames_to_write = max(1, target_frames - self.frames_written)
            # Avoid a huge burst if the computer briefly stalls.
            frames_to_write = min(frames_to_write, max(1, self.fps * 2))

            for _ in range(frames_to_write):
                self.video_writer.write(frame)
            self.frames_written += frames_to_write
            self.last_capture_time = capture_ts
        except Exception as e:
            print(f"Error appending video frame: {e}")

    def run(self):
        if not self.ffmpeg_path:
            self.error_occurred.emit("FFmpeg was not detected. It is required to render final MP4 recordings.")
            return

        self.is_recording = True
        self.status_update.emit("Initializing screen capture engine...")

        try:
            # OpenCV raw video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            width = self.monitor_rect['width']
            height = self.monitor_rect['height']
            
            # Ensure even dimensional bounds for H.264 macroblock compliance
            if width % 2 != 0: width -= 1
            if height % 2 != 0: height -= 1
            
            self.video_writer = cv2.VideoWriter(self.temp_video, fourcc, float(self.fps), (width, height))
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to initialize OpenCV VideoWriter buffer.")

            # Start Video Capture Worker Thread
            self.video_thread = VideoCaptureThread(self.monitor_rect, self.fps, self.camera_enabled, self.camera_index, self.camera_scale, self.camera_position, self.temp_camera if self.camera_enabled else None, self.cursor_highlight)
            self.video_thread.frame_ready.connect(self.on_frame_ready)
            self.video_thread.error_occurred.connect(self.error_occurred.emit)
            self.video_thread.start()

            # Start Microphone Worker Thread
            if self.record_mic:
                self.mic_thread = AudioCaptureThread(self.temp_mic_wav, device_index=self.mic_dev_idx, is_loopback=False)
                self.mic_thread.error_occurred.connect(self.error_occurred.emit)
                self.mic_thread.start()

            # Start System Audio Loopback Worker Thread
            if self.record_sys_audio:
                self.sys_thread = AudioCaptureThread(self.temp_sys_wav, device_index=self.sys_dev_idx, is_loopback=True)
                self.sys_thread.error_occurred.connect(self.error_occurred.emit)
                self.sys_thread.start()

            self.status_update.emit("Recording in progress...")
            
            while self.is_recording:
                time.sleep(0.1)

        except Exception as e:
            self.error_occurred.emit(f"Recorder launch failed: {e}")
            self.cleanup()
            return

        # --- Stop Captured Streams & Encode to MP4 ---
        self.status_update.emit("Finalizing media encoding with FFmpeg...")
        
        if self.video_thread:
            self.video_thread.stop()
        if self.mic_thread:
            self.mic_thread.stop()
        if self.sys_thread:
            self.sys_thread.stop()

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        try:
            self.mux_and_encode()
        except Exception as e:
            self.error_occurred.emit(f"FFmpeg encoding failed: {e}")
        finally:
            self.cleanup_temp_files()
            self.recording_finished.emit(self.output_path)

    def mux_and_encode(self):
        """Encode the screen track, optional camera track and independent audio tracks."""
        crf_map={"High":"18","Medium":"23","Low":"28"}; crf=crf_map.get(self.quality,"23")
        if not os.path.exists(self.temp_video): raise Exception("Temporary video buffer not found.")
        has_cam=self.camera_enabled and os.path.exists(self.temp_camera)
        has_mic=self.record_mic and os.path.exists(self.temp_mic_wav)
        has_sys=self.record_sys_audio and os.path.exists(self.temp_sys_wav)
        cmd=[self.ffmpeg_path,'-y','-i',self.temp_video]
        if has_cam: cmd += ['-i',self.temp_camera]
        if has_mic: cmd += ['-i',self.temp_mic_wav]
        if has_sys: cmd += ['-i',self.temp_sys_wav]
        video_filters=[]
        res_map={'1920x1080':(1920,1080),'2560x1440':(2560,1440),'3840x2160 (4K)':(3840,2160),'7680x4320 (8K)':(7680,4320)}
        if self.output_resolution in res_map:
            w,h=res_map[self.output_resolution]
            video_filters.append(f'scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2')
        if self.auto_enhance: video_filters.append('eq=contrast=1.03:saturation=1.04:brightness=0.01')
        if video_filters: cmd += ['-vf',','.join(video_filters)]
        cmd += ['-map','0:v:0','-c:v:0','libx264','-crf',crf,'-preset','fast','-pix_fmt','yuv420p']
        next_input=1
        if has_cam:
            cmd += ['-map',f'{next_input}:v:0','-c:v:1','libx264','-crf',crf,'-preset','fast','-pix_fmt','yuv420p','-metadata:s:v:1','title=Camera Track']
            next_input += 1
        audio_inputs=[]
        if has_mic:
            af=f'volume={self.mic_volume:.3f}'
            if self.smart_cleanup: af += ',afftdn=nf=-25,highpass=f=70,lowpass=f=14000'
            audio_inputs.append((next_input,'Microphone Track',af)); next_input+=1
        if has_sys:
            audio_inputs.append((next_input,'System Audio Track',f'volume={self.system_volume:.3f}')); next_input+=1
        if audio_inputs:
            filters=[]
            maps=[]
            for idx,(ai,title,af) in enumerate(audio_inputs):
                label=f'a{idx}'
                filters.append(f'[{ai}:a:0]{af}[{label}]')
                maps += ['-map',f'[{label}]','-c:a:'+str(idx),'aac','-b:a:'+str(idx),'192k','-metadata:s:a:'+str(idx),f'title={title}']
            cmd += ['-filter_complex',';'.join(filters)] + maps
        else:
            cmd += ['-an']
        cmd += ['-movflags','+faststart',self.output_path]
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**_no_console_kwargs())
        _,err=process.communicate()
        if process.returncode!=0:
            raise Exception('FFmpeg error:\n'+err.decode('utf-8',errors='ignore')[-1200:])

    def stop(self):
        self.is_recording = False

    def pause(self):
        self.is_paused = True
        if self.video_thread: self.video_thread.pause()
        if self.mic_thread: self.mic_thread.pause()
        if self.sys_thread: self.sys_thread.pause()

    def resume(self):
        self.is_paused = False
        if self.video_thread: self.video_thread.resume()
        if self.mic_thread: self.mic_thread.resume()
        if self.sys_thread: self.sys_thread.resume()

    def cleanup(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.cleanup_temp_files()

    def cleanup_temp_files(self):
        for temp_file in [self.temp_video, self.temp_camera, self.temp_mic_wav, self.temp_sys_wav]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    print(f"Could not remove scratch file {temp_file}: {e}")


class RegionSelector(QWidget):
    """Transparent desktop overlay for interactive rectangular area selection."""
    region_selected = Signal(QRect)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        # Calculate combined screen bounds across multi-monitor setups
        desktop = QApplication.primaryScreen().availableGeometry()
        for screen in QApplication.screens():
            desktop = desktop.united(screen.geometry())
        self.setGeometry(desktop)

        self.start_point = QPoint()
        self.current_rect = QRect()
        self.is_drawing = False
        
        # Guidance banner
        self.prompt_label = QLabel("Click and drag to select recording rectangle.\nPress ESC to cancel.", self)
        self.prompt_label.setStyleSheet("""
            color: #ffffff; 
            background-color: rgba(15, 23, 42, 220); 
            border: 1px solid #3b82f6;
            padding: 12px 20px; 
            border-radius: 8px; 
            font-size: 15px; 
            font-weight: bold;
        """)
        self.prompt_label.adjustSize()
        self.prompt_label.move(self.width() // 2 - self.prompt_label.width() // 2, 60)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Semi-transparent backdrop overlay
        painter.fillRect(self.rect(), QColor(15, 23, 42, 140))

        if not self.current_rect.isNull():
            # Clear selected selection hole
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.current_rect, Qt.GlobalColor.transparent)
            
            # Highlight border
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen(QColor(59, 130, 246), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()
            self.is_drawing = True
            self.current_rect = QRect()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.current_rect = QRect(self.start_point, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            if self.current_rect.width() > 10 and self.current_rect.height() > 10:
                self.region_selected.emit(self.current_rect)
                self.close()
            else:
                self.current_rect = QRect()
                self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


class WindowPicker(QWidget):
    """Simple Windows window picker backed by the Win32 API when available."""
    window_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("METHVIN • Select Window")
        self.setMinimumSize(620, 420)
        self.setStyleSheet("background:#0f172a;color:#f8fafc;")
        layout = QVBoxLayout(self)
        title = QLabel("Choose a window to record")
        title.setStyleSheet("font-size:20px;font-weight:800;color:#38bdf8;")
        layout.addWidget(title)
        self.list = QListWidget()
        layout.addWidget(self.list)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.populate)
        use = QPushButton("Use Selected Window")
        use.clicked.connect(self.choose)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.close)
        row.addWidget(refresh); row.addWidget(use); row.addWidget(cancel)
        layout.addLayout(row)
        self.populate()

    def enumerate_windows(self):
        results=[]
        if os.name != "nt":
            return results
        try:
            import ctypes
            from ctypes import wintypes
            user32=ctypes.windll.user32
            EnumWindows=user32.EnumWindows
            IsWindowVisible=user32.IsWindowVisible
            GetWindowTextLengthW=user32.GetWindowTextLengthW
            GetWindowTextW=user32.GetWindowTextW
            GetWindowRect=user32.GetWindowRect
            GetWindowLongW=user32.GetWindowLongW
            GWL_EXSTYLE=-20
            WS_EX_TOOLWINDOW=0x80
            @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            def cb(hwnd, lparam):
                if not IsWindowVisible(hwnd): return True
                if GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW: return True
                n=GetWindowTextLengthW(hwnd)
                if n <= 0: return True
                buf=ctypes.create_unicode_buffer(n+1)
                GetWindowTextW(hwnd, buf, n+1)
                title=buf.value.strip()
                if not title: return True
                rect=wintypes.RECT()
                if GetWindowRect(hwnd, ctypes.byref(rect)):
                    w=rect.right-rect.left; h=rect.bottom-rect.top
                    if w>100 and h>80:
                        results.append({"title":title,"left":rect.left,"top":rect.top,"width":w,"height":h,"hwnd":int(hwnd)})
                return True
            EnumWindows(cb, 0)
        except Exception:
            pass
        return results

    def populate(self):
        self.list.clear()
        self.windows=self.enumerate_windows()
        for w in self.windows:
            item=QListWidgetItem(f"{w['title']}  •  {w['width']}×{w['height']}")
            item.setData(Qt.ItemDataRole.UserRole,w)
            self.list.addItem(item)
        if not self.windows:
            self.list.addItem("Window selection is available on Windows.")

    def choose(self):
        item=self.list.currentItem()
        if item:
            data=item.data(Qt.ItemDataRole.UserRole)
            if data:
                self.window_selected.emit(data)
                self.close()


class AnnotationOverlay(QWidget):
    """On-screen annotation layer captured by MSS during recording."""
    closed=Signal()
    def __init__(self):
        super().__init__(); desktop=QApplication.primaryScreen().availableGeometry()
        for sc in QApplication.screens(): desktop=desktop.united(sc.geometry())
        self.setGeometry(desktop); self.setWindowFlags(Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.Tool); self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground); self.setCursor(Qt.CursorShape.CrossCursor)
        self.points=[]; self.pen_color=QColor(255,70,90); self.pen_width=5; self.drawing=False; self.mode='free'; self.start=QPoint(); self.current=QRect()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); pen=QPen(self.pen_color,self.pen_width,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap,Qt.PenJoinStyle.RoundJoin); p.setPen(pen)
        if self.mode=='free' or self.points:
            for pts in self.points:
                if len(pts)>=2:
                    for a,b in zip(pts,pts[1:]): p.drawLine(a,b)
        if self.drawing and not self.current.isNull():
            if self.mode=='rect': p.drawRect(self.current)
            elif self.mode=='ellipse': p.drawEllipse(self.current)
    def mousePressEvent(self,e):
        if e.button()==Qt.MouseButton.LeftButton:
            self.drawing=True; self.start=e.position().toPoint(); self.current=QRect(self.start,self.start)
            if self.mode=='free': self.points.append([self.start])
            self.update()
        elif e.button()==Qt.MouseButton.RightButton:
            self.points.clear(); self.current=QRect(); self.update()
    def mouseMoveEvent(self,e):
        if self.drawing:
            pos=e.position().toPoint()
            if self.mode=='free' and self.points: self.points[-1].append(pos)
            else: self.current=QRect(self.start,pos).normalized()
            self.update()
    def mouseReleaseEvent(self,e):
        if e.button()==Qt.MouseButton.LeftButton:
            self.drawing=False
            if self.mode in ('rect','ellipse') and not self.current.isNull(): self.points.append([self.current.topLeft(),self.current.topRight(),self.current.bottomRight(),self.current.bottomLeft(),self.current.topLeft()]); self.current=QRect()
            self.update()
    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_Escape: self.close()
        elif e.key()==Qt.Key.Key_F: self.mode='free'
        elif e.key()==Qt.Key.Key_R: self.mode='rect'
        elif e.key()==Qt.Key.Key_C: self.mode='ellipse'
    def closeEvent(self,e): self.closed.emit(); e.accept()




class AIPhotoEditor(QWidget):
    """Lightweight photo editor with background-preserving face cleanup and 4K export."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('METHVIN - AI Photo Studio')
        self.resize(1100, 760)
        self.current_path = None
        self.original = None
        self.preview = None
        self.history = []
        self.history_index = -1
        self.compare_mode = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel('AI Photo Studio')
        title.setStyleSheet('font-size:28px;font-weight:900;color:#38bdf8;')
        top.addWidget(title)
        top.addStretch()
        open_btn = QPushButton('📂 Add Photo'); open_btn.clicked.connect(self.open_photo)
        save_btn = QPushButton('💾 Save Photo'); save_btn.clicked.connect(self.save_photo)
        top.addWidget(open_btn); top.addWidget(save_btn)
        back_btn = QPushButton('← Back')
        back_btn.clicked.connect(self.close)
        top.addWidget(back_btn)
        root.addLayout(top)

        self.image_label = QLabel('Add a photo to begin')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(700, 480)
        self.image_label.setStyleSheet('background:#0c1426;border:2px solid #263556;border-radius:16px;color:#94a3b8;font-size:20px;')
        root.addWidget(self.image_label, 1)

        tools = QHBoxLayout()
        buttons = [
            ('✨ Enhance', self.enhance),
            ('🧹 Face Clean', self.face_clean),
            ('☀ Face Brighten', self.face_brighten),
            ('🔍 4K Upscale', self.upscale_4k),
            ('↶ Undo', self.undo),
            ('↷ Redo', self.redo),
            ('👁 Before', self.toggle_before),
            ('↩ Reset', self.reset_photo),
        ]
        for name, fn in buttons:
            b=QPushButton(name); b.setMinimumHeight(48); b.clicked.connect(fn); tools.addWidget(b)
        root.addLayout(tools)
        note=QLabel('Face tools affect detected face areas only. Background is not replaced or blurred.')
        note.setStyleSheet('color:#94a3b8;font-size:12px;'); note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(note)

    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Photo', '', 'Images (*.png *.jpg *.jpeg *.webp *.bmp)')
        if not path: return
        img=QImage(path)
        if img.isNull():
            QMessageBox.warning(self,'Photo','Could not open this image.'); return
        self.current_path=path
        self.original=img.convertToFormat(QImage.Format.Format_RGBA8888)
        self.preview=self.original.copy()
        self.history=[self.preview.copy()]
        self.history_index=0
        self.compare_mode=False
        self.show_preview()

    def show_preview(self):
        if self.preview is None: return
        shown = self.original if self.compare_mode and self.original is not None else self.preview
        pm=QPixmap.fromImage(shown).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pm)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.preview is not None: self.show_preview()

    def _cv_image(self):
        if self.preview is None: return None
        img=self.preview.convertToFormat(QImage.Format.Format_RGBA8888)
        w,h=img.width(),img.height()
        # PySide6 exposes bits() as a memoryview in newer versions; setsize() is unavailable.
        raw=bytes(img.bits())
        arr=np.frombuffer(raw, dtype=np.uint8).reshape((h,w,4)).copy()
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def _qimage_from_cv(self, bgr):
        rgba=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        h,w=rgba.shape[:2]
        return QImage(rgba.data,w,h,rgba.strides[0],QImage.Format.Format_RGBA8888).copy()

    def _face_boxes(self, img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if cascade.empty(): return []
        return cascade.detectMultiScale(gray,1.1,5,minSize=(80,80))

    def _push_history(self):
        if self.preview is None: return
        if self.history_index < len(self.history)-1:
            self.history=self.history[:self.history_index+1]
        self.history.append(self.preview.copy())
        if len(self.history)>20:
            self.history.pop(0)
        self.history_index=len(self.history)-1
        self.compare_mode=False

    def undo(self):
        if self.preview is None or self.history_index <= 0:
            return
        self.history_index -= 1
        self.preview=self.history[self.history_index].copy()
        self.compare_mode=False
        self.show_preview()

    def redo(self):
        if self.preview is None or self.history_index >= len(self.history)-1:
            return
        self.history_index += 1
        self.preview=self.history[self.history_index].copy()
        self.compare_mode=False
        self.show_preview()

    def toggle_before(self):
        if self.preview is None or self.original is None: return
        self.compare_mode=not self.compare_mode
        self.show_preview()

    def enhance(self):
        img=self._cv_image()
        if img is None: return
        # Gentle detail/contrast enhancement without changing geometry/background.
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab); clahe=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)); l=clahe.apply(l)
        out=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
        self.preview=self._qimage_from_cv(out); self._push_history(); self.show_preview()

    def face_clean(self):
        img=self._cv_image()
        if img is None: return
        boxes=self._face_boxes(img)
        if not boxes:
            QMessageBox.information(self,'Face Clean','No clear face was detected. The photo was left unchanged.'); return
        out=img.copy()
        for x,y,w,h in boxes:
            # Conservative inner-face mask: avoids hair, ears and background.
            mx=int(w*0.14); my=int(h*0.18)
            x1,y1=max(0,x+mx),max(0,y+my); x2,y2=min(out.shape[1],x+w-mx),min(out.shape[0],y+h-int(h*0.08))
            if x2<=x1 or y2<=y1: continue
            roi=out[y1:y2,x1:x2].copy()
            smooth=cv2.bilateralFilter(roi,9,35,35)
            mask=np.zeros(roi.shape[:2],np.uint8)
            cv2.ellipse(mask,(roi.shape[1]//2,roi.shape[0]//2),(max(1,int(roi.shape[1]*.46)),max(1,int(roi.shape[0]*.46))),0,0,360,255,-1)
            alpha=(mask.astype(np.float32)/255.0)[...,None]*0.28
            out[y1:y2,x1:x2]=(roi*(1-alpha)+smooth*alpha).astype(np.uint8)
        self.preview=self._qimage_from_cv(out); self._push_history(); self.show_preview()

    def face_brighten(self):
        img=self._cv_image()
        if img is None: return
        boxes=self._face_boxes(img)
        if not boxes:
            QMessageBox.information(self,'Face Brighten','No clear face was detected. The photo was left unchanged.'); return
        out=img.copy()
        for x,y,w,h in boxes:
            mx=int(w*.16); my=int(h*.18)
            x1,y1=max(0,x+mx),max(0,y+my); x2,y2=min(out.shape[1],x+w-mx),min(out.shape[0],y+h-int(h*.08))
            if x2<=x1 or y2<=y1: continue
            roi=out[y1:y2,x1:x2]
            hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,2]=np.clip(hsv[:,:,2]*1.08+3,0,255)
            bright=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            mask=np.zeros(roi.shape[:2],np.uint8); cv2.ellipse(mask,(roi.shape[1]//2,roi.shape[0]//2),(max(1,int(roi.shape[1]*.46)),max(1,int(roi.shape[0]*.46))),0,0,360,255,-1)
            alpha=(mask.astype(np.float32)/255.0)[...,None]*0.45
            out[y1:y2,x1:x2]=(roi*(1-alpha)+bright*alpha).astype(np.uint8)
        self.preview=self._qimage_from_cv(out); self._push_history(); self.show_preview()

    def upscale_4k(self):
        if self.preview is None: return
        w,h=self.preview.width(),self.preview.height()
        # Fit inside 3840x2160 while preserving the original aspect ratio.
        scale=min(3840/max(1,w),2160/max(1,h))
        if scale<=1.0:
            QMessageBox.information(self,'4K Upscale','This photo is already at or above the 4K target size.'); return
        nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
        self.preview=self.preview.scaled(nw,nh,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        self._push_history()
        self.show_preview()

    def reset_photo(self):
        if self.original is not None:
            self.preview=self.original.copy(); self.history=[self.preview.copy()]; self.history_index=0; self.compare_mode=False; self.show_preview()

    def save_photo(self):
        if self.preview is None:
            QMessageBox.information(self,'Save Photo','Add a photo first.'); return
        path,_=QFileDialog.getSaveFileName(self,'Save Edited Photo','methvin_ai_photo_4k.png','PNG (*.png);;JPEG (*.jpg *.jpeg)')
        if not path: return
        if self.preview.save(path): QMessageBox.information(self,'Save Photo','Photo saved successfully.')
        else: QMessageBox.warning(self,'Save Photo','Could not save the photo.')


class Big123Window(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('METHVIN - 123')
        self.setMinimumSize(620, 360)
        self.setStyleSheet('background:#080d18;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        label = QLabel('123')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            'color:#38bdf8; font-size:190px; font-weight:900; '
            'background:#0c1426; border:2px solid #263556; border-radius:24px;'
        )
        layout.addWidget(label)


class MethvinGUI(QMainWindow):
    """METHVIN Screen Recorder: screen, window, camera, audio, smart tools and scheduling."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1080, 760)
        self.setMinimumSize(900, 650)
        self.setStyleSheet("""
        QMainWindow,QDialog,QWidget{background:#0b1020;color:#f8fafc;font-family:'Segoe UI',Arial;}
        QPushButton{background:#17213a;border:1px solid #2d3b5f;border-radius:9px;padding:10px 16px;color:#f8fafc;font-weight:700;}
        QPushButton:hover{background:#223154;border-color:#4b66a0;}
        QPushButton#record{background:#ef4444;border:0;font-size:18px;padding:14px;}
        QPushButton#record:hover{background:#dc2626;}
        QGroupBox{border:1px solid #263556;border-radius:12px;margin-top:12px;padding:16px;background:#111a2d;font-weight:800;}
        QGroupBox::title{color:#7dd3fc;subcontrol-origin:margin;left:14px;padding:0 7px;}
        QLabel{color:#dbeafe;}
        QComboBox,QSpinBox,QDateTimeEdit,QLineEdit{background:#0c1426;border:1px solid #2d3b5f;border-radius:8px;padding:8px;color:#f8fafc;}
        QCheckBox{spacing:8px;color:#dbeafe;}
        QSlider::groove:horizontal{height:6px;background:#263556;border-radius:3px;}
        QSlider::handle:horizontal{width:16px;margin:-5px 0;border-radius:8px;background:#38bdf8;}
        QTabWidget::pane{border:1px solid #263556;border-radius:12px;background:#10192b;}
        QTabBar::tab{background:#0b1020;color:#8ea0c5;padding:12px 18px;font-weight:800;}
        QTabBar::tab:selected{color:#38bdf8;border-bottom:2px solid #38bdf8;background:#111a2d;}
        QListWidget{background:#0c1426;border:1px solid #263556;border-radius:10px;padding:6px;}
        QStatusBar{background:#060a14;color:#94a3b8;}
        """)
        self.is_recording=False; self.is_paused=False; self.record_area_rect=None; self.recording_duration=0
        self.recorder=None; self.region_selector=None; self.window_picker=None; self.annotation_overlay=None
        self.countdown_timer=None; self.countdown_value=0; self.pending_action=None; self.schedule_timer=None
        self.camera_indices=self.detect_cameras(); self.mic_devices,self.sys_devices=get_audio_input_devices()
        self.settings_file=os.path.join(os.path.expanduser('~'),'.methvin_recorder_settings.json')
        self.default_output_dir=os.path.join(os.path.expanduser('~'),'Videos','METHVIN Recordings')
        self.settings=self.load_settings(); os.makedirs(self.settings.get('output_dir',self.default_output_dir),exist_ok=True)
        self.hotkey_listener=None; self.start_hotkey_listener()
        self.init_ui()
        if not get_ffmpeg_path():
            QMessageBox.warning(self,'FFmpeg Required','FFmpeg was not found. Put ffmpeg.exe beside this script or add FFmpeg to PATH.')

    def detect_cameras(self):
        found=[]
        for i in range(6):
            cap=None
            try:
                cap=cv2.VideoCapture(i,cv2.CAP_DSHOW)
                if cap.isOpened(): found.append(i)
            except Exception: pass
            finally:
                if cap is not None: cap.release()
        return found

    def load_settings(self):
        default={
            'fps':30,'quality':'High','output_dir':self.default_output_dir,'record_mic':True,'record_sys_audio':False,
            'minimize_on_start':False,'mic_volume':100,'system_volume':100,'camera_enabled':False,'camera_position':'Bottom Right',
            'camera_size':'25%','capture_mode':'Cross-screen','resolution':'Native','smart_cleanup':True,'cursor_highlight':False,
            'click_effect':False,'auto_enhance':False,'countdown':3}
        try:
            if os.path.exists(self.settings_file): default.update(json.load(open(self.settings_file,'r')))
        except Exception: pass
        return default

    def save_settings(self):
        self.settings.update({
            'fps':int(self.combo_fps.currentText()),'quality':self.combo_quality.currentText(),'record_mic':self.cb_mic.isChecked(),
            'record_sys_audio':self.cb_sys_audio.isChecked(),'minimize_on_start':self.cb_minimize.isChecked(),'mic_volume':self.slider_mic.value(),
            'system_volume':self.slider_sys.value(),'camera_enabled':self.cb_camera.isChecked(),'camera_position':self.combo_camera_position.currentText(),
            'camera_size':self.combo_camera_size.currentText(),'capture_mode':self.combo_source.currentText(),'resolution':self.combo_resolution.currentText(),
            'smart_cleanup':self.cb_smart_cleanup.isChecked(),'cursor_highlight':self.cb_cursor.isChecked(),'click_effect':self.cb_clicks.isChecked(),
            'auto_enhance':self.cb_enhance.isChecked()})
        try:
            with open(self.settings_file,'w') as f: json.dump(self.settings,f,indent=2)
            self.statusBar().showMessage('Settings saved',3000)
        except Exception as e: QMessageBox.warning(self,'Settings',str(e))

    def init_ui(self):
        root=QWidget(); self.setCentralWidget(root); main=QVBoxLayout(root); main.setContentsMargins(14,14,14,14)
        self.tabs=QTabWidget(); main.addWidget(self.tabs)
        self.tab_home=QWidget(); self.tab_record=QWidget(); self.tab_smart=QWidget(); self.tab_schedule=QWidget(); self.tab_cloud=QWidget(); self.tab_library=QWidget(); self.tab_settings=QWidget(); self.tab_about=QWidget()
        for w,n in [(self.tab_home,'Home'),(self.tab_record,'Record'),(self.tab_smart,'AI / Smart'),(self.tab_schedule,'Schedule'),(self.tab_cloud,'Cloud / Share'),(self.tab_library,'Files'),(self.tab_settings,'Settings'),(self.tab_about,'About')]: self.tabs.addTab(w,n)
        self.setup_home_tab(); self.setup_record_tab(); self.setup_smart_tab(); self.setup_schedule_tab(); self.setup_cloud_tab(); self.setup_library_tab(); self.setup_settings_tab(); self.setup_about_tab()
        self.timer=QTimer(self); self.timer.timeout.connect(self.update_timer_display)
        self.tray_icon=QSystemTrayIcon(self); pm=QPixmap(32,32); pm.fill(QColor('#38bdf8')); self.tray_icon.setIcon(QIcon(pm))
        menu=QMenu(); a=QAction('Restore METHVIN',self); a.triggered.connect(self.showNormal); q=QAction('Exit',self); q.triggered.connect(self.close); menu.addAction(a); menu.addAction(q); self.tray_icon.setContextMenu(menu); self.tray_icon.show()
        self.statusBar().showMessage('Ready')

    def setup_home_tab(self):
        l=QVBoxLayout(self.tab_home); l.setContentsMargins(30,25,30,25); l.setSpacing(18)
        brand=QLabel('METHVIN'); brand.setStyleSheet('font-size:42px;font-weight:900;color:#38bdf8;'); brand.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(brand)
        sub=QLabel('4K / 8K • 120 FPS • Screen • Window • Camera • Audio • Smart Tools'); sub.setStyleSheet('color:#94a3b8;font-size:15px;'); sub.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(sub)
        hero=QGroupBox(); hl=QVBoxLayout(hero)
        title=QLabel('Create. Record. Share.'); title.setStyleSheet('font-size:28px;font-weight:900;color:white;'); hl.addWidget(title)
        desc=QLabel('A modern all-in-one recorder with easy area selection, camera overlay, microphone and system audio controls, annotations, scheduled recording and a clean file library.'); desc.setWordWrap(True); desc.setStyleSheet('font-size:14px;color:#a9b8d3;'); hl.addWidget(desc)
        row=QHBoxLayout(); ai=QPushButton('🖼️  AI PHOTO STUDIO'); ai.clicked.connect(self.open_ai_photo); row.addWidget(ai); r=QPushButton('●  START RECORDING'); r.setObjectName('record'); r.clicked.connect(lambda:self.tabs.setCurrentWidget(self.tab_record)); r.clicked.connect(self.start_recording); row.addWidget(r); s=QPushButton('📸  SCREENSHOT'); s.clicked.connect(self.take_screenshot); row.addWidget(s); f=QPushButton('📁  MY FILES'); f.clicked.connect(self.open_library); row.addWidget(f); hl.addLayout(row); l.addWidget(hero)
        cards=QHBoxLayout()
        for name,text in [('Easy Area','Full screen • Cross-screen • Custom • Window'),('Smooth & Clear','Native • 1080p • 1440p • 4K • 8K • up to 120 FPS'),('Multi-track','Screen • Camera • Mic • System audio'),('Smart Tools','Noise cleanup • Enhance • Cursor effects • Annotations'),('Share','Package recordings for cloud upload or sharing')]:
            g=QGroupBox(name); q=QVBoxLayout(g); x=QLabel(text); x.setWordWrap(True); x.setStyleSheet('color:#9fb0ce;'); q.addWidget(x); cards.addWidget(g)
        l.addLayout(cards); l.addStretch()

    def open_ai_photo(self):
        self.ai_photo_window = AIPhotoEditor(self)
        self.ai_photo_window.show()
        self.ai_photo_window.raise_()
        self.ai_photo_window.activateWindow()

    def setup_record_tab(self):
        l=QVBoxLayout(self.tab_record); l.setSpacing(12)
        top=QHBoxLayout(); self.timer_label=QLabel('00:00:00'); self.timer_label.setStyleSheet('font-size:34px;font-weight:900;color:#10b981;'); top.addWidget(self.timer_label); top.addStretch(); self.lbl_mode=QLabel('Ready'); self.lbl_mode.setStyleSheet('color:#38bdf8;font-weight:800;'); top.addWidget(self.lbl_mode); l.addLayout(top)
        controls=QHBoxLayout(); self.btn_record=QPushButton('●  Start Recording'); self.btn_record.setObjectName('record'); self.btn_record.clicked.connect(self.start_recording); self.btn_pause=QPushButton('Pause'); self.btn_pause.clicked.connect(self.toggle_pause); self.btn_pause.hide(); self.btn_stop=QPushButton('Stop'); self.btn_stop.clicked.connect(self.stop_recording); self.btn_stop.hide(); self.btn_annotation=QPushButton('✎ Annotate (F8)'); self.btn_annotation.clicked.connect(self.toggle_annotations); controls.addWidget(self.btn_record); controls.addWidget(self.btn_pause); controls.addWidget(self.btn_stop); controls.addWidget(self.btn_annotation); l.addLayout(controls)
        area=QGroupBox('Easy Screen Area Selection'); al=QVBoxLayout(area)
        ar=QHBoxLayout(); self.combo_source=QComboBox(); self.combo_source.addItems(['Cross-screen','Full Screen / Primary','Custom Area','Window']); self.combo_source.setCurrentText(self.settings.get('capture_mode','Cross-screen')); self.combo_source.currentTextChanged.connect(self.on_source_changed); ar.addWidget(self.combo_source,1); self.btn_choose_area=QPushButton('Select Area'); self.btn_choose_area.clicked.connect(self.choose_area); ar.addWidget(self.btn_choose_area); self.btn_choose_window=QPushButton('Select Window'); self.btn_choose_window.clicked.connect(self.choose_window); ar.addWidget(self.btn_choose_window); al.addLayout(ar)
        self.lbl_area=QLabel('Capture: All connected displays'); self.lbl_area.setStyleSheet('color:#94a3b8;'); al.addWidget(self.lbl_area); l.addWidget(area)
        grid=QHBoxLayout()
        audio=QGroupBox('Audio'); a=QVBoxLayout(audio)
        mr=QHBoxLayout(); self.cb_mic=QCheckBox('Microphone'); self.cb_mic.setChecked(self.settings.get('record_mic',True)); self.combo_mic=QComboBox(); [self.combo_mic.addItem(n,i) for i,n in self.mic_devices]; mr.addWidget(self.cb_mic); mr.addWidget(self.combo_mic,1); a.addLayout(mr)
        vr=QHBoxLayout(); vr.addWidget(QLabel('Mic Volume')); self.slider_mic=QSlider(Qt.Orientation.Horizontal); self.slider_mic.setRange(0,200); self.slider_mic.setValue(int(self.settings.get('mic_volume',100))); self.lbl_mic=QLabel(f'{self.slider_mic.value()}%'); self.slider_mic.valueChanged.connect(lambda v:self.lbl_mic.setText(f'{v}%')); vr.addWidget(self.slider_mic,1); vr.addWidget(self.lbl_mic); a.addLayout(vr)
        sr=QHBoxLayout(); self.cb_sys_audio=QCheckBox('System Audio'); self.cb_sys_audio.setChecked(self.settings.get('record_sys_audio',False)); self.combo_sys=QComboBox(); [self.combo_sys.addItem(n,i) for i,n in self.sys_devices]; sr.addWidget(self.cb_sys_audio); sr.addWidget(self.combo_sys,1); a.addLayout(sr)
        vs=QHBoxLayout(); vs.addWidget(QLabel('System Volume')); self.slider_sys=QSlider(Qt.Orientation.Horizontal); self.slider_sys.setRange(0,200); self.slider_sys.setValue(int(self.settings.get('system_volume',100))); self.lbl_sys=QLabel(f'{self.slider_sys.value()}%'); self.slider_sys.valueChanged.connect(lambda v:self.lbl_sys.setText(f'{v}%')); vs.addWidget(self.slider_sys,1); vs.addWidget(self.lbl_sys); a.addLayout(vs); grid.addWidget(audio)
        cam=QGroupBox('Camera'); c=QVBoxLayout(cam); cr=QHBoxLayout(); self.cb_camera=QCheckBox('Camera Overlay'); self.cb_camera.setChecked(self.settings.get('camera_enabled',False)); self.combo_camera=QComboBox(); [self.combo_camera.addItem(f'Camera {i}',i) for i in self.camera_indices]; cr.addWidget(self.cb_camera); cr.addWidget(self.combo_camera,1); c.addLayout(cr); co=QHBoxLayout(); self.combo_camera_position=QComboBox(); self.combo_camera_position.addItems(['Bottom Right','Bottom Left','Top Right','Top Left']); self.combo_camera_position.setCurrentText(self.settings.get('camera_position','Bottom Right')); co.addWidget(QLabel('Position')); co.addWidget(self.combo_camera_position,1); self.combo_camera_size=QComboBox(); self.combo_camera_size.addItems(['20%','25%','30%','35%']); self.combo_camera_size.setCurrentText(self.settings.get('camera_size','25%')); co.addWidget(QLabel('Size')); co.addWidget(self.combo_camera_size); c.addLayout(co); grid.addWidget(cam); l.addLayout(grid)
        util=QHBoxLayout(); shot=QPushButton('📸 Screenshot • 3-2-1'); shot.clicked.connect(self.take_screenshot); util.addWidget(shot); util.addWidget(QLabel('Tip: F9 start/stop • F8 annotations')); l.addLayout(util); l.addStretch()

    def setup_smart_tab(self):
        l=QVBoxLayout(self.tab_smart); l.setSpacing(12)
        title=QLabel('AI / Smart Recording Features'); title.setStyleSheet('font-size:24px;font-weight:900;color:#38bdf8;'); l.addWidget(title)
        note=QLabel('Offline smart processing is used here so the app does not require an online AI service.'); note.setStyleSheet('color:#94a3b8;'); note.setWordWrap(True); l.addWidget(note)
        g=QGroupBox('Smart Audio & Video'); q=QVBoxLayout(g)
        self.cb_smart_cleanup=QCheckBox('Smart Mic Cleanup — reduce steady background noise with FFmpeg audio filtering'); self.cb_smart_cleanup.setChecked(self.settings.get('smart_cleanup',True)); q.addWidget(self.cb_smart_cleanup)
        self.cb_enhance=QCheckBox('Auto Enhance — gentle clarity/contrast processing during final export'); self.cb_enhance.setChecked(self.settings.get('auto_enhance',False)); q.addWidget(self.cb_enhance)
        self.cb_cursor=QCheckBox('Cursor Highlight — show a visible pointer halo in the recording'); self.cb_cursor.setChecked(self.settings.get('cursor_highlight',False)); q.addWidget(self.cb_cursor)
        self.cb_clicks=QCheckBox('Click Effect — show a small visual click indicator'); self.cb_clicks.setChecked(self.settings.get('click_effect',False)); q.addWidget(self.cb_clicks)
        l.addWidget(g)
        big123=QPushButton('🔢  Open Large 123 Window')
        big123.setMinimumHeight(52)
        big123.clicked.connect(self.open_big_123)
        l.addWidget(big123)
        ann=QGroupBox('On-screen Annotations'); a=QVBoxLayout(ann); a.addWidget(QLabel('Press F8 or the Annotate button while recording. Left-drag to draw; right-click to clear.'))
        for t in ('Free Drawing','Shapes / highlight layer','Whiteboard-style overlay','Cursor emphasis'): a.addWidget(QLabel('✓ '+t))
        l.addWidget(ann); l.addStretch()

    def open_big_123(self):
        self.big123_window = Big123Window(self)
        self.big123_window.show()
        self.big123_window.raise_()
        self.big123_window.activateWindow()

    def setup_schedule_tab(self):
        l=QVBoxLayout(self.tab_schedule); title=QLabel('Scheduled Recording'); title.setStyleSheet('font-size:24px;font-weight:900;color:#38bdf8;'); l.addWidget(title)
        g=QGroupBox('Plan a recording'); q=QVBoxLayout(g); q.addWidget(QLabel('Choose a future date and time. METHVIN will start the normal 3-second countdown automatically.'))
        self.schedule_datetime=QDateTimeEdit(); self.schedule_datetime.setCalendarPopup(True); self.schedule_datetime.setDateTime(QDateTime.currentDateTime().addSecs(60)); q.addWidget(self.schedule_datetime)
        row=QHBoxLayout(); b=QPushButton('Schedule Recording'); b.clicked.connect(self.schedule_recording); cancel=QPushButton('Cancel Schedule'); cancel.clicked.connect(self.cancel_schedule); row.addWidget(b); row.addWidget(cancel); q.addLayout(row)
        self.schedule_status=QLabel('No recording scheduled.'); self.schedule_status.setStyleSheet('color:#94a3b8;'); q.addWidget(self.schedule_status); l.addWidget(g); l.addStretch()

    def setup_cloud_tab(self):
        l=QVBoxLayout(self.tab_cloud); l.setSpacing(14)
        t=QLabel('Cloud Storage & Secure Sharing'); t.setStyleSheet('font-size:24px;font-weight:900;color:#38bdf8;'); l.addWidget(t)
        d=QLabel('METHVIN can prepare a clean share package from a recording. Direct cloud upload and encrypted share links require a cloud provider account/API, so this build keeps the local package step provider-neutral.'); d.setWordWrap(True); d.setStyleSheet('color:#94a3b8;'); l.addWidget(d)
        g=QGroupBox('Share Tools'); q=QVBoxLayout(g)
        b=QPushButton('Create Share Package (.zip)'); b.clicked.connect(self.create_share_package); q.addWidget(b)
        o=QPushButton('Open Recordings Folder'); o.clicked.connect(self.open_output_folder); q.addWidget(o)
        l.addWidget(g); l.addStretch()

    def create_share_package(self):
        item=self.file_list.currentItem() if hasattr(self,'file_list') else None
        path=item.data(Qt.ItemDataRole.UserRole) if item else None
        if not path or not os.path.exists(path):
            QMessageBox.information(self,'Share Package','Open Files and select a recording first.')
            return
        dest=QFileDialog.getSaveFileName(self,'Save Share Package',os.path.splitext(path)[0]+'_share.zip','ZIP (*.zip)')[0]
        if not dest:return
        try:
            with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
                z.write(path,os.path.basename(path))
                manifest={'created':datetime.datetime.now().isoformat(),'file':os.path.basename(path),'app':APP_NAME}
                z.writestr('share_manifest.json',json.dumps(manifest,indent=2))
            self.statusBar().showMessage('Share package created: '+dest,5000)
        except Exception as e: QMessageBox.warning(self,'Share Package',str(e))

    def setup_library_tab(self):
        l=QVBoxLayout(self.tab_library); t=QLabel('METHVIN Files'); t.setStyleSheet('font-size:24px;font-weight:900;color:#38bdf8;'); l.addWidget(t)
        self.file_list=QListWidget(); self.file_list.itemDoubleClicked.connect(self.open_selected_file); l.addWidget(self.file_list)
        row=QHBoxLayout(); b=QPushButton('Refresh'); b.clicked.connect(self.refresh_library); o=QPushButton('Open Folder'); o.clicked.connect(self.open_output_folder); row.addWidget(b); row.addWidget(o); l.addLayout(row); self.refresh_library()

    def setup_settings_tab(self):
        l=QVBoxLayout(self.tab_settings); g=QGroupBox('Smooth & Clear Recording'); q=QVBoxLayout(g)
        fr=QHBoxLayout(); fr.addWidget(QLabel('FPS')); self.combo_fps=QComboBox(); self.combo_fps.addItems(['15','30','60','90','120']); self.combo_fps.setCurrentText(str(self.settings.get('fps',30))); fr.addWidget(self.combo_fps); q.addLayout(fr)
        rr=QHBoxLayout(); rr.addWidget(QLabel('Resolution')); self.combo_resolution=QComboBox(); self.combo_resolution.addItems(['Native','1920x1080','2560x1440','3840x2160 (4K)','7680x4320 (8K)']); self.combo_resolution.setCurrentText(self.settings.get('resolution','Native')); rr.addWidget(self.combo_resolution); q.addLayout(rr)
        qr=QHBoxLayout(); qr.addWidget(QLabel('Quality')); self.combo_quality=QComboBox(); self.combo_quality.addItems(['High','Medium','Low']); self.combo_quality.setCurrentText(self.settings.get('quality','High')); qr.addWidget(self.combo_quality); q.addLayout(qr); l.addWidget(g)
        out=QGroupBox('Save Location'); o=QHBoxLayout(out); self.lbl_out=QLabel(self.settings.get('output_dir',self.default_output_dir)); self.lbl_out.setWordWrap(True); b=QPushButton('Browse'); b.clicked.connect(self.browse_output_dir); o.addWidget(self.lbl_out,1); o.addWidget(b); l.addWidget(out)
        pref=QGroupBox('Preferences'); p=QVBoxLayout(pref); self.cb_minimize=QCheckBox('Minimize to tray when recording starts'); self.cb_minimize.setChecked(self.settings.get('minimize_on_start',False)); p.addWidget(self.cb_minimize); p.addWidget(QLabel('Global hotkeys: F9 Start/Stop • F8 Annotation layer')); l.addWidget(pref)
        save=QPushButton('Save All Settings'); save.clicked.connect(self.save_settings); l.addWidget(save); l.addStretch()

    def setup_about_tab(self):
        l=QVBoxLayout(self.tab_about); t=QLabel('METHVIN Screen Recorder'); t.setStyleSheet('font-size:26px;font-weight:900;color:#38bdf8;'); l.addWidget(t)
        d=QLabel('Enhanced build with area/window selection, up to 120 FPS, 4K/8K export options, camera overlay, microphone/system audio controls, screenshots, annotations, smart processing, scheduling and file management.'); d.setWordWrap(True); l.addWidget(d); l.addStretch()

    def on_source_changed(self,mode):
        self.lbl_area.setText({'Cross-screen':'Capture: all connected displays','Full Screen / Primary':'Capture: primary display','Custom Area':'Capture: custom selected rectangle','Window':'Capture: selected application window'}.get(mode,'Capture selected source'))
        self.btn_choose_area.setEnabled(mode=='Custom Area'); self.btn_choose_window.setEnabled(mode=='Window')

    def choose_area(self):
        self.hide(); self.region_selector=RegionSelector(); self.region_selector.region_selected.connect(self.on_region_selected); self.region_selector.show()

    @Slot(QRect)
    def on_region_selected(self,rect):
        self.record_area_rect={'top':rect.top(),'left':rect.left(),'width':rect.width(),'height':rect.height()}; self.lbl_area.setText(f'Custom: {rect.width()}×{rect.height()} at {rect.left()},{rect.top()}'); self.showNormal(); self.activateWindow()

    def choose_window(self):
        self.window_picker=WindowPicker(self); self.window_picker.window_selected.connect(self.on_window_selected); self.window_picker.show(); self.window_picker.raise_(); self.window_picker.activateWindow()

    def on_window_selected(self,data):
        self.record_area_rect={k:data[k] for k in ('top','left','width','height')}; self.lbl_area.setText(f"Window: {data['title'][:70]} • {data['width']}×{data['height']}")

    def choose_monitor_rect(self):
        with mss.mss() as sct:
            mode=self.combo_source.currentText()
            if mode=='Cross-screen': return dict(sct.monitors[0])
            if mode=='Full Screen / Primary': return dict(sct.monitors[1])
            if self.record_area_rect: return dict(self.record_area_rect)
            return dict(sct.monitors[0])

    def show_countdown(self,callback):
        if self.countdown_timer: self.countdown_timer.stop()
        self.countdown_value=3; self.pending_action=callback; self.timer_label.setText('3'); self.timer_label.setStyleSheet('font-size:58px;font-weight:900;color:#38bdf8;')
        self.countdown_timer=QTimer(self); self.countdown_timer.setInterval(1000)
        def tick():
            self.countdown_value-=1
            if self.countdown_value>0: self.timer_label.setText(str(self.countdown_value))
            else:
                self.countdown_timer.stop(); self.countdown_timer=None; cb=self.pending_action; self.pending_action=None; self.timer_label.setText('GO!'); QTimer.singleShot(250,cb)
        self.countdown_timer.timeout.connect(tick); self.countdown_timer.start()

    def start_recording(self):
        if self.is_recording or self.pending_action: return
        self.pending_action=self.start_recording_now; self.show_countdown(self.start_recording_now)

    def start_recording_now(self):
        self.pending_action=None
        if not get_ffmpeg_path(): QMessageBox.critical(self,'FFmpeg','FFmpeg is required.'); return
        out=self.settings.get('output_dir',self.default_output_dir); os.makedirs(out,exist_ok=True)
        filename=os.path.join(out,'Recording_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.mp4')
        monitor=self.choose_monitor_rect(); fps=int(self.combo_fps.currentText()); quality=self.combo_quality.currentText()
        mic=self.cb_mic.isChecked(); sysa=self.cb_sys_audio.isChecked(); mic_idx=self.combo_mic.currentData() if mic and self.combo_mic.count() else None; sys_idx=self.combo_sys.currentData() if sysa and self.combo_sys.count() else None
        self.recorder=RecorderManager(filename,monitor,fps,quality,mic,mic_idx,sysa,sys_idx,mic_volume=self.slider_mic.value()/100.0,camera_enabled=self.cb_camera.isChecked(),camera_index=(self.combo_camera.currentData() or 0),camera_scale=int(self.combo_camera_size.currentText()[:-1])/100.0,camera_position=self.combo_camera_position.currentText(), output_resolution=self.combo_resolution.currentText(), smart_cleanup=self.cb_smart_cleanup.isChecked(), auto_enhance=self.cb_enhance.isChecked(), system_volume=self.slider_sys.value()/100.0, cursor_highlight=(self.cb_cursor.isChecked() or self.cb_clicks.isChecked()))
        self.recorder.recording_finished.connect(self.on_recording_finished); self.recorder.error_occurred.connect(self.on_recorder_error); self.recorder.status_update.connect(lambda s:self.statusBar().showMessage(s)); self.is_recording=True; self.btn_record.hide(); self.btn_pause.show(); self.btn_stop.show(); self.lbl_mode.setText('● RECORDING'); self.recording_duration=0; self.timer_label.setText('00:00:00'); self.timer.start(1000); self.recorder.start()
        if self.cb_minimize.isChecked(): self.hide()

    def toggle_pause(self):
        if not self.is_recording:return
        if self.is_paused: self.recorder.resume(); self.is_paused=False; self.btn_pause.setText('Pause'); self.lbl_mode.setText('● RECORDING');
        else: self.recorder.pause(); self.is_paused=True; self.btn_pause.setText('Resume'); self.lbl_mode.setText('Ⅱ PAUSED')

    def stop_recording(self):
        if self.is_recording and self.recorder: self.recorder.stop(); self.btn_stop.setEnabled(False); self.statusBar().showMessage('Finalizing recording...')

    def on_recorder_error(self,msg): self.reset_ui_state(); QMessageBox.critical(self,'Recording Error',msg)
    def on_recording_finished(self,path): self.reset_ui_state(); self.refresh_library(); self.statusBar().showMessage(f'Saved: {path}',5000)

    def reset_ui_state(self):
        self.is_recording=False; self.is_paused=False; self.recorder=None; self.btn_record.show(); self.btn_pause.hide(); self.btn_stop.hide(); self.btn_stop.setEnabled(True); self.lbl_mode.setText('Ready'); self.timer.stop()
        if self.annotation_overlay: self.annotation_overlay.close(); self.annotation_overlay=None

    def take_screenshot(self): self.show_countdown(self.capture_screenshot_now)
    def capture_screenshot_now(self):
        try:
            out=self.settings.get('output_dir',self.default_output_dir); os.makedirs(out,exist_ok=True); fn=os.path.join(out,'Screenshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')
            with mss.mss() as sct: m=sct.monitors[0] if self.combo_source.currentText()=='Cross-screen' else (self.record_area_rect or sct.monitors[1]); img=sct.grab(m); mss.tools.to_png(img.rgb,img.size,output=fn)
            self.refresh_library(); self.statusBar().showMessage(f'Screenshot saved: {fn}',4000)
        except Exception as e: QMessageBox.warning(self,'Screenshot Error',str(e))

    def toggle_annotations(self):
        if self.annotation_overlay:
            self.annotation_overlay.close(); self.annotation_overlay=None; return
        self.annotation_overlay=AnnotationOverlay(); self.annotation_overlay.show(); self.annotation_overlay.raise_(); self.annotation_overlay.activateWindow()

    def schedule_recording(self):
        target=self.schedule_datetime.dateTime(); now=QDateTime.currentDateTime(); secs=now.secsTo(target)
        if secs<=0: QMessageBox.warning(self,'Schedule','Please choose a future time.'); return
        self.cancel_schedule(); self.schedule_timer=QTimer(self); self.schedule_timer.setSingleShot(True); self.schedule_timer.timeout.connect(self.start_recording); self.schedule_timer.start(max(1,secs*1000)); self.schedule_status.setText('Scheduled for '+target.toString('yyyy-MM-dd HH:mm:ss')); self.statusBar().showMessage('Recording scheduled')

    def cancel_schedule(self):
        if self.schedule_timer: self.schedule_timer.stop(); self.schedule_timer=None
        if hasattr(self,'schedule_status'): self.schedule_status.setText('No recording scheduled.')

    def refresh_library(self):
        if not hasattr(self,'file_list'): return
        self.file_list.clear(); out=self.settings.get('output_dir',self.default_output_dir); os.makedirs(out,exist_ok=True)
        files=sorted([f for f in os.listdir(out) if f.lower().endswith(('.mp4','.png','.jpg','.jpeg','.wav'))],key=lambda x:os.path.getmtime(os.path.join(out,x)),reverse=True)
        for name in files[:200]:
            item=QListWidgetItem(name); item.setData(Qt.ItemDataRole.UserRole,os.path.join(out,name)); self.file_list.addItem(item)
        if not files:self.file_list.addItem('No recordings or screenshots yet.')

    def open_selected_file(self,item):
        p=item.data(Qt.ItemDataRole.UserRole)
        if p and os.path.exists(p):
            try: os.startfile(p)
            except Exception as e: QMessageBox.warning(self,'Open File',str(e))
    def open_output_folder(self):
        p=self.settings.get('output_dir',self.default_output_dir); os.makedirs(p,exist_ok=True); os.startfile(p)
    def open_library(self): self.refresh_library(); self.tabs.setCurrentWidget(self.tab_library)
    def browse_output_dir(self):
        d=QFileDialog.getExistingDirectory(self,'Choose save folder',self.settings.get('output_dir',self.default_output_dir))
        if d: self.settings['output_dir']=d; self.lbl_out.setText(d)

    def start_hotkey_listener(self):
        def on_press(key):
            try:
                if key==keyboard.Key.f9: QTimer.singleShot(0,self.handle_hotkey)
                elif key==keyboard.Key.f8: QTimer.singleShot(0,self.toggle_annotations)
            except Exception: pass
        self.hotkey_listener=keyboard.Listener(on_press=on_press); self.hotkey_listener.start()
    def handle_hotkey(self): self.stop_recording() if self.is_recording else self.start_recording()
    def update_timer_display(self):
        if self.is_recording and not self.is_paused:
            self.recording_duration+=1; h=self.recording_duration//3600; m=(self.recording_duration%3600)//60; s=self.recording_duration%60; self.timer_label.setText(f'{h:02d}:{m:02d}:{s:02d}')

    def closeEvent(self,e):
        if self.is_recording:
            r=QMessageBox.question(self,'Exit','Stop and save the current recording?',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if r==QMessageBox.StandardButton.Yes:
                self.stop_recording();
                if self.recorder:self.recorder.wait(5000)
            else:e.ignore(); return
        if self.hotkey_listener:self.hotkey_listener.stop()
        if self.annotation_overlay:self.annotation_overlay.close()
        e.accept()

if __name__=='__main__':
    app=QApplication(sys.argv); app.setQuitOnLastWindowClosed(False); w=MethvinGUI(); w.show(); sys.exit(app.exec())
