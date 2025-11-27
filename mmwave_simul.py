import cv2
import numpy as np
import random
import math
from ultralytics import YOLO
import threading
import queue
import time
import traceback
import torch
import ultralytics
import subprocess
import sys
import os
import datetime

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        
        # Check if file exists at base_path
        path = os.path.join(base_path, relative_path)
        if os.path.exists(path):
            return path
            
        # If not, check if it is in _internal (common in newer PyInstaller)
        internal_path = os.path.join(os.path.dirname(sys.executable), '_internal', relative_path)
        if os.path.exists(internal_path):
            return internal_path
            
        # Also check root of executable
        root_path = os.path.join(os.path.dirname(sys.executable), relative_path)
        if os.path.exists(root_path):
            return root_path
            
    return relative_path

class VoiceAlert:
    def __init__(self, cooldown=30.0):
        self.last_alert_time = 0
        self.cooldown = cooldown
        self._lock = threading.Lock()

    def speak(self, text):
        with self._lock:
            now = time.time()
            if now - self.last_alert_time < self.cooldown:
                return
            self.last_alert_time = now
        
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        try:
            # PowerShell command to speak text
            cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}');"
            
            # Set creation flag to hide the console window on Windows
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW
                
            subprocess.run(["powershell", "-Command", cmd], check=False, creationflags=creationflags)
        except Exception:
            traceback.print_exc()

# Fix for PyTorch 2.6+ compatibility with Ultralytics
def register_ultralytics_safe_globals():
    """Add Ultralytics task classes to torch safe globals.

    Returns list of added globals (for debugging); best-effort and non-fatal.
    """
    added = []
    try:
        torch.serialization.add_safe_globals([])
        import ultralytics.nn.tasks as tasks
        for name in dir(tasks):
            obj = getattr(tasks, name)
            if isinstance(obj, type):
                try:
                    torch.serialization.add_safe_globals([obj])
                    added.append(f"{tasks.__name__}.{name}")
                except Exception:
                    # continue on individual failures
                    pass
    except Exception:
        traceback.print_exc()
    return added

# register safe globals eagerly
_ADDED_SAFE_GLOBALS = register_ultralytics_safe_globals()

# Load models with robust fallback
def safe_load_yolo(path, task=None):
    """Try to load a YOLO model with safe-globals registered first.

    Falls back to using a safe_globals context manager if available, and as a
    last resort (only for trusted checkpoints) will retry by allowing full
    pickle loading (weights_only=False). Returns the loaded model or raises.
    """
    try:
        return YOLO(path) if task is None else YOLO(path, task=task)
    except Exception as e:
        # First fallback: try loading inside a safe_globals context that
        # explicitly allowlists Ultralytics task classes.
        try:
            import ultralytics.nn.tasks as tasks
            globals_list = [getattr(tasks, n) for n in dir(tasks) if isinstance(getattr(tasks, n), type)]
            try:
                with torch.serialization.safe_globals(globals_list):
                    return YOLO(path) if task is None else YOLO(path, task=task)
            except Exception:
                pass
        except Exception:
            pass

        # Last resort: load with pickles enabled. Only use for trusted files.
        try:
            print(f"Warning: falling back to torch.load(..., weights_only=False) for {path}. Ensure checkpoint is trusted.")
            _ = torch.load(path, map_location='cpu', weights_only=False)
            return YOLO(path) if task is None else YOLO(path, task=task)
        except Exception:
            traceback.print_exc()
            raise e

def synthesize_mmwave_for_roi(mm_img, bbox, keypoints=None, probability=0.4):
    """Draw a simulated hidden-object silhouette inside bbox on mm_img.
       Returns flag whether an object was embedded."""
    x1, y1, x2, y2 = bbox
    h = max(4, y2 - y1)
    w = max(4, x2 - x1)

    # Decide randomly whether this person has a hidden object in simulation
    if random.random() > probability:
        return False

    # Place an ellipse roughly in the torso area (lower-middle of bbox)
    cx = int(x1 + 0.5 * w + random.uniform(-0.1, 0.1) * w)
    cy = int(y1 + 0.5 * h + random.uniform(-0.05, 0.05) * h)

    # Avoid face if keypoints are available
    if keypoints is not None and len(keypoints) > 0:
        nose = keypoints[0]
        # Check confidence (index 2)
        if float(nose[2]) > 0.5:
            nose_x, nose_y = float(nose[0]), float(nose[1])
            # Distance to generated point
            dist = math.sqrt((cx - nose_x)**2 + (cy - nose_y)**2)
            # If within ~15% of height from nose, move it down to torso/waist
            if dist < (h * 0.15):
                cy = int(y1 + 0.65 * h + random.uniform(-0.05, 0.05) * h)

    axis_major = int(w * random.uniform(0.08, 0.25))
    axis_minor = int(h * random.uniform(0.04, 0.12))
    angle = random.uniform(-30, 30)

    # Draw filled white shape on mm_img
    cv2.ellipse(mm_img, (cx, cy), (axis_major, axis_minor), angle, 0, 360, 255, -1)

    # Add blur to simulate spread of mmWave return and low resolution
    ksize = max(1, int((axis_major + axis_minor) / 6) | 1)  # odd
    mm_img[:] = cv2.GaussianBlur(mm_img, (ksize, ksize), 0)

    # Add speckle noise
    noise = (np.random.randn(*mm_img.shape) * 8).astype(np.int16)
    mm_img[:] = np.clip(mm_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return True

def detect_synthetic_objects(mm_img, min_area=200):
    """Simple contour-based detector on synthetic mmwave image."""
    _, th = cv2.threshold(mm_img, 50, 255, cv2.THRESH_BINARY)
    # morphological open to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(c)
            detections.append((x,y,x+w,y+h, area))
    return detections, th

def generate_human_point_cloud(num_points=500, scale_w=1.0, scale_h=1.0, offset_x=0, offset_y=0, offset_z=0):
    """Generates a 3D point cloud resembling a human shape."""
    points = []
    # Torso (ellipsoid-ish)
    for _ in range(int(num_points * 0.6)):
        x = random.gauss(0, 0.15) * scale_w + offset_x
        y = random.uniform(-0.5, 0.5) * scale_h + offset_y
        z = random.gauss(0, 0.15) * scale_w + offset_z
        points.append([x, y, z])
    
    # Head
    for _ in range(int(num_points * 0.15)):
        x = random.gauss(0, 0.1) * scale_w + offset_x
        y = random.uniform(0.5, 0.7) * scale_h + offset_y
        z = random.gauss(0, 0.1) * scale_w + offset_z
        points.append([x, y, z])
        
    # Limbs (random scatter around)
    for _ in range(int(num_points * 0.25)):
        x = random.uniform(-0.4, 0.4) * scale_w + offset_x
        y = random.uniform(-0.5, 0.5) * scale_h + offset_y
        z = random.uniform(-0.2, 0.2) * scale_w + offset_z
        points.append([x, y, z])
        
    return np.array(points)

def render_point_cloud(points, img_w, img_h, title="Point Cloud", noise_level=0.0, point_size=2):
    """Renders 3D points onto a 2D image with a grid and rotation."""
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Camera setup
    cam_pitch = 0.1
    cam_yaw = 0.0 # Fixed view to match camera perspective
    cam_dist = 0.0 # Points are already in camera coordinates (Z is depth)
    f = 300
    cx, cy = img_w // 2, img_h // 2
    
    c_y, s_y = math.cos(cam_yaw), math.sin(cam_yaw)
    c_p, s_p = math.cos(cam_pitch), math.sin(cam_pitch)
    
    # Draw Grid (Floor at y=-1.0)
    grid_size = 5.0
    steps = 10
    for i in range(-steps, steps+1):
        u = i * grid_size / steps
        # Z-lines (from z=0 to z=10)
        p1 = (u, -1.0, 0.0); p2 = (u, -1.0, 10.0)
        # X-lines
        p3 = (-grid_size, -1.0, u + 5.0); p4 = (grid_size, -1.0, u + 5.0) # Shift grid forward
        
        for start, end in [(p1, p2)]:
            # Project start
            x, y, z = start
            x1 = x * c_y - z * s_y; z1 = x * s_y + z * c_y
            y2 = y * c_p - z1 * s_p; z2 = y * s_p + z1 * c_p
            z2 += cam_dist
            if z2 <= 0.1: continue
            sx1 = int(cx + (x1 * f) / z2); sy1 = int(cy - (y2 * f) / z2)
            
            # Project end
            x, y, z = end
            x1 = x * c_y - z * s_y; z1 = x * s_y + z * c_y
            y2 = y * c_p - z1 * s_p; z2 = y * s_p + z1 * c_p
            z2 += cam_dist
            if z2 <= 0.1: continue
            sx2 = int(cx + (x1 * f) / z2); sy2 = int(cy - (y2 * f) / z2)
            
            cv2.line(img, (sx1, sy1), (sx2, sy2), (50, 50, 50), 1)

    # Draw Points
    if len(points) > 0:
        # Add noise copy
        pts_noisy = points + np.random.normal(0, noise_level, points.shape)
        
        for p in pts_noisy:
            x, y, z = p
            
            # Rotate
            x1 = x * c_y - z * s_y
            z1 = x * s_y + z * c_y
            y2 = y * c_p - z1 * s_p
            z2 = y * s_p + z1 * c_p
            z2 += cam_dist
            
            if z2 <= 0.1: continue
            
            sx = int(cx + (x1 * f) / z2)
            sy = int(cy - (y2 * f) / z2)
            
            if 0 <= sx < img_w and 0 <= sy < img_h:
                # Color by height (original y)
                # Map y from [-1.5, 0.5] to [0, 1] for color
                norm_y = (y + 1.5) / 2.0 
                norm_y = max(0, min(1, norm_y))
                # Jet colormap approximation
                r = int(255 * norm_y)
                g = int(255 * (1 - abs(norm_y - 0.5)*2))
                b = int(255 * (1 - norm_y))
                
                cv2.circle(img, (sx, sy), point_size, (b, g, r), -1)

    # Add label
    cv2.rectangle(img, (0, 0), (img_w, 40), (0,0,0), -1)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

class VitalSignSimulator:
    def __init__(self):
        self.people_vitals = {}  # Map track_id or index to vital state

    def get_vitals(self, person_id):
        """Returns current heart rate, breath rate, and a waveform point."""
        now = time.time()
        
        if person_id not in self.people_vitals:
            # Initialize random vitals for new person
            self.people_vitals[person_id] = {
                'base_hr': random.uniform(60, 90),
                'base_br': random.uniform(12, 20),
                'phase_hr': random.uniform(0, 2*math.pi),
                'phase_br': random.uniform(0, 2*math.pi),
                'history': []
            }
        
        v = self.people_vitals[person_id]
        
        # Simulate slight fluctuations
        current_hr = v['base_hr'] + math.sin(now * 0.5) * 5
        current_br = v['base_br'] + math.sin(now * 0.2) * 2
        
        # Generate waveform point (Composite of Heart + Breath)
        # Breathing is slow/high amplitude, Heart is fast/low amplitude
        t = now
        breath_signal = math.sin(2 * math.pi * (current_br/60) * t + v['phase_br']) * 1.0
        heart_signal = math.sin(2 * math.pi * (current_hr/60) * t + v['phase_hr']) * 0.2
        noise = random.uniform(-0.05, 0.05)
        
        signal_val = breath_signal + heart_signal + noise
        
        # Update history for plotting
        v['history'].append(signal_val)
        if len(v['history']) > 50: # Keep last 50 points
            v['history'].pop(0)
            
        return int(current_hr), int(current_br), v['history']

class MMSimulator:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self._thread = None
        self.weapon_mode = False
        self.view_mode = "Full Debug" # Default
        self.card_inserted = False
        
        self.vitals_sim = VitalSignSimulator()
        
        # Load models using safe loader
        try:
            pose_path = get_resource_path('yolov8n-pose.pt')
            print(f"Loading pose model from: {pose_path}")
            self.model = safe_load_yolo(pose_path)
        except Exception:
            print("Error loading pose model")
            traceback.print_exc()
            self.model = None

        # Load weapon detection model
        try:
            weapon_path = get_resource_path('weapons.pt')
            print(f"Loading weapon model from: {weapon_path}")
            self.weapon_model = safe_load_yolo(weapon_path)
        except Exception:
            print("Warning: weapons.pt not found or failed to load. Weapon detection disabled.")
            self.weapon_model = None
            
        self.cap = None
        self.voice_alert = VoiceAlert(cooldown=4.0)
        
        # Tracking state
        self.locked_user_center = None
        
        # Recording state
        self.is_recording = False
        self.recording_frames = []
        self.recording_lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def toggle_weapon_mode(self):
        self.weapon_mode = not self.weapon_mode

    def set_view_mode(self, mode):
        self.view_mode = mode

    def set_card_inserted(self, inserted):
        self.card_inserted = inserted

    def save_recording(self):
        """Saves the buffered frames to a video file."""
        with self.recording_lock:
            if not self.recording_frames:
                print("No frames to save.")
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "recordings"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"evidence_{timestamp}.avi")
            
            # Get dimensions from first frame
            if len(self.recording_frames) > 0:
                height, width, layers = self.recording_frames[0].shape
                size = (width, height)
                
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                
                for i in range(len(self.recording_frames)):
                    out.write(self.recording_frames[i])
                out.release()
                print(f"Recording saved to {filename}")
            
            # Clear buffer after saving? Or keep until explicitly cleared?
            # Usually evidence is saved and we can clear or keep recording.
            # Let's clear to free memory, assuming "save" is the end of the event handling.
            self.recording_frames = []
            self.is_recording = False

    def clear_recording(self):
        """Stops recording and clears buffer without saving."""
        with self.recording_lock:
            self.is_recording = False
            self.recording_frames = []
            print("Recording cleared.")

    def _run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("ERROR: Camera not opened")
            self.running = False
            return

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                threat_detected = None
                height, width = frame.shape[:2]

                # Run person detection
                results = self.model(frame, verbose=False)
                # Use plot() to render skeletal points (keypoints) and boxes
                annotated = results[0].plot()

                # Run weapon detection if model is loaded AND weapon mode is active
                detected_weapon_boxes = []
                any_weapon_detected = False
                if self.weapon_mode and self.weapon_model:
                    weapon_results = self.weapon_model(frame, verbose=False)
                    for r in weapon_results:
                        for box in r.boxes:
                            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                            wconf = float(box.conf[0])
                            wcls_id = int(box.cls[0])
                            wlabel = self.weapon_model.names[wcls_id]
                            
                            # Filter for specific threats
                            threats = ['knife', 'scissors', 'gun', 'pistol', 'skimmer']
                            
                            if wlabel in threats and wconf > 0.4:
                                any_weapon_detected = True
                                detected_weapon_boxes.append((wx1, wy1, wx2, wy2))
                                label_text = f"{wlabel} {wconf:.2f}"
                                
                                # Draw red box for weapons
                                cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)
                                cv2.putText(annotated, label_text, (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                
                                # Alert
                                cv2.putText(annotated, f"REAL THREAT: {wlabel.upper()}", (50, 150), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                # Flash screen border red
                                cv2.rectangle(annotated, (0,0), (width, height), (0, 0, 255), 10)
                                
                                # Voice Alert
                                self.voice_alert.speak(f"Warning. Weapon detected.")

                # ATM Overlay on Annotated - MOVED TO END
                # cv2.putText(annotated, "ATM CAMERA 01", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                # cv2.putText(annotated, time.strftime("%Y-%m-%d %H:%M:%S"), (width - 250, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # if self.weapon_mode:
                #     cv2.putText(annotated, "WEAPON MODE: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # mm_img is a single channel image representing simulated mmWave returns
                mm_img = np.zeros((height, width), dtype=np.uint8)
                
                # Point cloud data
                all_pcl_points = []
                person_detected = False
                simulated_dist = 0.0

                # iterate results (could contain multiple result objects)
                detected_persons = []
                for r in results:
                    keypoints_data = r.keypoints.data if r.keypoints is not None else None
                    
                    # boxes available in r.boxes
                    for i, box in enumerate(r.boxes):
                        # get class and conf
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # only consider person class in COCO (class 0)
                        # Increased confidence threshold to 0.6 to avoid detecting hands/objects as persons
                        if cls_id != 0 or conf < 0.60:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Estimate distance
                        h_px = y2 - y1
                        dist_z = 0.0
                        if h_px > 0:
                            # Simple pinhole camera model estimation
                            focal_length = 600
                            real_height = 1.7
                            dist_z = (real_height * focal_length) / h_px

                        # Get keypoints
                        kpts = None
                        if keypoints_data is not None and len(keypoints_data) > i:
                            if keypoints_data[i].shape[0] > 0:
                                kpts = keypoints_data[i]

                        detected_persons.append({
                            'box': box,
                            'coords': (x1, y1, x2, y2),
                            'conf': conf,
                            'dist_z': dist_z,
                            'keypoints': kpts
                        })

                # Sort persons by distance (nearest first)
                detected_persons.sort(key=lambda p: p['dist_z'])
                
                nearest_person = None
                
                if detected_persons:
                    if self.card_inserted:
                        if self.locked_user_center:
                            # Find person closest to locked center (Tracking)
                            best_person = None
                            min_dist = float('inf')
                            
                            for p in detected_persons:
                                x1, y1, x2, y2 = p['coords']
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                dist = math.sqrt((cx - self.locked_user_center[0])**2 + (cy - self.locked_user_center[1])**2)
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_person = p
                            
                            # Threshold for tracking (e.g., 200 pixels movement between frames)
                            if best_person and min_dist < 200:
                                nearest_person = best_person
                                # Update locked center
                                nx1, ny1, nx2, ny2 = nearest_person['coords']
                                self.locked_user_center = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)
                            else:
                                # Lost tracking (moved too fast or disappeared), reset to nearest by Z
                                nearest_person = detected_persons[0]
                                nx1, ny1, nx2, ny2 = nearest_person['coords']
                                self.locked_user_center = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)
                        else:
                            # First lock on card insertion
                            nearest_person = detected_persons[0]
                            nx1, ny1, nx2, ny2 = nearest_person['coords']
                            self.locked_user_center = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)
                    else:
                        # No card, just show nearest (no locking)
                        self.locked_user_center = None
                        nearest_person = detected_persons[0]
                else:
                    self.locked_user_center = None

                # nearest_person = detected_persons[0] if detected_persons else None
                second_nearest = None
                if len(detected_persons) > 1:
                    # Find second nearest that is NOT the nearest_person
                    for p in detected_persons:
                        if p != nearest_person:
                            second_nearest = p
                            break
                
                current_user_has_weapon = False

                for p in detected_persons:
                    x1, y1, x2, y2 = p['coords']
                    conf = p['conf']
                    dist_z = p['dist_z']
                    keypoints = p['keypoints']
                    
                    person_detected = True
                    is_nearest = (p == nearest_person) and self.card_inserted
                    is_this_person_threat = False

                    # Check weapon intersection
                    for wb in detected_weapon_boxes:
                        wcx = (wb[0] + wb[2]) / 2
                        wcy = (wb[1] + wb[3]) / 2
                        if x1 < wcx < x2 and y1 < wcy < y2:
                            is_this_person_threat = True
                            if is_nearest:
                                current_user_has_weapon = True
                            break

                    # --- Keypoint Logic Integration ---
                    if keypoints is not None:
                        # YOLO Keypoint Indices: 
                        # 0: Nose, 1: L-Eye, 2: R-Eye, 5: L-Shoulder, 6: R-Shoulder, 9: L-Wrist, 10: R-Wrist
                        
                        nose = keypoints[0] 
                        left_eye = keypoints[1]
                        right_eye = keypoints[2]
                        left_shoulder = keypoints[5]
                        right_shoulder = keypoints[6]
                        left_wrist = keypoints[9]
                        right_wrist = keypoints[10]

                        # Confidence threshold to ensure keypoints are valid
                        conf_thresh = 0.5

                        # --- LOGIC 1: Secure Zone Visualization (Shoulder Width) ---
                        if left_shoulder[2] > conf_thresh and right_shoulder[2] > conf_thresh and nose[2] > conf_thresh:
                            # Calculate Euclidean distance
                            shoulder_width = math.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                                                     (left_shoulder[1] - right_shoulder[1])**2)
                            
                            safe_radius = int(shoulder_width * 1.5)
                            center_x, center_y = int(nose[0]), int(nose[1])
                            
                            cv2.circle(annotated, (center_x, center_y), safe_radius, (0, 255, 0), 2)

                        # --- LOGIC 2: Hands Up Detection ---
                        # Check if wrists are above eyes (Y value is smaller at the top)
                        if (left_wrist[2] > conf_thresh and right_wrist[2] > conf_thresh and 
                            left_eye[2] > conf_thresh and right_eye[2] > conf_thresh):
                            
                            if left_wrist[1] < left_eye[1] and right_wrist[1] < right_eye[1]:
                                threat_detected = "HANDS_UP"
                                is_this_person_threat = True
                                status = "THREAT: HANDS UP"
                                x = 10
                                y = 30
                                cv2.putText(annotated, status, ((x, y)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                # Flash screen border red
                                cv2.rectangle(annotated, (0,0), (width, height), (0, 0, 255), 10)
                                
                                # Voice Alert
                                self.voice_alert.speak("Warning. Hands up detected.")

                        # --- LOGIC 3: Peeking / Anomalous Position ---
                        is_peeking = False
                        peeking_msg = ""
                        
                        # Only detect peeking if there is an active user (card inserted)
                        # and this person is NOT the active user
                        if self.card_inserted and p != nearest_person:
                            # Condition A: Edge of frame
                            if nose[2] > conf_thresh:
                                nose_x = nose[0]
                                if nose_x < 0.2 * width or nose_x > 0.8 * width:
                                    is_peeking = True
                                    peeking_msg = "WARNING: PEEKING (SIDE)"
                            
                            # Condition B: Close to nearest person (Behind)
                            # Check if distance is close to nearest (e.g. within 1.0 unit)
                            if nearest_person and (p['dist_z'] - nearest_person['dist_z']) < 1.0:
                                is_peeking = True
                                peeking_msg = "WARNING: PEEKING (BEHIND)"

                        if is_peeking:
                            if not threat_detected:
                                threat_detected = "PEEKING"
                            is_this_person_threat = True
                            # Use nose if available, else top of box
                            txt_x = int(nose[0]) if nose[2] > conf_thresh else x1
                            txt_y = int(nose[1]) - 40 if nose[2] > conf_thresh else y1 - 40
                            cv2.putText(annotated, peeking_msg, (txt_x, txt_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                        # --- Privacy: Blur Face ---
                        # Collect face keypoints: Nose(0), Eyes(1,2), Ears(3,4)
                        face_pts = []
                        for i in [0, 1, 2, 3, 4]:
                            if len(keypoints) > i and keypoints[i][2] > 0.5:
                                face_pts.append(keypoints[i])
                        
                        if face_pts:
                            fxs = [int(p[0]) for p in face_pts]
                            fys = [int(p[1]) for p in face_pts]
                            min_x, max_x = min(fxs), max(fxs)
                            min_y, max_y = min(fys), max(fys)
                            
                            # Estimate head size from body box height (approx 1/7 of height)
                            box_h = y2 - y1
                            est_head_size = box_h / 7.0
                            
                            # Calculate dimensions of detected features
                            fw = max_x - min_x
                            fh = max_y - min_y
                            
                            # Use max dimension for robust padding
                            # If features are close (fw, fh small), use body-based estimate
                            scale = max(fw, fh, est_head_size)
                            
                            pad_x = int(scale * 0.2)
                            pad_y = int(scale * 0.4)
                            
                            bx1 = max(0, min_x - pad_x)
                            by1 = max(0, min_y - pad_y)
                            bx2 = min(width, max_x + pad_x)
                            by2 = min(height, max_y + pad_y)
                            
                            # Apply blur to annotated image
                            # Only blur if the person is NOT a threat (even if they are the current user)
                            if not is_this_person_threat and by2 > by1 and bx2 > bx1:
                                roi = annotated[by1:by2, bx1:bx2]
                                annotated[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (99, 99), 30)
                                
                                # Apply blur to original frame (propagates to overlay)
                                roi_f = frame[by1:by2, bx1:bx2]
                                frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi_f, (99, 99), 30)

                    # --- LOGIC 4: Weapon Simulation (Keyboard Trigger) ---
                    if self.weapon_mode and is_nearest and keypoints is not None:
                         # YOLO Keypoint Index 12: Right Hip
                         if len(keypoints) > 12:
                            right_hip = keypoints[12]
                            if right_hip[2] > conf_thresh:
                                hip_x, hip_y = int(right_hip[0]), int(right_hip[1])
                                
                                # Draw "Heatmap" (Red Blob)
                                overlay_blob = annotated.copy()
                                cv2.circle(overlay_blob, (hip_x, hip_y), 60, (0, 0, 255), -1)
                                annotated = cv2.addWeighted(overlay_blob, 0.6, annotated, 0.4, 0)
                                
                                # Text Alert
                                cv2.putText(annotated, "DENSITY ANOMALY DETECTED (mmWave)", (50, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Estimate distance for Point Cloud
                    if dist_z > 0:
                        focal_length = 600
                        cx = (x1 + x2) / 2
                        img_cx = width / 2
                        dist_x = (cx - img_cx) * dist_z / focal_length
                        
                        simulated_dist = dist_z # Use Z distance for display
                        
                        # Generate points for this person at the estimated location
                        # Shift y by -0.5 to put feet near floor (-1.0)
                        pts = generate_human_point_cloud(800, offset_x=dist_x, offset_y=-0.5, offset_z=dist_z)
                        all_pcl_points.append(pts)

                    # Draw person box on annotated frame
                    color = (0, 255, 0) if is_nearest else (0, 165, 255) # Green for nearest, Orange for others
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    if is_nearest:
                        cv2.putText(annotated, "CURRENT ATM USER", (x1, y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # --- VITAL SIGNS VISUALIZATION ---
                        hr, br, waveform = self.vitals_sim.get_vitals(0) # Use 0 as ID for current user
                        
                        # Draw Vitals Box
                        v_w = 150
                        v_h = 100
                        
                        # Smart Positioning: Try Right -> Left -> Inside
                        if x2 + 10 + v_w <= width:
                            v_x = x2 + 10
                            v_y = y1
                        elif x1 - 10 - v_w >= 0:
                            v_x = x1 - 10 - v_w
                            v_y = y1
                        else:
                            # Fallback: Inside top-right of bbox
                            v_x = x2 - v_w - 5
                            v_y = y1 + 5
                            
                        # Ensure Y is within bounds
                        v_y = max(0, min(v_y, height - v_h))

                        # Background
                        cv2.rectangle(annotated, (v_x, v_y), (v_x + v_w, v_y + v_h), (0, 0, 0), -1)
                        cv2.rectangle(annotated, (v_x, v_y), (v_x + v_w, v_y + v_h), (0, 255, 0), 1)
                        
                        # Text Info
                        cv2.putText(annotated, "VITAL SIGNS (mmWave)", (v_x + 5, v_y + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(annotated, f"HR: {hr} BPM", (v_x + 5, v_y + 35), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(annotated, f"BR: {br} RPM", (v_x + 5, v_y + 55), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Draw Waveform
                        if len(waveform) > 1:
                            graph_x_start = v_x + 5
                            graph_y_center = v_y + 80
                            graph_width = v_w - 10
                            step = graph_width / 50
                            
                            pts = []
                            for j, val in enumerate(waveform):
                                px = int(graph_x_start + j * step)
                                # Scale amplitude (val is roughly -1.2 to 1.2)
                                py = int(graph_y_center - (val * 10)) 
                                pts.append((px, py))
                            
                            cv2.polylines(annotated, [np.array(pts)], False, (0, 255, 0), 1)

                    cv2.putText(annotated, f"Dist: {dist_z * 0.1:.2f}m", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                    # Synthesize mmWave return inside person bbox
                    if self.weapon_mode:
                        synth = synthesize_mmwave_for_roi(mm_img, (x1,y1,x2,y2), keypoints=keypoints, probability=0.5)
                        # if synth:
                        #     cv2.putText(annotated, "sim_hidden_obj", (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

                # Determine final threat status based on weapon ownership
                if any_weapon_detected:
                    if not self.card_inserted or current_user_has_weapon:
                        threat_detected = "WEAPON_LOCK"
                    elif threat_detected is None:
                        threat_detected = "WEAPON"

                # Combine all points
                if all_pcl_points:
                    pcl_points = np.vstack(all_pcl_points)
                else:
                    pcl_points = np.empty((0, 3))
                
                if self.weapon_mode:
                    # Detect synthetic objects on mm_img
                    detections, mask = detect_synthetic_objects(mm_img)

                    # Visualize mm_img using a heatmap
                    colored_mm = cv2.applyColorMap(cv2.equalizeHist(mm_img), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.6, colored_mm, 0.4, 0)

                    # Draw detection boxes (from mm detection) on overlay and annotated
                    # for (x1,y1,x2,y2,area) in detections:
                    #     cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
                    #     cv2.putText(overlay, f"sim_obj {area:.0f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    #     cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                    #     cv2.putText(annotated, f"sim_obj", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    overlay = frame.copy()

                # --- Draw UI Overlays (Late Render to avoid blur overlap) ---
                cv2.putText(annotated, "ATM CAMERA 01", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(annotated, time.strftime("%Y-%m-%d %H:%M:%S"), (width - 250, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if self.weapon_mode:
                    cv2.putText(annotated, "WEAPON MODE: ON", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # --- Recording Logic ---
                if threat_detected and not self.is_recording:
                    with self.recording_lock:
                        self.is_recording = True
                        print(f"Threat detected ({threat_detected}). Starting recording...")

                if self.is_recording:
                    # Determine lighting condition (simple brightness check)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray_frame)
                    is_dark = avg_brightness < 80 # Threshold for dark
                    
                    frame_to_record = frame.copy()
                    
                    if is_dark:
                        # Simulate Infrared: Grayscale
                        frame_to_record = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                        cv2.putText(frame_to_record, "IR REC", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame_to_record, "REC", (width - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                    with self.recording_lock:
                        self.recording_frames.append(frame_to_record)

                # --- Render based on View Mode ---
                final_output = None
                
                if self.view_mode == "Full Debug":
                    # Top row: Video + Overlay
                    left = cv2.resize(annotated, (640,480))
                    right = cv2.resize(overlay, (640,480))
                    top_row = np.hstack((left, right))

                    # Bottom row: Point Cloud Comparison
                    pcl_high = render_point_cloud(pcl_points, 640, 480, f"High Res (3 GHz BW)", noise_level=0.02, point_size=2)
                    pcl_low_points = pcl_points[::4] if len(pcl_points) > 0 else []
                    pcl_low = render_point_cloud(pcl_low_points, 640, 480, f"Low Res (250 MHz BW)", noise_level=0.15, point_size=3)
                    
                    bottom_row = np.hstack((pcl_high, pcl_low))
                    final_output = np.vstack((top_row, bottom_row))
                
                elif self.view_mode == "Standard Monitoring":
                    # Just the annotated camera view, but bigger
                    final_output = cv2.resize(annotated, (1280, 960))
                
                elif self.view_mode == "Privacy Mode":
                    # Show mmWave overlay primarily, maybe blur the camera feed heavily
                    # For now, just show the overlay (which is camera + heatmap)
                    final_output = cv2.resize(overlay, (1280, 960))
                
                elif self.view_mode == "Security Mode":
                    # Show Point Cloud High Res
                    pcl_high = render_point_cloud(pcl_points, 1280, 960, f"Security Scan (3D Point Cloud)", noise_level=0.02, point_size=3)
                    final_output = pcl_high

                # Put the result in the queue
                if final_output is not None:
                    if not self.frame_queue.full():
                        self.frame_queue.put((final_output, threat_detected))
                    else:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put((final_output, threat_detected))
                        except queue.Empty:
                            pass
                
                # Small sleep to prevent CPU hogging if processing is fast
                time.sleep(0.01)

        except Exception:
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
