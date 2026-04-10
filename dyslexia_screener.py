#!/usr/bin/env python3
"""
Dyslexia Screener - Webcam-Based Eye-Tracking with MediaPipe
==============================================================
Uses MediaPipe FaceMesh iris landmarks to track gaze while the user
reads three text passages. Extracts the same 21 features used in training,
then predicts dyslexia risk using the tuned SVM model.

Architecture:
  Phase 0 - Instructions
  Phase 1 - 9-point calibration (maps iris position -> screen coords)
  Phase 2 - Reading Task T1 (Syllables)
  Phase 3 - Reading Task T4 (Meaningful Text)
  Phase 4 - Reading Task T5 (Pseudo-Text)
  Phase 5 - Feature extraction + Model prediction + Results

Requirements: mediapipe, opencv-python, numpy, joblib, scikit-learn
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = Path(r"c:\Users\ankit\Downloads\13332134 (1)")

# Calibration
CALIB_POINTS = 9          # 3x3 grid
CALIB_DWELL_SEC = 2.0     # seconds per point
CALIB_MARGIN = 0.12       # margin from screen edge (fraction)

# Fixation detection (I-DT: Identification by Dispersion Threshold)
FIXATION_DISPERSION_PX = 30    # reduced to detect more fixations (closer to lab count)
FIXATION_MIN_DURATION_MS = 80  # reduced to capture shorter fixations

# Gaze smoothing
GAZE_SMOOTH_WINDOW = 15        # strong smoothing to reduce webcam jitter

# Reading tasks: line-based ROIs will be computed from these texts
READING_TEXTS = {
    "T1": {
        "title": "Task 1: Syllable Reading",
        "instruction": "Read each syllable aloud, left to right, top to bottom.",
        "lines": [
            "ba   ke   li   mo   su   ta   ri   po   de   na",
            "fi   gu   ra   te   mu   lo   ni   sa   je   ku",
            "pe   di   vo   la   si   bu   me   ro   ta   fu",
            "ka   ne   li   to   su   ra   mi   go   be   da",
            "nu   fe   si   po   la   te   ri   mu   ko   ba",
            "do   li   me   ta   gu   ne   fi   ra   su   ke",
            "pi   bo   na   le   fu   di   so   ka   me   tu",
            "re   ga   bi   no   su   le   ta   mi   ko   fe",
            "da   lu   pe   ri   mo   sa   te   bu   ni   go",
            "ku   fi   la   de   ro   me   na   si   po   ta",
        ],
    },
    "T4": {
        "title": "Task 4: Meaningful Text",
        "instruction": "Read the following paragraph silently at your normal pace.",
        "lines": [
            "Little Peter preferred to stay at home reading books full of",
            "adventures rather than playing outside with the other children.",
            "Perhaps his injured leg contributed to this preference since it",
            "prevented him from many outdoor activities. He would rather",
            "immerse himself in stories of sneaking scouts, brave explorers,",
            "or wicked pirates on the high seas. Sometimes he sadly gazed",
            "out the window at his friends playing and watched their games",
        ],
    },
    "T5": {
        "title": "Task 5: Pseudo-Text Reading",
        "instruction": "Read the following nonsense text silently, trying to decode each word.",
        "lines": [
            "Datik Herpel blufened ko sall og hemon broding guls em",
            "padvenglors blather vurn plassing imbide gith ner hoger smildren",
            "Burheps nis grinjored neg fontriboted ko gris weference shince ot",
            "trevended nim grom fany imtdoor lavitivies. Ne wuold blather",
            "grimmerse nimself om stokies em snirking scalts, drabe eploxers,",
            "ir blicked tipares un ner sigh teas. Cometimes ne ladsy daseg",
            "olt ner blondow ig nis griends tlaming und blotched deir mages",
        ],
    },
}

# Colors (BGR)
COL_BG       = (30, 30, 30)
COL_TEXT     = (230, 230, 230)
COL_TITLE    = (100, 200, 255)
COL_ACCENT   = (0, 180, 255)
COL_GREEN    = (0, 200, 100)
COL_RED      = (0, 80, 220)
COL_CALIB    = (0, 255, 255)
COL_GAZE     = (0, 200, 0)
COL_DIM      = (120, 120, 120)


# =====================================================================
# DATA CLASSES
# =====================================================================
@dataclass
class GazePoint:
    x: float
    y: float
    timestamp_ms: float

@dataclass
class Fixation:
    x: float           # centroid x
    y: float           # centroid y
    start_ms: float
    end_ms: float
    duration_ms: float
    line_roi: int = -1  # which line ROI it belongs to (-1 = none)

@dataclass
class Saccade:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    amplitude: float
    direction_x: float  # positive = forward (left-to-right)


# =====================================================================
# GAZE TRACKER (MediaPipe Iris)
# =====================================================================
class GazeTracker:
    """Extracts iris landmark positions from webcam frames using MediaPipe."""

    # MediaPipe FaceMesh landmark indices
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.smooth_buffer_x = deque(maxlen=GAZE_SMOOTH_WINDOW)
        self.smooth_buffer_y = deque(maxlen=GAZE_SMOOTH_WINDOW)

    def get_iris_ratio(self, frame) -> Optional[Tuple[float, float]]:
        """
        Returns the normalized iris position within the eye opening.
        (0,0) = looking far left & up, (1,1) = looking far right & down.
        Returns None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Get iris centers and eye corners (average both eyes)
        l_iris = np.array([landmarks[self.LEFT_IRIS_CENTER].x,
                           landmarks[self.LEFT_IRIS_CENTER].y])
        r_iris = np.array([landmarks[self.RIGHT_IRIS_CENTER].x,
                           landmarks[self.RIGHT_IRIS_CENTER].y])

        # Left eye horizontal range
        l_inner = landmarks[self.LEFT_EYE_INNER].x
        l_outer = landmarks[self.LEFT_EYE_OUTER].x
        l_top = landmarks[self.LEFT_EYE_TOP].y
        l_bot = landmarks[self.LEFT_EYE_BOTTOM].y

        # Right eye horizontal range
        r_inner = landmarks[self.RIGHT_EYE_INNER].x
        r_outer = landmarks[self.RIGHT_EYE_OUTER].x
        r_top = landmarks[self.RIGHT_EYE_TOP].y
        r_bot = landmarks[self.RIGHT_EYE_BOTTOM].y

        # Horizontal ratio for each eye
        l_h_range = abs(l_inner - l_outer)
        r_h_range = abs(r_inner - r_outer)
        if l_h_range < 0.001 or r_h_range < 0.001:
            return None

        l_ratio_x = (l_iris[0] - min(l_inner, l_outer)) / l_h_range
        r_ratio_x = (r_iris[0] - min(r_inner, r_outer)) / r_h_range

        # Vertical ratio for each eye
        l_v_range = abs(l_bot - l_top)
        r_v_range = abs(r_bot - r_top)
        if l_v_range < 0.001 or r_v_range < 0.001:
            return None

        l_ratio_y = (l_iris[1] - l_top) / l_v_range
        r_ratio_y = (r_iris[1] - r_top) / r_v_range

        # Average both eyes
        ratio_x = (l_ratio_x + r_ratio_x) / 2.0
        ratio_y = (l_ratio_y + r_ratio_y) / 2.0

        # Smooth
        self.smooth_buffer_x.append(ratio_x)
        self.smooth_buffer_y.append(ratio_y)

        sx = np.mean(self.smooth_buffer_x)
        sy = np.mean(self.smooth_buffer_y)

        # OVERLAY: Draw eye landmarks on the webcam frame for the user to see
        # Draw Irises (Yellow)
        for idx in [self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER]:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 3, (0, 255, 255), -1)
        # Draw Eye Corners (Red)
        for idx in [self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER]:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 0, 255), -1)
        # Draw Top/Bottom lids (Green)
        for idx in [self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM]:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

        return (sx, sy)

    def close(self):
        self.face_mesh.close()


# =====================================================================
# CALIBRATOR (Maps iris ratio -> screen pixels)
# =====================================================================
class Calibrator:
    """9-point calibration to map iris ratios to screen coordinates."""

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.calib_data = []  # list of (screen_x, screen_y, iris_ratio_x, iris_ratio_y)
        self.transform_x = None  # polynomial coefficients
        self.transform_y = None

    def get_calibration_points(self) -> List[Tuple[int, int]]:
        """Returns the 9 screen points for calibration."""
        margin_x = int(self.screen_w * CALIB_MARGIN)
        margin_y = int(self.screen_h * CALIB_MARGIN)
        cx, cy = self.screen_w // 2, self.screen_h // 2
        right = self.screen_w - margin_x
        bottom = self.screen_h - margin_y

        points = [
            (margin_x, margin_y),       # top-left
            (cx, margin_y),             # top-center
            (right, margin_y),          # top-right
            (margin_x, cy),             # mid-left
            (cx, cy),                   # center
            (right, cy),               # mid-right
            (margin_x, bottom),        # bottom-left
            (cx, bottom),              # bottom-center
            (right, bottom),           # bottom-right
        ]
        return points

    def add_sample(self, screen_x, screen_y, iris_rx, iris_ry):
        self.calib_data.append((screen_x, screen_y, iris_rx, iris_ry))

    def compute_mapping(self):
        """Fit a polynomial mapping from iris ratios to screen coords."""
        if len(self.calib_data) < 4:
            print("  [WARN] Not enough calibration data, using linear fallback.")
            self.transform_x = lambda rx, ry: rx * self.screen_w
            self.transform_y = lambda rx, ry: ry * self.screen_h
            return

        data = np.array(self.calib_data)
        sx, sy = data[:, 0], data[:, 1]
        rx, ry = data[:, 2], data[:, 3]

        # Fit 2nd-order polynomial: screen = a*rx^2 + b*ry^2 + c*rx*ry + d*rx + e*ry + f
        A = np.column_stack([rx**2, ry**2, rx*ry, rx, ry, np.ones_like(rx)])

        # Solve for x mapping
        coeffs_x, _, _, _ = np.linalg.lstsq(A, sx, rcond=None)
        # Solve for y mapping
        coeffs_y, _, _, _ = np.linalg.lstsq(A, sy, rcond=None)

        self.transform_x = lambda irx, iry: float(
            np.clip(coeffs_x[0]*irx**2 + coeffs_x[1]*iry**2 +
                    coeffs_x[2]*irx*iry + coeffs_x[3]*irx +
                    coeffs_x[4]*iry + coeffs_x[5], 0, self.screen_w))
        self.transform_y = lambda irx, iry: float(
            np.clip(coeffs_y[0]*irx**2 + coeffs_y[1]*iry**2 +
                    coeffs_y[2]*irx*iry + coeffs_y[3]*irx +
                    coeffs_y[4]*iry + coeffs_y[5], 0, self.screen_h))

    def map_to_screen(self, iris_rx, iris_ry) -> Tuple[int, int]:
        """Convert iris ratios to screen pixel coordinates."""
        if self.transform_x is None:
            return (int(iris_rx * self.screen_w), int(iris_ry * self.screen_h))
        return (int(self.transform_x(iris_rx, iris_ry)),
                int(self.transform_y(iris_rx, iris_ry)))


# =====================================================================
# FIXATION DETECTOR (I-DT Algorithm)
# =====================================================================
class FixationDetector:
    """Identifies fixations from a gaze point stream using I-DT algorithm."""

    def __init__(self):
        self.window: List[GazePoint] = []
        self.fixations: List[Fixation] = []
        self.saccades: List[Saccade] = []

    def reset(self):
        self.window.clear()
        self.fixations.clear()
        self.saccades.clear()

    def add_point(self, gp: GazePoint):
        """Add a gaze point and check if a fixation has ended."""
        self.window.append(gp)

        if len(self.window) < 3:
            return

        # Compute dispersion of current window
        xs = [p.x for p in self.window]
        ys = [p.y for p in self.window]
        dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))

        if dispersion <= FIXATION_DISPERSION_PX:
            # Window is still a fixation, keep growing
            return
        else:
            # Dispersion exceeded: finalize the fixation (all points except last)
            if len(self.window) >= 3:
                fix_points = self.window[:-1]
                dur = fix_points[-1].timestamp_ms - fix_points[0].timestamp_ms

                if dur >= FIXATION_MIN_DURATION_MS:
                    cx = np.mean([p.x for p in fix_points])
                    cy = np.mean([p.y for p in fix_points])
                    fix = Fixation(
                        x=cx, y=cy,
                        start_ms=fix_points[0].timestamp_ms,
                        end_ms=fix_points[-1].timestamp_ms,
                        duration_ms=dur
                    )
                    # Create saccade from previous fixation
                    if self.fixations:
                        prev = self.fixations[-1]
                        dx = fix.x - prev.x
                        dy = fix.y - prev.y
                        amp = np.sqrt(dx**2 + dy**2)
                        self.saccades.append(Saccade(
                            start_x=prev.x, start_y=prev.y,
                            end_x=fix.x, end_y=fix.y,
                            amplitude=amp, direction_x=dx
                        ))
                    self.fixations.append(fix)

            # Start new window from the last point
            last = self.window[-1]
            self.window.clear()
            self.window.append(last)

    def finalize(self):
        """Call after reading is done to capture any remaining fixation."""
        if len(self.window) >= 3:
            dur = self.window[-1].timestamp_ms - self.window[0].timestamp_ms
            if dur >= FIXATION_MIN_DURATION_MS:
                cx = np.mean([p.x for p in self.window])
                cy = np.mean([p.y for p in self.window])
                fix = Fixation(
                    x=cx, y=cy,
                    start_ms=self.window[0].timestamp_ms,
                    end_ms=self.window[-1].timestamp_ms,
                    duration_ms=dur
                )
                if self.fixations:
                    prev = self.fixations[-1]
                    dx = fix.x - prev.x
                    dy = fix.y - prev.y
                    amp = np.sqrt(dx**2 + dy**2)
                    self.saccades.append(Saccade(
                        start_x=prev.x, start_y=prev.y,
                        end_x=fix.x, end_y=fix.y,
                        amplitude=amp, direction_x=dx
                    ))
                self.fixations.append(fix)
        self.window.clear()


# =====================================================================
# FEATURE EXTRACTOR
# =====================================================================
class FeatureExtractor:
    """Computes the 21 model features from fixations and saccades."""

    def __init__(self, line_rois: List[Tuple[int, int]]):
        """
        line_rois: list of (y_top, y_bottom) for each text line on screen.
        """
        self.line_rois = line_rois

    def assign_line_rois(self, fixations: List[Fixation]):
        """Tag each fixation with its line ROI index."""
        for fix in fixations:
            fix.line_roi = -1
            for i, (y_top, y_bot) in enumerate(self.line_rois):
                if y_top - 20 <= fix.y <= y_bot + 20:  # 20px tolerance
                    fix.line_roi = i
                    break

    def extract(self, fixations: List[Fixation], saccades: List[Saccade],
                task_prefix: str) -> dict:
        """Extract features for one task. Returns dict with prefixed keys."""
        feat = {}

        # --- Group A features (temporal / count-based) ---
        if len(fixations) == 0:
            for k in ["fix_count", "fix_dur_mean", "fix_dur_sd",
                       "fix_dur_median", "total_read_time"]:
                feat[f"{task_prefix}_{k}"] = 0.0
        else:
            durations = np.array([f.duration_ms for f in fixations])
            feat[f"{task_prefix}_fix_count"] = len(fixations)
            feat[f"{task_prefix}_fix_dur_mean"] = float(np.mean(durations))
            feat[f"{task_prefix}_fix_dur_sd"] = float(np.std(durations)) if len(durations) > 1 else 0.0
            feat[f"{task_prefix}_fix_dur_median"] = float(np.median(durations))
            feat[f"{task_prefix}_total_read_time"] = fixations[-1].end_ms - fixations[0].start_ms

        # --- Group B features ---
        # Gaze linearity: total path length / straight-line displacement
        if len(fixations) >= 2:
            path_length = sum(
                np.sqrt((fixations[i+1].x - fixations[i].x)**2 +
                        (fixations[i+1].y - fixations[i].y)**2)
                for i in range(len(fixations) - 1)
            )
            displacement = np.sqrt(
                (fixations[-1].x - fixations[0].x)**2 +
                (fixations[-1].y - fixations[0].y)**2
            )
            # Use mean absolute y-deviation (matches training data definition)
            y_diffs = [abs(fixations[i+1].y - fixations[i].y)
                       for i in range(len(fixations) - 1)]
            feat[f"{task_prefix}_gaze_linearity"] = float(np.mean(y_diffs))
        else:
            feat[f"{task_prefix}_gaze_linearity"] = 0.0

        # Revisit count: number of times gaze returns to a previously visited line
        self.assign_line_rois(fixations)
        visited_lines = set()
        revisit_count = 0
        for fix in fixations:
            if fix.line_roi >= 0:
                if fix.line_roi in visited_lines:
                    revisit_count += 1
                visited_lines.add(fix.line_roi)
        feat[f"{task_prefix}_revisit_count"] = revisit_count

        return feat


# =====================================================================
# DOMAIN ADAPTER (Webcam -> ETDD70 Training Distribution)
# =====================================================================
class DomainAdapter:
    """
    Bridges the domain gap between webcam (30fps) and lab tracker (250Hz).
    
    Strategy: We define the MEDIAN webcam value for a normal reader as the
    anchor point, mapping it to the non-dyslexic mean from training data.
    Deviations above the webcam median are scaled proportionally into the
    dyslexic range. No hard clamping -- values can extrapolate freely.
    
    This preserves relative differences: a user who fixates more/longer
    than average will score closer to dyslexic, and vice versa.
    """
    
    # (webcam_nondys_typical, webcam_spread, train_nondys_mean, train_dys_mean)
    # webcam_nondys_typical = what we expect from a normal reader on webcam
    # webcam_spread = expected range of variation on webcam
    FEATURE_PROFILES = {
        #                        webcam_typical  webcam_spread  train_ND    train_DYS
        "fix_count":             (35,            50,            155,        195),
        "fix_dur_mean":          (200,           200,           437,        575),
        "fix_dur_sd":            (120,           250,           299,        509),
        "fix_dur_median":        (150,           180,           381,        431),
        "total_read_time":       (25000,         30000,         72681,      121980),
        "gaze_linearity":        (200,           300,           4.5,        3.6),
        "revisit_count":         (8,             15,            24,         36),
    }
    
    TASK_SCALE = {
        "t1": 1.0,
        "t4": 1.15,
        "t5": 1.55,
    }
    
    @classmethod
    def adapt(cls, features: dict) -> dict:
        """Rescale webcam features into ETDD70 training distribution."""
        adapted = {}
        
        for fname, fval in features.items():
            parts = fname.split("_", 1)
            if len(parts) != 2:
                adapted[fname] = fval
                continue
            
            task_prefix = parts[0]
            feat_name = parts[1]
            
            if feat_name in cls.FEATURE_PROFILES:
                wc_typical, wc_spread, train_nd, train_dys = cls.FEATURE_PROFILES[feat_name]
                task_mult = cls.TASK_SCALE.get(task_prefix, 1.0)
                
                t_nd = train_nd * task_mult
                t_dys = train_dys * task_mult
                
                # How far is this webcam value from "typical non-dyslexic"?
                if wc_spread > 0:
                    deviation = (fval - wc_typical) / wc_spread
                else:
                    deviation = 0.0
                
                # Special handling for gaze_linearity (inverted in training data)
                if feat_name == "gaze_linearity":
                    # High webcam linearity = noisy tracking, NOT dyslexia
                    # Low webcam linearity = smooth reading = non-dyslexic
                    # In training data: LOWER linearity = MORE dyslexic
                    # Map: low webcam -> non-dys (high training), high webcam -> non-dys (high training)
                    # Center on non-dyslexic mean, deviation pushes toward dys
                    adapted_val = t_nd + deviation * (t_nd - t_dys)
                    # Clamp to reasonable range
                    adapted_val = max(t_dys * 0.5, min(t_nd * 1.5, adapted_val))
                else:
                    # Normal: higher webcam value -> higher training value (toward dyslexic)
                    adapted_val = t_nd + deviation * (t_dys - t_nd)
                    # Soft clamp: allow 50% beyond the training range
                    low_bound = t_nd - 0.5 * abs(t_dys - t_nd)
                    high_bound = t_dys + 0.5 * abs(t_dys - t_nd)
                    adapted_val = max(low_bound, min(high_bound, adapted_val))
                
                adapted[fname] = adapted_val
            else:
                adapted[fname] = fval
        
        return adapted


# =====================================================================
# SCREEN RENDERER
# =====================================================================
class Renderer:
    """Handles all OpenCV drawing for the screener UI."""

    def __init__(self, width, height):
        self.w = width
        self.h = height

    def blank(self) -> np.ndarray:
        return np.full((self.h, self.w, 3), COL_BG, dtype=np.uint8)

    def draw_text_centered(self, img, text, y, color=COL_TEXT, scale=0.7,
                           thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (self.w - size[0]) // 2
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def draw_instructions(self, title, lines):
        img = self.blank()
        self.draw_text_centered(img, title, 80, COL_TITLE, 1.0, 2)
        for i, line in enumerate(lines):
            self.draw_text_centered(img, line, 160 + i * 40, COL_TEXT, 0.6, 1)
        self.draw_text_centered(img, "Press SPACE to continue",
                                self.h - 60, COL_ACCENT, 0.7, 1)
        return img

    def draw_calibration_point(self, point, progress_frac, point_idx, total):
        img = self.blank()
        x, y = point
        # Outer ring
        cv2.circle(img, (x, y), 30, COL_DIM, 2, cv2.LINE_AA)
        # Progress arc
        angle = int(360 * progress_frac)
        if angle > 0:
            cv2.ellipse(img, (x, y), (30, 30), -90, 0, angle, COL_CALIB, 3, cv2.LINE_AA)
        # Center dot
        cv2.circle(img, (x, y), 6, COL_CALIB, -1, cv2.LINE_AA)
        # Counter
        self.draw_text_centered(img, f"Calibration Point {point_idx+1}/{total}",
                                self.h - 40, COL_DIM, 0.5, 1)
        self.draw_text_centered(img, "Look at the yellow dot",
                                self.h - 70, COL_ACCENT, 0.6, 1)
        return img

    def draw_reading_task(self, task_data, gaze_xy=None, progress_text=""):
        """Render a reading passage and optionally overlay gaze position."""
        img = self.blank()

        # Title
        self.draw_text_centered(img, task_data["title"], 50, COL_TITLE, 0.8, 2)
        # Instruction
        self.draw_text_centered(img, task_data["instruction"], 90, COL_DIM, 0.5, 1)

        # Text lines
        start_y = 180
        line_height = 75
        line_rois = []  # (y_top, y_bottom) for each line

        for i, line in enumerate(task_data["lines"]):
            y = start_y + i * line_height
            # Make the hit-box much larger vertically so exact gaze isn't as required
            y_top = y - 35
            y_bot = y + 25
            line_rois.append((y_top, y_bot))

            # Draw line number
            cv2.putText(img, f"{i+1:2d}.", (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, COL_DIM, 1, cv2.LINE_AA)
            # Draw text
            cv2.putText(img, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COL_TEXT, 1, cv2.LINE_AA)

        # Draw gaze dot
        if gaze_xy is not None:
            gx, gy = int(gaze_xy[0]), int(gaze_xy[1])
            cv2.circle(img, (gx, gy), 8, COL_GAZE, -1, cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 12, COL_GAZE, 1, cv2.LINE_AA)

        # Progress / status
        if progress_text:
            self.draw_text_centered(img, progress_text, self.h - 30, COL_DIM, 0.5, 1)

        # Instruction to finish
        self.draw_text_centered(img, "Press SPACE when done reading",
                                self.h - 60, COL_ACCENT, 0.6, 1)

        return img, line_rois

    def draw_results(self, prediction, confidence, features, model_name):
        img = self.blank()

        self.draw_text_centered(img, "DYSLEXIA SCREENING RESULTS", 60, COL_TITLE, 1.0, 2)

        if prediction == 1:
            label = "HIGHER RISK OF DYSLEXIA"
            color = COL_RED
        else:
            label = "LOWER RISK OF DYSLEXIA"
            color = COL_GREEN

        self.draw_text_centered(img, label, 130, color, 1.2, 2)
        self.draw_text_centered(img, f"Confidence: {confidence:.1f}%",
                                175, COL_TEXT, 0.7, 1)
        self.draw_text_centered(img, f"Model: {model_name}",
                                210, COL_DIM, 0.5, 1)

        # Feature summary
        y = 270
        self.draw_text_centered(img, "-- Extracted Features --", y, COL_ACCENT, 0.6, 1)
        y += 35

        # Display features in 2 columns
        feat_items = list(features.items())
        col_w = self.w // 2 - 40
        for i, (fname, fval) in enumerate(feat_items):
            col = i % 2
            row = i // 2
            x = 40 + col * col_w
            fy = y + row * 25
            short_name = fname.replace("_", " ")
            cv2.putText(img, f"{short_name}: {fval:.1f}", (x, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)

        self.draw_text_centered(img, "Press ESC to exit | Press R to re-run",
                                self.h - 40, COL_ACCENT, 0.6, 1)
        return img

    def overlay_webcam(self, img, frame):
        """Draws a picture-in-picture webcam view in the bottom right corner."""
        if frame is None:
            return
        
        # Resize webcam frame to 320x240 for Picture-in-Picture
        pip_w, pip_h = 320, 240
        pip = cv2.resize(frame, (pip_w, pip_h))
        
        # Mirror the PIP so it acts like a mirror
        pip = cv2.flip(pip, 1)
        
        # Position at bottom right with 20px padding
        x_offset = self.w - pip_w - 20
        y_offset = self.h - pip_h - 20
        
        # Draw white border
        cv2.rectangle(img, (x_offset - 2, y_offset - 2), 
                      (x_offset + pip_w + 2, y_offset + pip_h + 2), 
                      COL_TEXT, 2)
        
        # Overlay the PIP image
        img[y_offset:y_offset+pip_h, x_offset:x_offset+pip_w] = pip


# =====================================================================
# MAIN APPLICATION
# =====================================================================
class DyslexiaScreener:
    """Main application orchestrating the full screening flow."""

    def __init__(self):
        # Get screen size
        self.screen_w = 1280  # will be updated from actual window
        self.screen_h = 720

        self.cap = None
        self.tracker = GazeTracker()
        self.calibrator = None
        self.detector = FixationDetector()
        self.renderer = None

        # Load trained model
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feat_config = None
        self._load_model()

    def _load_model(self):
        """Load the tuned model and preprocessing artifacts."""
        # Prefer Logistic Regression (more robust to domain shift)
        try:
            self.model = joblib.load(BASE_DIR / "tuned_logistic_regression.joblib")
            self.scaler = joblib.load(BASE_DIR / "tuned_scaler.joblib")
            self.imputer = joblib.load(BASE_DIR / "tuned_imputer.joblib")
            self.feat_config = joblib.load(BASE_DIR / "final_feature_config.joblib")
            print("[OK] Model loaded: tuned_logistic_regression.joblib")
        except Exception as e:
            print(f"[WARN] Could not load LR, trying SVM: {e}")
            try:
                self.model = joblib.load(BASE_DIR / "tuned_svm_rbf.joblib")
                self.scaler = joblib.load(BASE_DIR / "tuned_scaler.joblib")
                self.imputer = joblib.load(BASE_DIR / "tuned_imputer.joblib")
                self.feat_config = joblib.load(BASE_DIR / "final_feature_config.joblib")
                print("[OK] Fallback model loaded: tuned_svm_rbf.joblib")
            except Exception as e2:
                print(f"[ERROR] No model found: {e2}")
                self.model = None

    def run(self):
        """Main entry point."""
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open webcam!")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create fullscreen window
        cv2.namedWindow("Dyslexia Screener", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Dyslexia Screener", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        # Get actual screen dimensions from the window
        # Use a reasonable default; user can resize
        self.screen_w = 1280
        self.screen_h = 720

        self.renderer = Renderer(self.screen_w, self.screen_h)
        self.calibrator = Calibrator(self.screen_w, self.screen_h)

        try:
            # Phase 0: Instructions
            if not self._phase_instructions():
                return

            # Phase 1: Calibration
            if not self._phase_calibration():
                return

            # Phases 2-4: Reading tasks
            all_features = {}
            for task_id in ["T1", "T4", "T5"]:
                features = self._phase_reading(task_id)
                if features is None:
                    return
                all_features.update(features)

            # Phase 5: Prediction
            self._phase_prediction(all_features)

        finally:
            self.cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()

    def _read_frame(self):
        """Read a webcam frame."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def _phase_instructions(self) -> bool:
        """Show welcome screen."""
        img = self.renderer.draw_instructions(
            "DYSLEXIA SCREENING TEST",
            [
                "This test will track your eye movements while you read.",
                "",
                "You will complete 3 short reading tasks:",
                "  1. Syllable reading (isolated syllables)",
                "  2. Meaningful text (a paragraph)",
                "  3. Pseudo-text (nonsense words)",
                "",
                "Before starting, you will calibrate the eye tracker.",
                "Please sit comfortably ~50cm from the screen.",
                "Keep your head still during the test.",
                "",
                "The test takes approximately 3-5 minutes.",
            ]
        )

        while True:
            cv2.imshow("Dyslexia Screener", img)
            key = cv2.waitKey(30) & 0xFF
            if key == 32:  # SPACE
                return True
            if key == 27:  # ESC
                return False

    def _phase_calibration(self) -> bool:
        """Run 9-point calibration."""
        points = self.calibrator.get_calibration_points()

        for idx, (px, py) in enumerate(points):
            start_time = time.time()
            samples = []

            while True:
                elapsed = time.time() - start_time
                progress = min(elapsed / CALIB_DWELL_SEC, 1.0)

                # Draw calibration screen
                img = self.renderer.draw_calibration_point(
                    (px, py), progress, idx, len(points))

                # Track iris
                frame = self._read_frame()
                if frame is not None:
                    ratio = self.tracker.get_iris_ratio(frame)
                    if ratio is not None and elapsed > 0.5:  # skip first 0.5s settling
                        samples.append(ratio)
                    self.renderer.overlay_webcam(img, frame)

                cv2.imshow("Dyslexia Screener", img)

                key = cv2.waitKey(10) & 0xFF
                if key == 27:
                    return False

                if elapsed >= CALIB_DWELL_SEC:
                    break

            # Average the samples for this calibration point
            if samples:
                avg_rx = np.mean([s[0] for s in samples])
                avg_ry = np.mean([s[1] for s in samples])
                self.calibrator.add_sample(px, py, avg_rx, avg_ry)
                print(f"  Calib point {idx+1}: screen=({px},{py}) "
                      f"iris=({avg_rx:.4f},{avg_ry:.4f}) "
                      f"samples={len(samples)}")

        self.calibrator.compute_mapping()
        print("[OK] Calibration complete.")

        # Show brief confirmation
        img = self.renderer.blank()
        self.renderer.draw_text_centered(img, "Calibration Complete!",
                                         self.screen_h // 2 - 20, COL_GREEN, 1.0, 2)
        self.renderer.draw_text_centered(img, "Press SPACE to begin reading tasks",
                                         self.screen_h // 2 + 30, COL_ACCENT, 0.6, 1)
        cv2.imshow("Dyslexia Screener", img)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 32:
                return True
            if key == 27:
                return False

    def _phase_reading(self, task_id: str) -> Optional[dict]:
        """Run one reading task and return extracted features."""
        task_data = READING_TEXTS[task_id]
        prefix = task_id.lower()  # "t1", "t4", "t5"

        # Show pre-task instruction
        img = self.renderer.draw_instructions(
            task_data["title"],
            [
                task_data["instruction"],
                "",
                "The text will appear on the next screen.",
                "A green dot shows your estimated gaze position.",
                "Press SPACE when you have finished reading.",
            ]
        )
        cv2.imshow("Dyslexia Screener", img)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 32:
                break
            if key == 27:
                return None

        # Reset detector for this task
        self.detector.reset()
        start_time_ms = time.time() * 1000

        # Pre-render line positions
        _, line_rois = self.renderer.draw_reading_task(task_data)

        # Reading loop
        frame_count = 0
        gaze_xy = None

        while True:
            frame = self._read_frame()
            now_ms = time.time() * 1000 - start_time_ms

            if frame is not None:
                ratio = self.tracker.get_iris_ratio(frame)
                if ratio is not None:
                    sx, sy = self.calibrator.map_to_screen(ratio[0], ratio[1])
                    gaze_xy = (sx, sy)

                    gp = GazePoint(x=sx, y=sy, timestamp_ms=now_ms)
                    self.detector.add_point(gp)
                    frame_count += 1

            # Draw reading screen with gaze overlay
            elapsed_sec = now_ms / 1000
            progress = f"Time: {elapsed_sec:.1f}s | Fixations: {len(self.detector.fixations)} | Frames: {frame_count}"
            img, _ = self.renderer.draw_reading_task(task_data, gaze_xy, progress)
            
            if frame is not None:
                self.renderer.overlay_webcam(img, frame)
                
            cv2.imshow("Dyslexia Screener", img)

            key = cv2.waitKey(10) & 0xFF
            if key == 32:  # SPACE = done reading
                break
            if key == 27:  # ESC = abort
                return None

        # Finalize: capture any remaining fixation
        self.detector.finalize()

        fixations = self.detector.fixations
        saccades = self.detector.saccades

        print(f"\n  [{task_id}] Fixations: {len(fixations)}, "
              f"Saccades: {len(saccades)}, "
              f"Frames: {frame_count}")

        # Extract features
        extractor = FeatureExtractor(line_rois)
        features = extractor.extract(fixations, saccades, prefix)

        for k, v in features.items():
            print(f"    {k}: {v:.2f}")

        return features

    def _phase_prediction(self, features: dict):
        """Run the model on extracted features and show results."""
        if self.model is None:
            print("[ERROR] No model loaded. Cannot predict.")
            img = self.renderer.blank()
            self.renderer.draw_text_centered(img, "ERROR: No model file found!",
                                             self.screen_h // 2, COL_RED, 1.0, 2)
            cv2.imshow("Dyslexia Screener", img)
            cv2.waitKey(0)
            return

        # === DOMAIN ADAPTATION: Rescale webcam features to training range ===
        print("\n  [Domain Adaptation] Raw webcam -> ETDD70 training range:")
        adapted_features = DomainAdapter.adapt(features)
        
        for fname in features:
            raw = features[fname]
            adp = adapted_features[fname]
            print(f"    {fname:<30s}  raw={raw:10.2f}  -> adapted={adp:10.2f}")

        # Build feature vector in the exact order the model expects
        feat_names = self.feat_config["feature_names"]
        X = np.array([[adapted_features.get(f, 0.0) for f in feat_names]])

        print(f"\n  Adapted feature vector: {X[0]}")

        # Impute and scale
        X_imp = self.imputer.transform(X)
        X_sc = self.scaler.transform(X_imp)

        # Predict
        prediction = self.model.predict(X_sc)[0]

        # Get probability/confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_sc)[0]
            confidence = proba[prediction] * 100
        elif hasattr(self.model, "decision_function"):
            dec = self.model.decision_function(X_sc)[0]
            confidence = min(abs(dec) * 30 + 50, 99)  # rough mapping
        else:
            confidence = 75.0

        model_name = type(self.model).__name__
        print(f"\n  Prediction: {prediction} ({'Dyslexic' if prediction == 1 else 'Non-Dyslexic'})")
        print(f"  Confidence: {confidence:.1f}%")

        # Show results screen (show RAW features so user sees real numbers)
        while True:
            img = self.renderer.draw_results(prediction, confidence, features, model_name)
            cv2.imshow("Dyslexia Screener", img)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:   # ESC
                break
            if key == ord('r') or key == ord('R'):   # Re-run
                self.run()
                return


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DYSLEXIA SCREENER v1.0")
    print("  Webcam-Based Eye Tracking with MediaPipe")
    print("=" * 60)

    screener = DyslexiaScreener()
    screener.run()

    print("\n[DONE] Screener session ended.")
