# Dyslexia Screener — Complete Pipeline Flow

## High-Level Architecture

```mermaid
flowchart TD
    A["Phase 0: Instructions"] --> B["Phase 1: 9-Point Calibration"]
    B --> C["Phase 2: Task T1 - Syllables"]
    C --> D["Phase 3: Task T4 - Meaningful Text"]
    D --> E["Phase 4: Task T5 - Pseudo-Text"]
    E --> F["Phase 5: Domain Adaptation"]
    F --> G["Phase 6: Model Prediction"]
    G --> H["Results Screen"]
    
    style A fill:#2d2d2d,color:#fff
    style B fill:#1a5276,color:#fff
    style C fill:#1e8449,color:#fff
    style D fill:#1e8449,color:#fff
    style E fill:#1e8449,color:#fff
    style F fill:#7d3c98,color:#fff
    style G fill:#c0392b,color:#fff
    style H fill:#f39c12,color:#000
```

---

## Detailed Step-by-Step Flow

### Phase 0 — Startup & Model Loading
```
User runs: python dyslexia_screener.py
    |
    v
DyslexiaScreener.__init__()
    |
    ├── Opens webcam (cv2.VideoCapture)
    ├── Loads tuned_logistic_regression.joblib  <-- The trained ML model
    ├── Loads tuned_scaler.joblib               <-- StandardScaler (z-score normalizer)
    ├── Loads tuned_imputer.joblib              <-- Fills missing values
    └── Loads final_feature_config.joblib       <-- List of 21 feature names in exact order
```

### Phase 1 — 9-Point Calibration

> [!IMPORTANT]
> This phase creates the mathematical bridge between "where your iris is pointing" and "where on the screen you're looking."

```mermaid
flowchart LR
    A["Webcam frame"] --> B["MediaPipe FaceMesh\n478 landmarks"]
    B --> C["Extract iris center\nlandmarks 468, 473"]
    C --> D["Compute iris ratio\nwithin eye opening"]
    D --> E["Record ratio at\nknown screen point"]
    E --> F["After 9 points:\nFit 2nd-order polynomial"]
    
    style B fill:#1a5276,color:#fff
    style F fill:#7d3c98,color:#fff
```

**What happens per calibration point:**
1. A yellow dot appears at a known screen position (e.g., `(153, 86)`)
2. The webcam captures your face at 30fps for 2 seconds
3. MediaPipe detects 478 facial landmarks, including iris centers (indices 468 & 473)
4. For each frame, the code computes:
   - `iris_ratio_x = (iris_center_x - eye_corner_left) / eye_width` → value between 0 and 1
   - `iris_ratio_y = (iris_center_y - eye_top) / eye_height` → value between 0 and 1
5. These ratios are averaged across both eyes and across all frames
6. The result is stored: *"When user looks at screen pixel (153, 86), their iris ratio is (0.52, 0.47)"*
7. After all 9 points, a **2nd-degree polynomial** is fitted:
   - `screen_x = a·rx² + b·ry² + c·rx·ry + d·rx + e·ry + f`
   - `screen_y = ...` (same form, different coefficients)

---

### Phases 2-4 — Reading Tasks (The Core Data Collection)

```mermaid
flowchart TD
    A["Webcam Frame\n~30fps"] --> B["MediaPipe\nIris Detection"]
    B --> C["Iris Ratio\n0.0 to 1.0"]
    C --> D["Calibration Polynomial\nratio → screen pixels"]
    D --> E["Smoothed Gaze Point\nx, y, timestamp"]
    E --> F{"I-DT Algorithm:\nIs dispersion < 30px?"}
    F -->|"Yes: Eye is still"| G["Keep adding to\ncurrent window"]
    F -->|"No: Eye just jumped"| H{"Duration > 80ms?"}
    H -->|"Yes"| I["Register as FIXATION\nRecord: centroid, duration"]
    H -->|"No"| J["Discard as noise"]
    I --> K["Create SACCADE\nfrom previous fixation"]
    
    style F fill:#c0392b,color:#fff
    style I fill:#1e8449,color:#fff
    style J fill:#7f8c8d,color:#fff
```

**Per-frame loop (runs ~30 times per second):**
1. Read webcam frame
2. MediaPipe detects iris position → compute iris ratio
3. Polynomial maps ratio → estimated screen (x, y) pixel
4. Apply moving average smoothing (window=15 frames)
5. Feed smoothed point into **I-DT Fixation Detector**

**I-DT (Identification by Dispersion Threshold) algorithm:**
- Maintains a growing "window" of recent gaze points
- Computes dispersion = `(max_x - min_x) + (max_y - min_y)`
- If dispersion ≤ 30px → "eye is still here, keep watching"
- If dispersion > 30px → "eye just jumped away!"
  - Check if the accumulated window lasted ≥ 80ms
  - If yes → **FIXATION** recorded (centroid x,y + start/end timestamps)
  - If no → discarded as noise/jitter
- A **SACCADE** is created between every consecutive pair of fixations

**After user presses SPACE (done reading):**
- `finalize()` captures any remaining in-progress fixation
- Feature extraction begins for that task

---

### Feature Extraction (Per Task)

From the list of fixations and saccades, 7 features are computed:

| Feature | How It's Computed | What It Measures |
|---------|-------------------|------------------|
| `fix_count` | `len(fixations)` | Reading effort |
| `fix_dur_mean` | `mean(all fixation durations)` | Processing difficulty |
| `fix_dur_sd` | `std(all fixation durations)` | Reading erraticness |
| `fix_dur_median` | `median(all fixation durations)` | Typical fixation length |
| `total_read_time` | `last_fixation.end - first_fixation.start` | Overall reading speed |
| `gaze_linearity` | `mean(abs(y_diff between consecutive fixations))` | Vertical jumping |
| `revisit_count` | Count of fixations landing on a previously-visited line ROI | Re-reading behavior |

This produces **7 features × 3 tasks = 21 features total**.

---

### Phase 5 — Domain Adaptation

> [!WARNING]
> This is the critical bridge between webcam and lab data. Without it, every user appears "dyslexic."

```mermaid
flowchart LR
    A["Raw Webcam Features\nfix_count = 41\ngaze_linearity = 79"] --> B["DomainAdapter.adapt()"]
    B --> C["Adapted Features\nfix_count = 160\ngaze_linearity = 4.14"]
    C --> D["StandardScaler\nz-score normalization"]
    D --> E["Scaled Feature Vector\nready for model"]
    
    style B fill:#7d3c98,color:#fff
    style D fill:#1a5276,color:#fff
```

**Why it's needed:**
- Lab tracker (250Hz) detects ~155 fixations. Webcam (30fps) detects ~41.
- Lab gaze_linearity values are 3-5 pixels. Webcam values are 70-300 pixels.
- The `StandardScaler` was fitted on lab data. Feeding raw webcam values produces extreme z-scores.

**How it works:**
For each feature:
1. Define `webcam_typical` = what a normal webcam reader produces (e.g., 35 fixations)
2. Compute `deviation = (actual - typical) / spread`
3. Map: `adapted = train_nondys_mean + deviation × (train_dys_mean - train_nondys_mean)`
4. Special case: `gaze_linearity` is inverted (lower = more dyslexic in training data)

**Example from your run:**
```
t1_fix_count:      41 (webcam) → 160 (adapted)   // Within non-dyslexic range of 155
t1_gaze_linearity: 79 (webcam) → 4.14 (adapted)  // Within non-dyslexic range of 4.5
t1_revisit_count:  22 (webcam) → 35 (adapted)     // Slightly toward dyslexic range of 36
```

---

### Phase 6 — Model Prediction

```mermaid
flowchart LR
    A["21 Adapted Features"] --> B["SimpleImputer\nfill any NaN with median"]
    B --> C["StandardScaler\nz-score: (x - mean) / std"]
    C --> D["LogisticRegression.predict()\nL1 regularized"]
    D --> E{"Decision"}
    E -->|"Class 0"| F["LOWER RISK\nNon-Dyslexic"]
    E -->|"Class 1"| G["HIGHER RISK\nDyslexic"]
    
    style D fill:#c0392b,color:#fff
    style F fill:#1e8449,color:#fff
    style G fill:#e74c3c,color:#fff
```

**The LR model only uses 5 features (the rest are zeroed out by L1 regularization):**

| Feature | Coefficient | Direction |
|---------|:-----------:|-----------|
| `t4_gaze_linearity` | -0.81 | Lower = more dyslexic |
| `t4_revisit_count` | +0.75 | Higher = more dyslexic |
| `t5_fix_dur_median` | +0.43 | Longer fixations = more dyslexic |
| `t1_fix_dur_sd` | +0.26 | More erratic = more dyslexic |
| `t1_gaze_linearity` | -0.20 | Lower = more dyslexic |

**Decision:** `predict_proba()` returns probability for each class. The class with higher probability wins. Your run: `P(Non-Dyslexic) = 0.92`, so prediction = **Non-Dyslexic at 92% confidence**.

---

## Complete File Dependency Map

```mermaid
flowchart TD
    A["dyslexia_screener.py\nThe main application"] --> B["tuned_logistic_regression.joblib\nTrained LR model"]
    A --> C["tuned_scaler.joblib\nStandardScaler fitted on ETDD70"]
    A --> D["tuned_imputer.joblib\nMedian imputer fitted on ETDD70"]
    A --> E["final_feature_config.joblib\n21 feature names in order"]
    
    F["etdd70_final_training.py"] -->|"created"| G["etdd70_final_21feat.csv\n70 rows × 23 cols"]
    G -->|"fed into"| H["etdd70_hypertuning.py"]
    H -->|"produced"| B
    H -->|"produced"| C
    H -->|"produced"| D
    F -->|"produced"| E
    
    style A fill:#e74c3c,color:#fff
    style B fill:#3498db,color:#fff
    style C fill:#3498db,color:#fff
    style D fill:#3498db,color:#fff
    style E fill:#3498db,color:#fff
    style H fill:#27ae60,color:#fff
```
