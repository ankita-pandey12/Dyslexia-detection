# Dyslexia Detection and Screening Pipeline

This repository contains machine learning models and a real-time webcam screener for detecting dyslexia based on the ETDD70 Eye-Tracking dataset.

## Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate
```

### 3. Install Required Libraries
Install the necessary dependencies using pip:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn opencv-python mediapipe
```

## How to Run

### Real-Time Webcam Screener
To launch the real-time webcam dyslexia screener (which uses MediaPipe for eye tracking):
```bash
python dyslexia_screener.py
```

### Model Training & Tuning
To reproduce the model training or hyperparameter tuning stages using the ETDD70 dataset:
- **Final ML Training:** 
  ```bash
  python etdd70_final_training.py
  ```
- **Hyperparameter Tuning:**
  ```bash
  python etdd70_hypertuning.py
  ```

---

# ETDD70 Dataset Details (Original Reference)


## Description
This repository contains the ETDD70 (Eye-Tracking Dyslexia Dataset), a novel dataset collected to facilitate research in AI-based classification of dyslexia using eye-tracking data. The dataset comprises eye-tracking recordings from 70 Czech participants (35 dyslexic and 35 non-dyslexic) aged 9-10 years, captured during three text-reading tasks in the Czech language: syllables reading, meaningful-text reading, and pseudo-text reading.

## Key features of the dataset:

1. High-frequency (250 Hz) time series of 2D eye movement positions
2. Derived characteristics extracted from eye movement patterns
3. Data from 70 Czech participants (35 dyslexic, 35 non-dyslexic)
4. Three distinct reading tasks in Czech (syllables reading, natural reading, pseudo-text reading)

## The repository includes:

1. Raw eye-tracking data
	- *_raw.csv 
	
2. Processed data files:

	- *_fixations.csv: Fixation data detected using the i2mc algorithm
	- *_saccades.csv: Saccade data derived from consecutive fixations
	- *_metrics.csv: Derived statistical characteristics (see below for more information)
 
3. Stimuli data files:
	- *_rois.csv: Region of Interests (ROI): word and line boundaries
	- *_.jpg: Images of Stimuli

4. Dyslexia Label:
	- dyslexia_class_label.csv

5. Visual-based representations:
	- fixation_images.zip

## Characteristics provided:

	- Whole-task characteristics (e.g., number of fixations, saccades, regressions)
	- Region-of-interest (ROI) characteristics at the word level (e.g., dwell time, number of revisits)

This dataset supports research in AI-based methods for dyslexia classification, with a focus on Czech-speaking children. It has potential applications in improving diagnostic accuracy and early intervention strategies for dyslexia in Czech-language contexts. The best model trained on this data achieved an accuracy of around 90% in distinguishing between dyslexic and non-dyslexic individuals.

For detailed information on the dataset, data collection process, and derived characteristics, please refer to the accompanying paper: "ETDD70: Eye-Tracking Dataset for Classification of Dyslexia using AI-based Methods"

## Citations:

If you use this dataset, please cite the following:

Dataset: Dostalova, N., Svaricek, R., Sedmidubsky, J., Culemann, W., Sasinka, C., Zezula, P., & Cenek, J. (2024). ETDD70: Eye-tracking dyslexia dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13332134

Associated Paper: Sedmidubsky, J., Dostalova, N., Svaricek, R., & Culemann, W. (2024). ETDD70: Eye-tracking dataset for classification of dyslexia using AI-based methods. In Proceedings of the 17th International Conference on Similarity Search and Applications (SISAP) (pp. 1-14). Springer.

### Note:
All data has been collected with proper informed consent and in compliance with ethical guidelines for research involving minors. The dataset is specific to Czech-speaking children and may have limitations when applied to other languages or populations.