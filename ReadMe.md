
# Wind Turbine Blade Icing Predictive Warning System

## Overview
This repository implements a **Conditional Time Series Denoising Diffusion (CTSDD)** model and a **composite score** algorithm wind turbine blade icing predictive fault warning system. The system restores masked monitoring data, computes anomaly scores, and triggers predictive warnings based on learned normal patterns.

---

## Repository Structure
### Code & Configuration
.
├── config/                   # Configuration files
│   └── WT_ice.yaml           # Hyperparameters for model training and masking
├── models/                   # Model definitions and pretrained weights
│   ├── CTSDD.py              # CTSDD model architecture
│   └── CTSDD_pretrained_model.pth  # Pretrained model weights
├── dataset_wind_ice.py       # Dataset loader and preprocessing
├── exe_ice_warning.py        # Main script for fault warning
├── main_diff_model.py        # Diffusion model structure
├── plot_results.py           # Visualization utilities
├── utils_prediction.py       # CTSDD pipeline functions & Warning composite score function
└── results_view.ipynb        # Jupyter notebook for results visualization

### Datast
.
├── dataset/                  # Wind turbine monitoring datasets for testing
│   ├── normal/               # Normal operation data
│   │   ├── normal.csv         # Raw monitoring data
│   │   └── normal_mask0.3.csv # Masked data (30% random masking)
│   ├── case1/                # Icing fault case 1 data
│   │   ├── case1.csv         # Raw monitoring data
│   │   └── case1_mask0.3.csv # Masked data (30% random masking)
│   ├── case2/                # Icing fault case 2 data
│   │   ├── case2.csv         # Raw monitoring data
│   │   └── case2_mask0.3.csv # Masked data (30% random masking)
│   ├── normal_Robust_threshold.csv # Parameters related to the calculation of the composite score
│   └── data_mean_std.pk  # Preprocessed data statistics

### Results Save
.
├── save/                     
│   └── test_results_examples/ # Example of experimental results
│       ├── normal_test/       # Normal restoration outputs
│       │   ├── generated_outputs_nsample5.pk  # 5-time restoration samples
│       │   ├── scores.csv    # Composite scores
│       │   └── pictures/     # Visualization of WP restoration
│       ├── case1_test/       # case1 restoration outputs
│       │   ├── generated_outputs_nsample5.pk  # 5-time restoration samples
│       │   ├── scores.csv    # Composite scores
│       │   ├── scores_interval5.csv  # Composite scores warning effect
│       │   └── pictures/     # Visualization of WP restoration
│       ├── case2_test/       # case2 restoration outputs
│       │   ├── generated_outputs_nsample5.pk  # 5-time restoration samples
│       │   ├── scores.csv    # Composite scores
│       │   ├── scores_interval5.csv  # Composite scores warning effect
│       │   └── pictures/     # Visualization of WP restoration
│   └── test_results/         # Results of running experiments

---

## Dataset Overview

The dataset used in this study is sourced from the 2017 China Industrial Big Data Competition and includes data related to WT blade icing. It records 26 types of monitoring data and their timestamps. To align with the progression of WT blade icing and retain essential temporal details for model training, the data are resampled to a 1-minute resolution.

We provide monitoring data collected over a specified period under three distinct scenarios:
*   Normal: No ice fault observed.
*   Case 1: Ice fault recorded on November 9, 2015, at 21:21.
*   Case 2: Ice fault recorded on December 3, 2015, at 17:10.

These three datasts are organized within the `dataset` folder for easy access and utilization. The data is provided in CSV format, facilitating straightforward loading and manipulation using various data processing libraries, such as Pandas or NumPy.

We also provide normalization-related parameters in the file `data_mean_std.pk`, along with parameters for calculating the composite score and warning threshold in `normal_Robust_threshold.csv`. These parameters were derived from extensive experiments conducted on the complete testing set.
<!-- The monitoring variables include Wind Speed, Generator Speed, Wind Power, Wind Direction and its Mean Value, Yaw Position, Yaw Speed, angles (Blade 1/2/3 Pitch Angle), speeds (Blade 1/2/3 Pitch Speed), motor temperatures (Blade 1/2/3 Motor Temperature), NG5 temperatures (Blade 1/2/3 NG5 Temperature), NG5 DC currents (Blade 1/2/3 NG5 DC Current) for blades 1, 2, and 3, Acceleration in X/Y-axis, Ambient Temperature, and Internal Temperature information.  -->


## Quick Start

### 1. Prerequisites
*   Python (>= 3.6)
*   PyTorch (>= 1.8.0)
*   NumPy
*   Pandas
*   SciPy
*   Scikit-learn
*   Matplotlib
*   Jupyter Notebook 
*   Pickle
*   Fastdtw


### 2. Data & Pretrained Weights Preparation
- Place raw and masked data in `dataset/(case_name)`
- Ensure the composite score-related parameters and warning thresholds file `normal_Robust_threshold.csv` in `dataset/`
- Ensure pretrained weights file `CTSDD_pretrained_model.pth` in `models/normal/`

### 3. Fault Warning
To run the `exe_ice_warning.py` script with a specific case, you can set the `case` parameter. The available options for the case parameter are:
`normal`, `case1`, `case2`
Other parameters can be customized in the script using its parameter parser, or by modifying the values in the config file `config/WT_ice.yaml`.

For example:
```bash
python exe_ice_warning.py --case case1
```
* Generates WP restoration results `generated_outputs_nsample5.pk`, composite scores `scores.csv` and triggers warnings (outputs in `save/test_results/case1_test/`)

### 4. Visualization
You can load and interactively analyze the experiment results using `results_view.ipynb` Jupyter Notebook.
(We have provided some visualizations of the experimental results. If you need to visualize new experiment results, please change the path to the experiment results folder in the code.)
<!-- * Generates plots of WP restoration and scores -->

