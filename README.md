# Predictive Maintenance Failure Detection

A machine learning project that predicts whether a machine is likely to fail based on operational and sensor-related inputs. The final model is deployed as an interactive Streamlit app.

## Overview

This project tackles predictive maintenance using supervised machine learning on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). The goal is to classify machines at risk of failure early before a breakdown occurs using sensor readings and machine metadata.

Because failure events are rare (~3.4% of records), this is an **imbalanced classification problem**. Evaluation therefore prioritises **recall** and **F1-score** on the failure class over raw accuracy.

## Problem Statement

Unexpected machine failures cause costly downtime, emergency repairs, and production disruptions. By predicting failure risk in advance, maintenance teams can act proactively and schedule interventions at the right time.

## Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository — AI4I 2020 |
| Records | 10,000 |
| Features used | Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Product type |
| Target | `Machine failure` (binary: 0 = no failure, 1 = failure) |
| Class balance | ~96.6% no-failure / ~3.4% failure |

## Model Comparison

Three pipeline models (preprocessing → classifier) were evaluated on a 20% held-out test set:

| Model | Precision (fail) | Recall (fail) | F1 (fail) | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.67 | 0.26 | 0.38 | 97% |
| Random Forest | 0.85 | 0.56 | 0.67 | 98% |
| **Gradient Boosting** | **0.83** | **0.57** | **0.68** | **98%** |

**Gradient Boosting** was selected as the final model for its best balance of precision and recall on the minority failure class.

## Features

- **Product Quality Type**: L (Low), M (Medium), H (High)
- **Air Temperature (K)**: ambient air temperature
- **Process Temperature (K)**: temperature during the machining process
- **Rotational Speed (rpm)**: spindle speed
- **Torque (Nm)**: torque applied during operation
- **Tool Wear (min)**:  cumulative tool usage time

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data loading & manipulation |
| scikit-learn | Preprocessing, modelling, evaluation |
| Gradient Boosting | Final classifier |
| joblib | Model serialisation |
| matplotlib | EDA visualisations |
| Streamlit | Interactive web app |

## Project Structure

```text
riskpred/
├── README.md
├── requirements.txt
├── app/
│   └── app.py                  # Streamlit web application
├── data/
│   └── ai4i2020.csv            # Raw dataset
└── notebook/
    ├── notebook.ipynb          # EDA, training & model comparison
    ├── final_model.joblib      # Serialised Gradient Boosting pipeline
    ├── air_temperature_vs_machine_failure.png
    ├── process_temperature_vs_machine_failure.png
    ├── rotational_speed_vs_machine_failure.png
    └── torque_vs_machine_failure.png
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/16A9DA/predictive-maintenance-failure-detection.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/app.py
```

The app will open in your browser. Adjust the sliders and dropdowns to reflect a machine's current readings, then click **Predict** to see the failure risk.

## App Preview

The Streamlit app accepts the six input features via interactive sliders and a dropdown, then outputs:

- **No Failure Predicted** or **Machine Failure Predicted**
- Failure probability as a percentage
- A visual progress bar showing the probability level

## link:
https://predictive-maintenance-failure-detection.streamlit.app/