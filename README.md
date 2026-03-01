# Sentiment Analysis of Financial News Headlines

A machine learning project that classifies financial news headlines into **positive**, **neutral**, and **negative** sentiments using Support Vector Machines (SVM) and Logistic Regression with TF-IDF features.

---

## Project Overview

Financial sentiment analysis is a valuable tool for understanding market tone and investor behavior. This project builds and evaluates text classification models on a labeled dataset of financial news headlines, with a strong focus on handling class imbalance through optimization techniques. By automating sentiment detection, stakeholders can more efficiently monitor market events, anticipate shifts in investor sentiment, and support decision-making in areas like algorithmic trading, risk assessment, and financial forecasting.

---

## Dataset

- **Title:** Sentiment Analysis for Financial News
- **Size:** 4,846 headlines (reduced to 4,831 after duplicate removal)
- **Labels:** `positive`, `neutral`, `negative`
- **Class Distribution:** Neutral 59.4% · Positive 28.1% · Negative 12.5%
- **Format:** CSV with two columns — `Label` and `Text`
- **Source:** Malo et al. (2014) — *Good debt or bad debt: Detecting semantic orientations in economic texts*

> Note: The dataset is not included in this repository. Place your CSV file in the same directory and upload it when prompted in the notebook.

---

## Project Pipeline

### 1. Data Exploration
- Inspected class distribution via value counts and pie chart visualization
- Reviewed sample headlines per sentiment class
- Confirmed **no missing values** in either the Text or Label columns
- Identified **8 duplicate text entries** before cleaning (15 after text normalization)

### 2. Data Preprocessing
- Lowercased all text
- Removed punctuation and English stopwords (NLTK)
- Dropped 15 duplicate rows, reducing dataset from 4,846 → 4,831 entries
- Encoded labels numerically using `LabelEncoder`:
  - 0 = Negative · 1 = Neutral · 2 = Positive

### 3. Feature Extraction & Train/Test Split
- Applied **TF-IDF Vectorization** with a maximum of 5,000 features
- 80/20 train-test split (random_state=42)
  - Training set: 3,864 samples
  - Test set: 967 samples

### 4. Model Training (Baseline)
- **Support Vector Machine (SVM)** — linear kernel with probability enabled for AUC
- **Logistic Regression (LR)** — max_iter=1000 for convergence

### 5. Optimization
Two techniques were applied and compared across both models:

| Technique | Description |
|---|---|
| **Class Weighting** | `class_weight='balanced'` — adjusts weights inversely proportional to class frequencies |
| **SMOTE** | Synthetic Minority Oversampling applied to training data only; test set left unchanged |

### 6. Evaluation Metrics
- Training and test accuracy
- Classification report (precision, recall, F1-score per class)
- Confusion matrix (visualized)
- Multi-class AUC score (One-vs-Rest strategy)

---

## Results

| Model | Optimization | Train Accuracy | Test Accuracy | AUC |
|---|---|---|---|---|
| SVM | None | 89.8% | 73.9% | 0.850 |
| Logistic Regression | None | 85.3% | 73.0% | 0.853 |
| **SVM** | **Class Weights** | **94.7%** | **74.7%** | **0.865** |
| Logistic Regression | Class Weights | 93.5% | 72.5% | 0.856 |
| SVM | SMOTE | 96.6% | 74.6% | 0.856 |
| Logistic Regression | SMOTE | 95.2% | 73.1% | 0.856 |

### Key Finding
The **SVM with class weighting** achieved the best overall balance — highest AUC (0.865), strong test accuracy (74.7%), and noticeably improved recall and F1-scores for the minority negative and positive classes, without the overfitting risk introduced by SMOTE.

### Class Imbalance Insight
Without optimization, both models dominated predictions toward the majority `neutral` class (recall ~0.90–0.94), while struggling on `negative` (recall as low as 0.40). Class weighting and SMOTE both improved minority class recognition. However, SMOTE's large gap between training and test accuracy (e.g., 96.6% vs 74.6%) signals potential overfitting on synthetic samples.

---

## Balanced Test Set Evaluation

To further validate the best model, SVM with class weights was evaluated on a **balanced test subset** (116 samples per class):

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.87 | 0.58 | 0.69 |
| Neutral | 0.61 | 0.86 | 0.71 |
| Positive | 0.69 | 0.63 | 0.66 |
| **Overall Accuracy** | | | **69%** |

This confirms that reported metrics on the imbalanced test set underestimated the model's actual ability to distinguish between classes.

---

## Technologies Used

- **Python 3**
- **Pandas & NumPy** — data manipulation
- **NLTK** — text preprocessing and stopword removal
- **Scikit-learn** — TF-IDF, SVM, Logistic Regression, evaluation metrics
- **Imbalanced-learn** — SMOTE oversampling
- **Matplotlib** — visualization (pie charts, confusion matrices)
- **Google Colab** — development environment

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter
2. Install any missing dependencies:
   ```bash
   pip install nltk scikit-learn imbalanced-learn pandas matplotlib
   ```
3. Run all cells in order
4. When prompted, upload your CSV dataset file (two columns: `Label`, `Text`, no header)

---

## Key Learnings

- TF-IDF is a strong baseline for text classification even without deep learning
- Class imbalance (59% neutral vs 12.5% negative) significantly biases baseline models toward the majority class
- **Class weighting** improved minority class recall without altering the data distribution, making it more robust than SMOTE for this dataset
- High AUC scores (0.85+) alongside moderate accuracy reveal that accuracy alone is a misleading metric in imbalanced classification tasks
- Evaluating on a balanced test set exposed performance that the original imbalanced test set was masking
