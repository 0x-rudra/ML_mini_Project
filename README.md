# 📧 SMS / Email Spam Detector (ML Mini-Project)

> **Note:** This repository is an educational and beginner-friendly **learning project** developed to practice and understand fundamental Machine Learning and Natural Language Processing (NLP) concepts using Python.

---

## 📌 Project Overview

This project implements a text classification pipeline to detect whether a given message is **Spam** or **Ham (Not Spam)**. It uses TF-IDF vectorization for text feature extraction and a Multinomial Naive Bayes classifier for prediction.

### 🎯 Key Learning Objectives
- Understanding the end-to-end Machine Learning workflow: data loading, train-test splitting, model training, evaluation, and inference.
- Implementing Natural Language Processing (NLP) basics, specifically text vectorization with **TF-IDF (Term Frequency - Inverse Document Frequency)** and stop-word removal.
- Using scikit-learn's `Pipeline` abstraction to chain feature extractors and estimators cleanly.
- Evaluating classification performance using Precision, Recall, F1-Score, and overall Accuracy.
- Creating an interactive CLI loop to test predictions and probabilities in real time.



## ⚙️ How It Works

1. **Data Preprocessing & Splitting**:
   - Reads `spam_dataset.csv` with `Category` (target) and `Message` (input text).
   - Splits data into 65% training and 35% testing sets using `train_test_split`.
2. **Model Pipeline**:
   - **`TfidfVectorizer(stop_words='english')`**: Converts raw text messages into numerical TF-IDF feature matrices while ignoring common English stop words.
   - **`MultinomialNB()`**: A probabilistic Naive Bayes classifier suitable for discrete features such as word counts and TF-IDF weights.
3. **Evaluation**:
   - Generates a classification report evaluating precision, recall, and f1-score.
4. **Interactive Prediction**:
   - A command-line input prompt allowing you to input arbitrary text messages to view the predicted class and corresponding probabilities.

---

## 🚀 Usage & Dataset Path Setup

When running the notebook (`Spam_Detector.ipynb`), check **Cell 2** and set the dataset path based on where you are running the project:

- **If using Google Colab:**
  Upload `spam_dataset.csv` to the `/content/sample_data/` directory (via the Files panel in the left sidebar):
  ```python
  data = pd.read_csv('/content/sample_data/spam_dataset.csv')
  ```

- **If running locally (VS Code / Jupyter):**
  ```python
  data = pd.read_csv('spam_dataset.csv')
  ```

---

## 📊 Model Performance

Tested on a 35% test split (~1,951 samples):

| Metric | Ham (Not Spam) | Spam | Overall / Weighted Avg |
|---|---|---|---|
| **Precision** | 0.97 | 0.99 | 0.97 |
| **Recall** | 1.00 | 0.77 | 0.97 |
| **F1-Score** | 0.98 | 0.87 | 0.97 |
| **Accuracy** | — | — | **97%** |

---

## 🔮 Future Improvements / Learning Ideas
- Experiment with other classifiers (e.g., Logistic Regression, Support Vector Machines, or Random Forest).
- Apply text preprocessing techniques (lemmatization, stemming, regex cleaning for URLs and phone numbers).
- Address class imbalance to improve Spam recall.
- Deploy as a lightweight web app using Streamlit or Flask.
