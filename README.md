## 🔍 Objective

To develop a model capable of distinguishing between Higgs boson signal events and background noise in particle collision data.

---

## 🧰 Technologies & Libraries Used

- Python, NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow/Keras

---

## 📊 Dataset

- **Source**: [Kaggle Higgs Boson Dataset](https://www.kaggle.com/c/higgs-boson/data)
- **Description**: Contains 30 features and a binary target variable `signal`.

---

## 🧹 1. Data Preparation

- Handled missing values
- Checked class balance
- Normalized features using `StandardScaler`
- Split into training (70%), validation (15%), and test (15%) sets

---

## 🌲 2. Baseline Model (XGBoost)

- Trained an XGBoost classifier with:
  - 100 estimators
  - Max depth: 6
  - Learning rate: 0.1
- Validation accuracy: ~75-80% (baseline)

---

## 🤖 3. Neural Network Model

### Architecture:
- 3 hidden layers: 128 → 64 → 64
- ReLU activation
- Batch Normalization
- Dropout (0.3–0.4)
- L2 regularization
- Output layer: sigmoid activation

### Optimization:
- Loss: Binary Crossentropy
- Optimizers: Adam, RMSprop, AdamW
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## 📈 4. Training & Evaluation

### Metrics Used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

### Visualization:
- Training & validation loss/accuracy curves
- ROC curve
- Confusion matrix heatmap

---

## 🔁 5. Experimentation

- Deeper model with 4 layers and AdamW optimizer
- Increased dropout and L2 regularization
- Observed improved generalization with proper regularization

---

## 📌 Key Takeaways

- Neural networks can outperform XGBoost with the right architecture and regularization
- Dropout, BatchNorm, and EarlyStopping mitigate overfitting
- AdamW provided better generalization than Adam
- Model performance can be further enhanced with tuning and ensembles

---

## 💡 Future Work

- Hyperparameter tuning (e.g. with Optuna or Keras Tuner)
- Trying alternative architectures (e.g. ResNet-like blocks)
- Ensemble of models (NN + XGBoost)
- Use advanced activations (Swish, LeakyReLU)
- Build a Streamlit app for live prediction

Contact @ Touseefahmed00710@gmail.com
