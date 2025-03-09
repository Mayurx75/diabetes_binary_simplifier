### **📌 README.md for Diabetes Binary Classifier**
``markdown
# 🏥 Diabetes Binary Classifier

A machine learning model that predicts whether a person has diabetes based on medical attributes. The model is trained using the **Pima Indians Diabetes Dataset** and supports multiple classifiers like Logistic Regression, Decision Tree, Random Forest, and SVM.

## 📖 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Results](#-results)
- [Example Prediction](#-example-prediction)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 Overview
This project aims to build a **binary classifier** for predicting diabetes using **Scikit-learn**. The dataset is preprocessed by handling missing values and scaling the features. Various machine learning models are trained and evaluated, and the best model is selected using hyperparameter tuning.

## 📊 Dataset
The dataset used is the **Pima Indians Diabetes Dataset**, which consists of:
- **768 samples** with **8 medical features** (e.g., glucose levels, BMI, age).
- **Outcome variable**: `0` (Non-diabetic) or `1` (Diabetic).
- Downloaded from: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

---

## ⚙️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/diabetes-binary-classifier.git
cd diabetes-binary-classifier
```

### **2️⃣ Install Required Libraries**
Ensure you have Python installed, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### **1️⃣ Train the Model**
Run the Python script to train and evaluate the model:

```bash
python diabetes_binary_classifier.py
```

This will:
- Load and preprocess the dataset.
- Train multiple machine learning models.
- Evaluate their accuracy and ROC AUC scores.
- Tune hyperparameters for the best model.
- Save the trained model as `diabetes_classifier.pkl` for future predictions.

### **2️⃣ Make Predictions**
To predict diabetes for a new patient, modify the script to pass a sample input:

```python
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("diabetes_classifier.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# Sample patient data: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
new_patient = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], 
                           columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Scale the input and predict
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)[0]
probability = model.predict_proba(new_patient_scaled)[0][1]

print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
print(f"Probability: {probability:.2f}")
```

---

## 📊 Model Training & Evaluation

- Models used:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Support Vector Machine (SVM)**
- Performance Metrics:
  - **Accuracy**
  - **Confusion Matrix**
  - **ROC-AUC Curve**
- Feature Importance is visualized using **Random Forest**.

---

## 📈 Results
The model with the best performance is selected based on **ROC-AUC scores**.

- Example **Accuracy Scores**:
  - Logistic Regression: **78.4%**
  - Decision Tree: **75.1%**
  - Random Forest: **82.6%**
  - SVM: **79.2%**

- Example **Feature Importance (Random Forest)**:
  ```
  Glucose: 0.29
  BMI: 0.21
  Age: 0.15
  Diabetes Pedigree Function: 0.10
  ```

- Example **ROC-AUC Curve**:
  ![ROC Curve](feature_importance.png)

---

## 🔍 Example Prediction
If you provide the following patient data:

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age |
|-------------|---------|--------------|--------------|---------|------|-------------------------|-----|
| 6           | 148     | 72           | 35           | 0       | 33.6 | 0.627                   | 50  |

The model might predict:
```
Prediction: Diabetic
Probability: 0.78
```

---

## 🤝 Contributing
Contributions are welcome! Follow these steps:
1. **Fork** the repository.
2. **Create a new branch**: `git checkout -b feature-branch`
3. **Commit your changes**: `git commit -m "Added new feature"`
4. **Push to your branch**: `git push origin feature-branch`
5. **Create a Pull Request** 🚀

---

## 📜 License
This project is licensed under the **MIT License**.

## 🔗 Acknowledgement
I would like to express my gratitude to **Dr Victor Agughasi Ikechukwu** for the guidance and mentorship throughout this project. You can check out their GitHub profile here:  
👉 **[Sir's GitHub](https://github.com/Victor-Ikechukwu)**

