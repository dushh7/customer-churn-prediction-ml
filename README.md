# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn prediction is one of the most common and important business applications of Machine Learning. In this project, a machine learning model is built to predict whether a customer is likely to leave a telecom service based on customer-related information and usage patterns.

The project uses the Telco Customer Churn dataset and applies the Random Forest Classifier algorithm to analyze customer behavior and generate predictions. The workflow includes data preprocessing, feature encoding, model training, evaluation, and visualization.

This project was developed to gain practical experience in supervised machine learning, classification models, and data visualization techniques.

---

# Objectives

- Understand the complete machine learning workflow
- Perform data preprocessing and cleaning
- Handle categorical and numerical data
- Train a classification model
- Predict customer churn
- Evaluate model performance using metrics
- Visualize results using graphs and charts
- Gain practical exposure to GitHub project management

---

# Dataset Information

## Dataset Used

Telco Customer Churn Dataset

The dataset contains customer-related information such as:

- Gender
- Senior citizen status
- Partner and dependent details
- Internet service information
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

### Target Column

```text
Churn
```

The target column contains:
- Yes → Customer leaves the service
- No → Customer stays with the service

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code
- Git
- GitHub

---

# Machine Learning Workflow

The following steps were performed during the project development:

1. Data Collection
2. Data Cleaning
3. Handling Missing Values
4. Encoding Categorical Data
5. Feature Selection
6. Train-Test Split
7. Model Training
8. Prediction
9. Performance Evaluation
10. Data Visualization

---

# Machine Learning Algorithm Used

## Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Why Random Forest?

- Handles both categorical and numerical data effectively
- Provides better prediction accuracy
- Works well for classification problems
- Reduces overfitting
- Provides feature importance analysis

---

# Data Preprocessing

Several preprocessing techniques were applied before training the model:

## 1. Removing Unnecessary Columns

The `customerID` column was removed because it does not contribute to prediction.

## 2. Handling Missing Values

Missing values in the `TotalCharges` column were identified and replaced using the median value.

## 3. Encoding Categorical Columns

Categorical text values such as:

```text
Male
Female
Yes
No
```

were converted into numerical format using Label Encoding.

---

# Model Training

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The Random Forest Classifier was trained using the training dataset and tested using unseen test data.

---

# Model Evaluation

The model performance was evaluated using the following metrics:

## Accuracy Score

Measures the percentage of correct predictions made by the model.

## Classification Report

Provides:
- Precision
- Recall
- F1-Score
- Support

## Confusion Matrix

Shows:
- Correct predictions
- Incorrect predictions
- False positives
- False negatives

## ROC Curve

Measures the classification performance of the model at different threshold values.

## Feature Importance Graph

Displays which features contribute most to churn prediction.

---

# Output Visualizations

The project automatically generates and saves the following graphs:

- Confusion Matrix
- ROC Curve
- Feature Importance Graph

All images are stored inside the `images/` folder.

---

# Project Structure

```text
customer-churn-prediction-ml/
│
├── data/
│   └── dataset.csv
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction-ml.git
```

---

## Step 2 — Open the Project Folder

```bash
cd customer-churn-prediction-ml
```

---

## Step 3 — Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 4 — Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## Step 5 — Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Step 6 — Run the Project

```bash
python main.py
```

---

# Expected Output

After running the project successfully, the following outputs will be generated:

- Accuracy Score
- Classification Report
- Confusion Matrix
- ROC Curve
- Feature Importance Graph

---

# Sample Features Used for Prediction

Some important features used in the model include:

- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- PaymentMethod
- tenure

---

# Learning Outcomes

This project helped in gaining practical knowledge in:

- Supervised Machine Learning
- Data preprocessing techniques
- Feature encoding
- Classification algorithms
- Model training and testing
- Data visualization
- Performance evaluation
- Git and GitHub workflow

---

# Future Improvements

This project can be enhanced further by:

- Using multiple machine learning algorithms
- Hyperparameter tuning
- Cross-validation
- Building a Streamlit web application
- Deploying the model online
- Using advanced feature engineering techniques

---

# Conclusion

This project demonstrates how Machine Learning can be applied to solve real-world business problems such as customer churn prediction. By analyzing customer behavior and service usage patterns, the model helps predict whether a customer is likely to leave the service.

The project also provided hands-on experience in data preprocessing, model training, evaluation techniques, and GitHub project management.

---

# Author

## Dushyan S


```