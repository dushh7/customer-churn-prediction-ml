import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# -----------------------------------
# CREATE IMAGES FOLDER
# -----------------------------------
if not os.path.exists("images"):
    os.makedirs("images")

# -----------------------------------
# LOAD DATASET
# -----------------------------------
df = pd.read_csv("data/dataset.csv")

# -----------------------------------
# DISPLAY FIRST 5 ROWS
# -----------------------------------
print("\nFirst 5 Rows:\n")
print(df.head())

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------
print("\nDataset Information:\n")
print(df.info())

# -----------------------------------
# REMOVE customerID COLUMN
# -----------------------------------
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# -----------------------------------
# CONVERT TotalCharges TO NUMERIC
# -----------------------------------
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# -----------------------------------
# HANDLE MISSING VALUES
# -----------------------------------
df['TotalCharges'].fillna(
    df['TotalCharges'].median(),
    inplace=True
)

# -----------------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------------
categorical_columns = df.select_dtypes(
    include=['object']
).columns

label_encoder = LabelEncoder()

for column in categorical_columns:

    df[column] = label_encoder.fit_transform(
        df[column]
    )

# -----------------------------------
# DISPLAY DATA TYPES
# -----------------------------------
print("\nData Types After Encoding:\n")
print(df.dtypes)

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------
X = df.drop('Churn', axis=1)

y = df['Churn']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# RANDOM FOREST MODEL
# -----------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
model.fit(X_train, y_train)

# -----------------------------------
# MAKE PREDICTIONS
# -----------------------------------
predictions = model.predict(X_test)

# -----------------------------------
# ACCURACY SCORE
# -----------------------------------
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy Score:")
print(accuracy)

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

# SAVE IMAGE
plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

# -----------------------------------
# ROC CURVE
# -----------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

# SAVE IMAGE
plt.savefig(
    "images/roc_curve.png"
)

plt.show()

# -----------------------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

# SAVE IMAGE
plt.savefig(
    "images/feature_importance.png"
)

plt.show()

print("\nImages saved successfully in images folder.")