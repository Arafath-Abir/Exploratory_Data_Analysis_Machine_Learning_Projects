# Exploratory Data Analysis & Machine Learning Projects

This repository contains a collection of **data science mini projects** focusing on exploratory data analysis (EDA), visualization, and classic machine learning algorithms.  
It demonstrates how to work with real datasets, generate insights, and apply supervised ML models such as classification and regression.

---

## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA)**
  - Data cleaning and preprocessing
  - Descriptive statistics and distributions
  - Correlation analysis

- 📈 **Data Visualization**
  - Histograms, scatter plots, boxplots
  - 2D and 3D visualization of datasets (e.g., Iris dataset)

- 🤖 **Machine Learning**
  - Classification with Logistic Regression, KNN, and Decision Trees
  - Train/test split and performance evaluation
  - Metrics: Accuracy, Precision, Recall, Confusion Matrix

- 📂 **Multiple Datasets**
  - Mortality / health-related data (EDA focus)
  - Iris dataset (ML classification demo)

---

## 📂 Project Structure

```
.
├── notebooks/
│   └── DMML_final_projects.ipynb   # Main notebook with EDA & ML demos
├── src/
│   └── ml_projects.py              # Clean Python script version (optional)
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Technologies Used

- **Python 3**
- Libraries:
  - `pandas` – data handling
  - `numpy` – numerical computations
  - `matplotlib`, `seaborn` – data visualization
  - `scikit-learn` – machine learning algorithms & metrics

---

---

## 📸 Example Usage

**EDA Example**
```python
import pandas as pd
import seaborn as sns

df = pd.read_csv("data.csv")
print(df.head())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)
```

**ML Example (KNN Classifier on Iris Dataset)**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```


