# House Price Prediction using ML & Jupyterlabs

## 📌 Project Overview
This project predicts house prices using a machine learning model trained on real estate data. The dataset contains various features such as location, number of rooms, population, and median income to help predict the median house value.

## 🛠️ Setup Instructions
### 1️⃣ Install Jupyter Lab
If you haven't already installed Jupyter Lab, use:
```bash
pip install jupyterlab
```

### 2️⃣ Clone the Repository and Start Jupyter Lab
```bash
git clone https://github.com/BarraHarrison/House-Price-Prediction.git
cd House-Price-Prediction
jupyter lab
```

### 3️⃣ Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 Data Analysis and Processing

### **1. Importing Required Libraries**
The project starts by importing key libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### **2. Loading the Dataset**
The dataset is loaded using:
```python
data = pd.read_csv("housing.csv")
```

### **3. Exploratory Data Analysis (EDA)**
- **Checking dataset structure:**
  ```python
  data.info()
  data.describe()
  ```
- **Checking for missing values:**
  ```python
  data.isnull().sum()
  ```
- **Visualizing relationships:**
  - Histograms
  - Correlation heatmaps
  - Scatter plots

### **4. Data Cleaning & Feature Engineering**
- Handling missing values by filling with median:
  ```python
  data.fillna(data.median(), inplace=True)
  ```
- Encoding categorical variables:
  ```python
  data = pd.get_dummies(data, columns=['ocean_proximity'])
  ```

### **5. Splitting the Data**
```python
from sklearn.model_selection import train_test_split

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 🤖 Model Training & Evaluation
### **6. Training a Random Forest Model**
```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
```

### **7. Evaluating the Model**
- **R² Score**
  ```python
  forest.score(X_test, y_test)
  ```
- **Hyperparameter Tuning Using GridSearchCV**
  ```python
  from sklearn.model_selection import GridSearchCV
  
  param_grid = {
      "n_estimators": [3, 10, 30],
      "max_features": [2, 4, 6, 8]
  }
  
  grid_search = GridSearchCV(forest, param_grid, cv=5,
                             scoring="neg_mean_squared_error",
                             return_train_score=True)
  grid_search.fit(X_train, y_train)
  ```

## 🚀 Future Improvements
- Test additional regression models (XGBoost, Gradient Boosting, etc.)
- Feature scaling for better performance
- Deploy model using Flask or FastAPI

## 🏆 Conclusion
This project demonstrates **data exploration, feature engineering, and machine learning modeling** using Jupyter Lab. The Random Forest model provides a strong baseline, and further optimizations can improve accuracy.

## 📚 What I Learned
During this project, I used several commands for the first time and found similarities with SQL operations used in server-side programming:

- **`data.dropna()`** → This removes missing values from the dataset, similar to `WHERE column IS NOT NULL` in SQL.
- **`train_test_split()`** → This splits the dataset into training and testing subsets, similar to using `LIMIT` and `OFFSET` in SQL queries to segment data.
- **`plt.figure()`** → This sets up the figure for visualization, which is akin to structuring query results before displaying them in web applications.
- **`sns.heatmap()`** → This visualizes correlations between variables, much like using SQL aggregate functions and `GROUP BY` to analyze relationships between different fields.
- **`pd.get_dummies()`** → This encodes categorical variables into a numerical format, similar to using `CASE WHEN` or `JOIN` operations in SQL to transform categorical data into structured numeric values.

This experience deepened my understanding of how data transformations in Python mirror SQL operations used in backend web development.



