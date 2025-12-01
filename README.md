# 🏠 Real Estate Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![Regression](https://img.shields.io/badge/Task-Regression-red)

### 🎯 Project Overview
Predicting property values is a classic problem in Data Science with immense business value. 

This project builds a **Linear Regression Model** to predict the `House Price of Unit Area` based on various features such as the age of the house, distance to the nearest MRT (Mass Rapid Transit) station, and the number of convenience stores nearby.

---

### 📊 The Dataset
The dataset contains real estate transactions with the following features:

| Feature | Description |
| :--- | :--- |
| `X1 transaction date` | The date of the property sale. |
| `X2 house age` | Age of the house (years). |
| `X3 distance to MRT` | Distance to the nearest MRT station (meters). |
| `X4 convenience stores` | Number of stores in the living circle. |
| `X5 latitude` | Geographic latitude. |
| `X6 longitude` | Geographic longitude. |
| **`Y house price`** | **Target Variable: Price per unit area.** |

---

### 🧠 Methodology

1.  **Data Exploration:**
    * Loaded dataset using Pandas.
    * Checked for missing values (none found).
    * Split data into independent (`X`) and dependent (`y`) variables.

2.  **Modeling:**
    * Split data into **Training (80%)** and **Testing (20%)** sets.
    * Trained a **Linear Regression** model using `scikit-learn`.

3.  **Evaluation:**
    * Evaluated performance using **$R^2$ Score** (goodness of fit) and **Mean Squared Error (MSE)**.

---

### 📈 Results

* **$R^2$ Score:** **0.67** (The model explains ~67% of the variance in house prices).
* **Mean Squared Error (MSE):** **54.6**

#### Key Insights from Visualizations:
* **House Age:** Older houses tend to have slightly lower prices, but the correlation is weak.
* **MRT Distance:** Strong negative correlation. As distance to the MRT increases, house prices drop significantly.
* **Convenience Stores:** Positive correlation. More stores nearby generally indicate higher property value.

---

### 🛠️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```
2.  **Run the Notebook:**
    Open `Task6_Real_Estate_Price_Prediction_Completed.ipynb` in Jupyter Notebook.
3.  **Data:**
    The project uses `Real_estate.csv` (included in the repo).

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com

---

### 💻 Code Snippet: Model Training

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
