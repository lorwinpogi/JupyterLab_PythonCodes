**K-means clustering (K-Means Algorithm)** is a technique used to organize data into groups based on their similarity. For example, an online store uses K-Means to group customers based on purchase frequency and spending, creating segments like Budget Shoppers, Frequent Buyers, and Big Spenders for personalised marketing.

The algorithm works by first randomly picking some central points called centroids, and each data point is then assigned to the closest centroid, forming a cluster. After all the points are assigned to a cluster, the centroids are updated by finding the average position of the points in each cluster. This process repeats until the centroids stop changing, forming clusters. The goal of clustering is to divide the data points into clusters so that similar data points belong to the same group.

It is called "K-Means" because:

**K** is the number of clusters you specify.

The algorithm uses the **mean** (average) of the data points in a cluster to find its center (centroid).

**How K-Means Works (Step-by-Step):**

Choose K – the number of clusters you want to divide your data into.

Initialize centroids – randomly pick K data points as initial cluster centers.

Assign points to clusters – each data point is assigned to the nearest centroid.

Update centroids – for each cluster, recalculate the centroid (mean) of all points in the cluster.

Repeat steps 3–4 until the centroids don't change much or a maximum number of iterations is reached.

K-Means Formula:

![image](https://github.com/user-attachments/assets/3bf3c093-768e-4c5d-b31b-53831c483217)


Sample Code for K-Means:

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Generated Data")
plt.show()

# Apply KMeans algorithm
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering Result")
plt.show()
```
**Data forecasting** using machine learning is the practice of predicting future values or events using patterns learned from historical data. It merges traditional time series analysis with advanced ML algorithms for more accurate and scalable forecasting.

**Machine Learning Forecasting:**

1. **Supervised Learning**
  Historical input data (features) and known output (target) are used to train a model that can predict future values. Common algorithms:
- Linear Regression

- Decision Trees and Random Forests

- Gradient Boosting (e.g. XGBoost, LightGBM)


- Neural Networks (especially LSTM and RNN for time series)

2. **Time Series Forecasting**
 Specialized form of forecasting that considers temporal dependencies. Key models include:
- ARIMA/SARIMA (statistical baseline models)

- Facebook Prophet

- LSTM (Long Short-Term Memory)

- Temporal Convolutional Networks (TCNs)

- Transformer-based models for time series (e.g., Informer, TimesNet)

3. **Feature Engineering**
-Extract meaningful features like:
Time-based features: hour, day, month, season

- Lag features: previous values at time t-1, t-2, etc.

- Rolling statistics: moving averages, rolling std dev

- External variables (e.g., weather, holidays, promotions)

4. **Evaluation Metrics**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric MAPE)

Forecasting with Linear Regression sample code:
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample time series data
# Simulated monthly sales data
data = {
    'Month': pd.date_range(start='2022-01-01', periods=24, freq='M'),
    'Sales': [200 + i*5 + np.random.randn()*10 for i in range(24)]
}
df = pd.DataFrame(data)

# Prepare features: convert date to a number
df['Month_Num'] = np.arange(len(df)).reshape(-1, 1)

# Train/Test Split
train_size = 20
X_train = df['Month_Num'][:train_size].values.reshape(-1, 1)
y_train = df['Sales'][:train_size]
X_test = df['Month_Num'][train_size:].values.reshape(-1, 1)
y_test = df['Sales'][train_size:]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Forecast
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df['Month'], df['Sales'], label='Actual Sales')
plt.plot(df['Month'][train_size:], y_pred, label='Forecast', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecasting using Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```
Machine learning is increasingly used for forecasting because it offers significant advantages over traditional statistical methods. Unlike classical models, which often assume linearity and require manually defined patterns, machine learning algorithms can automatically learn complex, nonlinear relationships in data. They can also incorporate a wide variety of input features, such as time, weather, holidays, promotions, and economic indicators, making them more flexible and robust in real-world applications. Additionally, machine learning models are scalable and can handle large volumes of data with high dimensionality, which is essential for modern businesses dealing with big data. These capabilities make machine learning particularly powerful for generating more accurate, dynamic, and adaptive forecasts across industries like retail, finance, healthcare, and energy.

**Types of Machine Learning Forecasting**
**Time series forecasting** is the process of predicting future values based on past observations, where data points are indexed in time order (i.e., sequentially over time). The primary goal is to model temporal dependencies in the data to forecast future trends or values, making it especially useful for applications like sales prediction, stock prices, demand forecasting, weather forecasting, and more.

**Time series data** refers to data points collected or recorded at specific time intervals. It is typically structured with a time-based index, such as daily, monthly, or yearly records. For example, stock market prices recorded daily or temperatures measured every hour are typical examples of time series data.

1. **Temporal order**: Data points are ordered by time.

2. **Sequential dependency:** Future data points depend on past values.

3. **Patterns:** Time series data often exhibits trends, seasonal variations, and cycles.

Components of Time Series Data
Time series data generally has several components that influence its patterns:

 **Trend**
A long-term movement or general direction in the data over time, which can either be increasing, decreasing, or stable. For instance, a company’s sales might show an upward trend over several years due to growing market demand.

 **Seasonality**
Regular and predictable fluctuations that occur at fixed intervals within a year, month, day, or week. An example would be higher sales during the holiday season, or electricity demand increasing during hot summer months.

**Cyclical Patterns**
Fluctuations that occur at irregular intervals, often related to economic cycles, political events, or other long-term changes. Unlike seasonality, these cycles don’t have a fixed period.

**Noise**
Random variation or unexplained fluctuations in the data. Noise can obscure patterns and make it difficult to detect trends and seasonality.


Linear Regression for Time Series Sample Code:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dummy time series data
df = pd.DataFrame({
    'day': np.arange(1, 101),
    'sales': np.arange(1, 101) + np.random.randn(100)*5  # noisy linear trend
})

# Train model
X = df[['day']]
y = df['sales']
model = LinearRegression()
model.fit(X, y)

# Forecast next 10 days
future_days = pd.DataFrame({'day': np.arange(101, 111)})
forecast = model.predict(future_days)

# Plot
plt.plot(df['day'], y, label='Historical')
plt.plot(future_days['day'], forecast, label='Forecast', color='red')
plt.legend()
plt.show()
```

**Regression forecasting** is a technique used to predict a continuous numerical value based on one or more input variables. In the context of time series or general forecasting, regression models are employed to establish the relationship between the target variable (the value you want to predict) and predictor variables (features or factors that may influence the outcome). Unlike traditional time series methods, which are explicitly designed for temporal data, regression forecasting can be applied to any scenario where historical data is used to predict future outcomes.

Components of Regression Forecasting

1. **Dependent Variable** (Target Variable)
The dependent variable (also known as the target variable) is the primary quantity you aim to forecast using regression techniques. It is the outcome you're trying to predict based on patterns in historical data.

2. **Independent Variables** (Features or Predictors)
The independent variables (or features or predictors) are the input variables that influence or explain the variations in the dependent variable. In machine learning, the goal is to find relationships between these predictors and the target variable to make predictions.

3. **Training Data and Test Data**
In machine learning, the available data is typically divided into two parts:

- Training Data: This dataset is used to train the model, helping it learn the relationships between independent and dependent variables.
- Test Data: This dataset is used to evaluate how well the trained model performs on new, unseen data.
- Sometimes, there is also a validation dataset used during model tuning to assess the model's performance and adjust parameters.

4. **Regression Model (Algorithm)**
The regression model is the core of the forecasting process in machine learning. It learns the relationship between the independent variables and the dependent variable from the training data.

5. **Loss Function (Objective Function)**
The loss function (or objective function) is a critical component in machine learning. It measures the difference between the predicted values and the actual target values (from the test or validation set).


6. **Model Evaluation Metrics**
After training a regression model, its performance needs to be evaluated using various evaluation metrics. These metrics help assess how accurately the model is forecasting the dependent variable.

7. **Feature Engineering**
In machine learning, feature engineering is the process of transforming raw data into meaningful features that can help improve model performance.

8. **Hyperparameter Tuning**
Hyperparameters are the configuration settings of the machine learning model that are not learned from the data but set before training. For instance, in decision tree regression, hyperparameters include tree depth and the minimum number of samples required to split a node.

9. **Regularization**
To prevent overfitting and enhance generalization, regularization techniques are applied in machine learning regression models. Regularization methods introduce additional terms to the model's objective function to penalize complex models with large coefficients.

10. **Model Deployment and Forecasting**
Once the machine learning model has been trained, evaluated, and tuned, it can be used for forecasting:

- Prediction: The model can be deployed to make forecasts on future data based on newly available input features.

- Continuous learning: In some cases, the model can be retrained periodically with new data to ensure the forecasts stay relevant and accurate.


