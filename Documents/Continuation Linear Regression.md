**Mean Absolute Error (MAE)** is a commonly used metric in machine learning to measure the average magnitude of errors in a set of predictions, without considering their direction. It's the average of the absolute differences between predicted values and actual values.

Formula:

 ![alt text](image-4.png)
Where:
yii​ = actual value
y^i = predicted value
n = number of data points
∣yi−y^i∣ = absolute error for each instance

MAE gives equal weight to all errors.
It is in the same unit as the output variable.
Unlike MSE (Mean Squared Error), MAE is more robust to outliers.

Python Code Example (using scikit-learn):
```python
from sklearn.metrics import mean_absolute_error
# Actual and predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)
```

**Mean Squared Error (MSE)** is a commonly used regression metric in machine learning that measures the average of the squares of the errors, the average squared difference between the actual and predicted values. It tells you how far your model's predictions are from the actual values, with larger errors penalized more than smaller ones.



Where:

![alt text](image-5.png)

n: number of data points


yi​: actual value


y^​i​: predicted value


(yi−y^i)2: squared error for each point

Python Code Example with linear regression:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.1, 1.9, 3.0, 4.2, 5.1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```
**Root Mean Squared Error (RMSE)** is a popular regression metric in machine learning. It’s the square root of the Mean Squared Error (MSE) and measures the average magnitude of prediction errors in the same units as the target variable.
Formula:

![alt text](image-3.png)

Where:
yi​: actual value


y^i: predicted value


n: number of samples

It penalizes large errors more than MAE and is easier to interpret than MSE because it’s in the same unit as the actual values.

Python Code Example Using sklearn:
```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Actual vs predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Calculate RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)

Example using Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.1, 1.9, 3.0, 4.2, 5.1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)
```

**R-squared (R²)** is a regression performance metric that measures how well the predicted values from a model explain the variability of the actual target values. It is also known as the coefficient of determination.
Formula:

![alt text](image-2.png)

Where:
SSres=∑(yi​−y^​i​)2 → residual sum of squares (errors)


SStot=∑(yi​−yˉ​)2 → total sum of squares (variance from the mean)

Sample Python Code with sklearn:
``` python
from sklearn.metrics import r2_score

# Actual and predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Calculate R-squared
r2 = r2_score(y_true, y_pred)
print("R-squared Score:", r2)


Sample code with Linear Regression:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 2.0, 2.9, 4.1, 5.2])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Compute R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)
```


**Decision Tree Classifier** in machine learning is a supervised learning algorithm used for classification tasks. It works by splitting the data into branches based on decision rules inferred from the features, eventually leading to a prediction at the "leaf" nodes.

A Decision Tree builds a model in the form of a flowchart. Each internal node represents a test on a feature (e.g., Age > 30), each branch corresponds to the outcome of that test (True or False), and each leaf node represents a class label (e.g., "Yes" or "No"). The goal is to find the best splits, typically using metrics like Gini Impurity or Entropy, to achieve the most effective separation between classes.

Sample code:
``` python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
```

An **internal node** in a decision tree is any node that performs a test on a feature and splits the data based on the outcome. It is also known as a decision node, and unlike leaf nodes, it does not provide a final prediction. Instead, internal nodes direct the flow of the data down different branches of the tree depending on whether the condition they test is true or false. For example, an internal node might check if temperature > 20, and based on that, the data would go down one of two paths. These nodes are essential for breaking down complex decision-making processes into a series of simpler tests, ultimately leading to a prediction at the leaf nodes.


In a **decision tree**, a branch represents the outcome of a test at an internal (decision) node and connects one node to another. It is the path that the data follows based on the result of the condition being evaluated. Each internal node can have multiple branches, depending on the number of possible outcomes for that condition. For example, if a decision node checks whether Age > 30, the branches would represent the two possible outcomes: "Yes" or "No." These branches lead to further decision nodes or leaf nodes, which provide the final prediction. Essentially, branches guide the flow of data through the tree, ensuring that each decision step is made based on the test outcomes.

A **leaf node** in a decision tree is the final node where the decision-making process ends. Unlike internal nodes, which perform tests and split data into different branches, a leaf node represents the output or prediction of the model. Each leaf node corresponds to a specific class label in classification tasks (like "Yes" or "No") or a value in regression tasks. The data that reaches a leaf node has passed through several decision nodes and branches, and the leaf node provides the final classification or prediction result. Essentially, leaf nodes are where the tree "decides" what the outcome is based on the conditions it has processed along the way.
