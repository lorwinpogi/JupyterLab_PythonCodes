The core building blocks of regression techniques in machine learning include defining the problem, preparing the data, selecting features, choosing a model, training, evaluating, and deploying it. Regression models predict continuous values and can be linear or non-linear, with variations like polynomial, ridge, and lasso regression. The goal is to find a "best-fit" line or curve that minimizes the difference between predicted and actual values. 

1. **Problem Definition**: Clearly outline the target variable (the value to be predicted) and influencing factors (independent variables). 

2. **Data Preparation**: 
Handling Missing Values: Address missing data through methods like imputation (filling in missing values) or removal. Outlier Removal: Remove or transform outliers (extreme values) that can skew results. Normalization: Scale data to a common range (e.g., using standardization).

3. **Feature Selection**: Identify the most relevant and informative independent variables to use in the model. 

4. **Model Selection & Training**:
Model Choice:
Choose an appropriate regression model based on the data and problem (e.g., linear regression, polynomial regression, ridge regression). 

Model Training:
Use the training data to train the chosen model and adjust its parameters (e.g., coefficients in linear regression). 
Hyperparameter Tuning:
Optimize model parameters (e.g., regularization strength) for better performance. 

5. **Evaluation & Interpretation**: 
Evaluation Metrics:
Assess model accuracy using metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.
Interpretation:
Understand the relationships between the independent variables and the target variable.

6. **Deployment & Monitoring**: Implement the model in real-world applications and monitor its performance over time. 


Types of Regression Techniques:
**Linear Regression**: Predicts continuous values based on a linear relationship. 
**Polynomial Regression**: Handles non-linear relationships by fitting a polynomial curve to the data. 

**Ridge Regression** is a technique for analyzing multiple regression data. When multicollinearity occurs, least squares estimates are unbiased. This is a regularized linear regression model, it tries to reduce the model complexity by adding a penalty term to the cost function. A degree of bias is added to the regression estimates, and as a result, ridge regression reduces the standard errors.
                             
Sample code Scikit-learn in Python:
```python
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=10, noise=10)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict and score
predictions = ridge.predict(X_test)
print("R^2 score:", ridge.score(X_test, y_test))
```

**Lasso Regression** is a regression analysis method that performs both variable selection and regularization. Lasso regression uses soft thresholding. Lasso regression selects only a subset of the provided covariates for use in the final model. This is another regularized linear regression model. It works by adding a penalty term to the cost function, but it tends to zero out some features’ coefficients, which makes it useful for feature selection.

Sample code Scikit-learn in Python:

```python
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=100, n_features=10, noise=10)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Check results
print("Coefficients:", lasso.coef_)
print("R^2 Score:", lasso.score(X_test, y_test))
```

**Logistic Regression** is a supervised machine learning algorithm used for classification problems, not regression, despite its name. It predicts probabilities of class membership (usually binary), and then classifies data like "yes/no", "spam/ham", "disease/no disease", etc.

Output: Probability between 0 and 1
Used for: Binary classification (can be extended to multiclass)
Core idea: Model the log-odds of the probability as a linear combination of input features

Sample code Scikit-learn in Python:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate binary classification data
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probs = clf.predict_proba(X_test)

# Evaluate
print("Predictions:", predictions)
print("Probabilities:", probs[:5])
```

**Gradient Boosting** is a powerful ensemble machine learning technique used for both regression and classification tasks. It builds models sequentially, where each new model corrects the errors of the previous ones. Gradient Boosting builds an ensemble of weak learners (typically decision trees) in a stage-wise fashion. Each new model is trained to minimise the error (loss) made by the combined ensemble so far, using gradient descent.

Sample code Scikit-learn in Python:
```python

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train gradient boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Evaluate
accuracy = gb.score(X_test, y_test)
print("Accuracy:", accuracy)
```

**Feature engineering** is the process of creating, transforming, or selecting input variables (features) to improve the performance of a machine learning model. Feature engineering is the art and science of converting raw data into meaningful input features that better represent the underlying problem to the model.

Feature Engineering Techniques

**Feature Creation** is the process of generating new features (or variables) from the existing raw data to improve the model's predictive performance. It's an essential part of Feature Engineering, which focuses on transforming raw data into a format that can be better understood by machine learning algorithms. Feature creation is particularly powerful when raw features don't adequately represent the underlying relationships or patterns in the data. By constructing new features, you can help the machine learning model identify hidden relationships, reduce noise, and improve accuracy.

Types of Feature Creation Techniques
Feature creation involves a variety of strategies and approaches depending on the nature of the data and the machine learning problem at hand. Here are the most common techniques:

1.**Mathematical Transformations**
You can perform basic mathematical operations (e.g., addition, subtraction, multiplication, division) to create new features that might highlight relationships in the data.

Example:

If you have features like distance and time, you can create a new feature, speed = distance/time.

2.**Datetime Features**
When dealing with date or time data, it's often useful to extract components such as hour, day, month, weekday, etc., which may have predictive power.

Example:

Extract the hour from a timestamp to create an "hour_of_day" feature.

Use the day of the week to create a feature indicating whether it's a weekend or a weekday.

3.**Categorical Feature Encoding**
You can create new features by encoding categorical variables or combining multiple categorical variables.

Example:

Combining the values from pickup_location and dropoff_location into a new feature called location_pair, which captures the interaction between these two variables.

Encoding 'one-hot' or 'target' encoding for categorical variables.

4.**Aggregation**
Aggregating features involves summarizing data at a higher level of granularity, typically for grouped data. It can create new features by summarizing values (mean, max, sum, etc.) over a specified grouping.

Example:

If you are working with transaction data, you could create a feature like the average transaction value per customer.

5.**Interaction Features**
Interaction features are created by combining two or more existing features in a meaningful way. This can help capture non-linear relationships that may not be expressed in the original features.

Example:

Multiply age by income to create a new feature that represents an interaction between these two variables.

6.**Text-based Feature Creation**
When working with textual data, features like word count, sentiment score, or the presence of certain keywords are common feature creation techniques.

Example:

TF-IDF (Term Frequency-Inverse Document Frequency) values or simple word counts can be created for documents.

Sentiment analysis on a review text can create a feature that indicates whether the sentiment is positive, negative, or neutral.

7.**Domain-Specific Features**
In many cases, domain knowledge can help create meaningful features that directly relate to the problem at hand.

Example:

In finance, a credit risk score could be created based on a combination of several variables such as income, debts, and payment history.

In e-commerce, a customer lifetime value (CLV) feature can be created to predict future sales or the likelihood of a customer returning.

8.**Polynomial Features**
For numerical features, polynomial features involve creating higher-degree terms (e.g., squaring or cubing a feature), which might help capture non-linear relationships between the feature and the target variable.

Example:

Create a squared feature for age (e.g., age^2), which can help a model detect quadratic relationships.

Feature Creation Sample code: 
```python
import pandas as pd

# Sample data
data = {
    'pickup_time': pd.to_datetime([
        '2025-05-01 08:00',
        '2025-05-01 17:30',
        '2025-05-02 13:15'
    ]),
    'dropoff_time': pd.to_datetime([
        '2025-05-01 08:25',
        '2025-05-01 18:00',
        '2025-05-02 13:45'
    ]),
    'pickup_location': ['Downtown', 'Uptown', 'Downtown'],
    'dropoff_location': ['Airport', 'Downtown', 'Downtown'],
    'distance_km': [10.5, 8.2, 5.0],
    'fare_usd': [20.0, 18.5, 10.0]
}

# Create DataFrame
df = pd.DataFrame(data)

# --- Feature Creation ---

# 1. Trip duration in minutes
df['trip_duration_min'] = (df['dropoff_time'] - df['pickup_time']).dt.total_seconds() / 60

# 2. Is trip within the same location
df['is_same_location'] = df['pickup_location'] == df['dropoff_location']

# 3. Hour of pickup
df['pickup_hour'] = df['pickup_time'].dt.hour

# 4. Is trip during rush hour (7–9am or 4–6pm)
df['rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if x in list(range(7, 10)) + list(range(16, 19)) else 0)

# 5. Fare per km
df['fare_per_km'] = df['fare_usd'] / df['distance_km']

# 6. Location interaction feature
df['location_pair'] = df['pickup_location'] + "_to_" + df['dropoff_location']

# --- Display final DataFrame ---
print(df)
```

**Feature Transformation** refers to the process of altering or modifying the existing features in your dataset to improve the performance of machine learning models. Unlike Feature Creation, which involves generating entirely new features from raw data, Feature Transformation focuses on changing the scale, distribution, or form of the original features to make them more suitable for learning algorithms. The goal of feature transformation is to optimize the feature space so that models can learn more efficiently, leading to better accuracy, faster convergence, and improved generalization to unseen data.
Types of Feature Transformation

1.**Scaling/Normalization**

Scaling transforms the features to a specific range or scale, ensuring that no single feature dominates due to its magnitude.

Types:
Min-Max Scaling (Normalization): Scales features to a fixed range, typically [0, 1].

 
Use case: Used for algorithms like K-NN, Neural Networks, and SVM.

Standardization (Z-Score Scaling): Centers features by subtracting the mean and dividing by the standard deviation, resulting in a feature distribution with a mean of 0 and a standard deviation of 1.
 
Use case: Used for algorithms like Linear Regression, Logistic Regression, PCA, and SVM.

2.**Log Transformation**

Log transformation is often applied to features that have skewed distributions (e.g., income or population data). It helps compress the scale of large values and bring features closer to a normal distribution.

Use case: Used when features have a long right tail (skewed data).


Example: Applying log transformation to monetary values like price or salary.

3.**Power Transformation**

Power transformations, such as Box-Cox or Yeo-Johnson, are used to stabilize variance and make the data more normally distributed.

Box-Cox: Applies a power transformation to make data more Gaussian (it can only handle positive data).


Yeo-Johnson: An extension of Box-Cox that can handle both positive and negative data.

Use case: Helps in cases where a feature has a non-normal distribution and can stabilize variance.

4.**Binning/Discretization**

Binning is the process of converting continuous features into categorical ones by grouping values into discrete intervals or bins.

Example:

Age feature can be transformed into categories like '0-18', '19-30', '31-50', and '51+'.

Use case: Useful when dealing with high-cardinality features or when continuous values have no meaningful relationship but should still be used for prediction.

5.**Polynomial Transformation**

Polynomial transformations involve generating higher-degree polynomial features from existing features. This is useful when trying to capture non-linear relationships between features and the target variable.

Example:

From a feature X, create new features like X^2, X^3, etc.

Use case: Common in regression problems when you suspect that the relationship between the feature and the target is non-linear.

6.**Encoding Categorical Features**

Categorical features need to be transformed into numeric formats before they can be used by machine learning algorithms.

One-Hot Encoding: Creates binary columns (0 or 1) for each category in a feature.

Example: A color feature with values like 'Red', 'Green', and 'Blue' would be transformed into three columns: color_Red, color_Green, color_Blue.

Label Encoding: Assigns an integer value to each category (useful when categories have an inherent order).

Example: Low = 0, Medium = 1, High = 2.

Target Encoding: Replaces categories with the mean of the target variable for that category.

7.**Feature Hashing (Hashing Trick)**

Feature hashing is used to reduce dimensionality by converting high-cardinality categorical features into a smaller set of numeric features using a hash function.

Use case: Useful when dealing with categorical features that have a large number of unique values (e.g., words in text).

Feature Transformation sample code in Python:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'salary': [40000, 50000, 60000, 70000],
    'gender': ['Male', 'Female', 'Female', 'Male']
})

# 1. Standardization (Z-score scaling)
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])

# 2. Min-Max Scaling
scaler = MinMaxScaler()
df['salary_scaled'] = scaler.fit_transform(df[['salary']])

# 3. One-Hot Encoding for categorical data
encoder = OneHotEncoder(sparse=False)
encoded_gender = encoder.fit_transform(df[['gender']])
encoded_df = pd.DataFrame(encoded_gender, columns=encoder.get_feature_names_out(['gender']))

# Concatenate the encoded features
df_encoded = pd.concat([df, encoded_df], axis=1)

print(df_encoded)
```
**Encoding categorical variables** in feature engineering is converting non-numerical data into numerical representations that machine learning algorithms can understand. This is a crucial step because many algorithms require numerical inputs. Common encoding techniques include one-hot, label, ordinal, and more.

Engineering Techniques for Encoding Categorical Variables:

1. **Label Encoding** is a technique used in machine learning to convert categorical variables (i.e. data with text labels like "red", "blue", "green") into a numeric format that algorithms can process. The categorical variable is ordinal (has an inherent order). Converts each category into an integer value.

Sample Code:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Quality'] = le.fit_transform(data['Quality'])  # ['Low', 'Medium', 'High'] → [0, 1, 2]
```

2. **One-Hot Encoding** is a method used to convert categorical variables into a format that can be provided to machine learning algorithms by creating binary columns for each category. Instead of assigning a single integer like Label Encoding, One-Hot Encoding creates a new column for each unique category, with 1s and 0s indicating the presence or absence of that category. The variable is nominal (no inherent order). Creates binary columns for each category.

Sample Code:
```python
import pandas as pd

df = pd.DataFrame({'Colour': ['Red', 'Blue', 'Green']})
encoded = pd.get_dummies(df, columns=['Colour'])

print(encoded)
```

3. **Ordinal Encoding** is a technique used to convert categorical variables into integers based on a specific order or ranking. It is especially useful when the categories have a natural, meaningful order (but not necessarily equally spaced), such as levels of education or product sizes. The categories have a known rank but not equidistant values. Manually assign numbers based on domain knowledge.

Sample code:
```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

data = np.array([['Small'], ['Medium'], ['Large']])
encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
encoded = encoder.fit_transform(data)

print(encoded)
```

4. **Target Encoding**, also known as Mean Encoding, is a technique where each category in a categorical feature is replaced with the mean of the target variable for that category. It’s primarily used in supervised learning, where you have a known target variable (e.g., price, likelihood, etc.). Categories are high in number and correlated with the target variable. Replaces each category with the mean of the target variable for that category.

Sample Code:

```python
import pandas as pd

df = pd.DataFrame({
    'Brand': ['Ford', 'BMW', 'Ford', 'BMW', 'Toyota'],
    'Resale_Value': [10000, 20000, 11000, 21000, 15000]
})

# Calculate mean per brand
means = df.groupby('Brand')['Resale_Value'].mean()

# Replace brand with the mean resale value
df['Brand_Encoded'] = df['Brand'].map(means)

print(df)
```

5. **Binary Encoding** is a hybrid encoding technique that combines the benefits of Label Encoding and One-Hot Encoding, used to represent categorical variables as binary digits. You have high-cardinality categorical variables. Converts categories to integers and then to binary, each binary digit becomes a new feature.

It works by:

- Label Encoding each category into an integer.

- Converting the integer into its binary form.

- Splitting the binary digits into separate columns.


Sample Code:
```python
import pandas as pd
import category_encoders as ce

df = pd.DataFrame({'Category': ['A', 'B', 'C', 'D', 'E']})
encoder = ce.BinaryEncoder(cols=['Category'])
binary_encoded = encoder.fit_transform(df)

print(binary_encoded)
```

6. **Hash Encoding**, also called the Hashing Trick, is a method for encoding high-cardinality categorical variables into a fixed number of columns using a hash function. Instead of assigning integers or creating separate columns for each category (like One-Hot or Binary Encoding), it applies a hashing algorithm to map each category into one of k columns, where k is a number you define in advance. The dataset is very large and has high-cardinality categorical features. Applies a hash function to map categories to a fixed number of dimensions.

Sample Code:

```python

import pandas as pd
import category_encoders as ce

df = pd.DataFrame({'Fruit': ['Apple', 'Banana', 'Cherry', 'Durian']})
encoder = ce.HashingEncoder(cols=['Fruit'], n_components=4)
hashed_df = encoder.fit_transform(df)

print(hashed_df)
```

7. **Frequency Encoding** is a technique used to encode categorical variables by replacing each category with the frequency (or count) of that category in the dataset. In simpler terms, it transforms each category into a numerical value representing how often that category appears in the data. The frequency of a category occurrence is important or indicative. Replace category with its frequency (count or proportion).

Sample Code:

```python

import pandas as pd

# Sample data
df = pd.DataFrame({'Fruit': ['Apple', 'Banana', 'Apple', 'Cherry', 'Banana', 'Apple']})

# Frequency encoding
freq_encoding = df['Fruit'].value_counts()
df['Encoded'] = df['Fruit'].map(freq_encoding)

print(df)
```
**Handling missing values** is an essential step in data preprocessing for machine learning and engineering tasks. In real-world datasets, missing values can arise due to various reasons, and how you deal with them can significantly affect the performance of your model. Below are common engineering techniques to handle missing values, their use cases, and considerations.

Techniques for Handling Missing Values:

1. **Removing missing data** is one of the most straightforward methods for handling missing values in a dataset. It involves deleting rows or columns that contain missing values, ensuring that the dataset is complete and free of gaps. This technique is particularly useful when the missing data is random or not critical to the analysis or model. The missing data is small, non-critical, or doesn’t significantly impact the overall dataset. Remove rows or columns that contain missing values.


Sample Code:
```python
import pandas as pd

# Sample DataFrame
data = {'Name': ['John', 'Emma', 'Ryan', 'Sarah', 'David'],
        'Age': [30, None, 25, None, 35],
        'City': ['London', 'Paris', None, 'Madrid', 'Berlin']}

df = pd.DataFrame(data)

# Remove rows with missing data
df_cleaned_rows = df.dropna()

# Remove columns with missing data
df_cleaned_cols = df.dropna(axis=1)

print("Rows removed with missing data:\n", df_cleaned_rows)
print("\nColumns removed with missing data:\n", df_cleaned_cols)
```

2. **Imputation with Mean, Median, or Mode** is a technique used to fill in missing values by replacing them with a statistical measure (mean, median, or mode) of the non-missing values in a column. This method is simple and efficient, especially when dealing with numerical or categorical missing data. The missing values are assumed to be missing at random and the data distribution is not too skewed. Replace missing values with the mean (for numerical features), median (for skewed numerical data), or mode (for categorical data) of the column.

Sample Code:
```python 
import pandas as pd
from sklearn.impute import SimpleImputer

# Sample data
data = {'Name': ['John', 'Emma', 'Ryan', 'Sarah', 'David'],
        'Age': [30, None, 25, None, 35],
        'City': ['London', 'Paris', 'Paris', 'Madrid', 'Berlin']}

df = pd.DataFrame(data)

# Mean Imputation for Age
imputer_mean = SimpleImputer(strategy='mean')
df['Age_imputed_mean'] = imputer_mean.fit_transform(df[['Age']])

# Mode Imputation for City
imputer_mode = SimpleImputer(strategy='most_frequent')
df['City_imputed_mode'] = imputer_mode.fit_transform(df[['City']])

print(df)
```

3. **Imputation with Predictive Models** is an advanced technique used to fill in missing data by predicting the missing values based on other observed variables in the dataset. Instead of using simple statistical measures (like mean, median, or mode), this method uses machine learning models (such as linear regression, k-nearest neighbors, decision trees, etc.) to make predictions for missing values. The idea is to treat the missing value as a target variable and use the rest of the features in the dataset as predictors to train a model. Once the model is trained, it can predict the missing values for each row based on the observed data. The feature with missing values is important and requires a more sophisticated approach. Train a predictive model (e.g., linear regression, k-NN, decision tree) on the features without missing values to predict the missing values in the target feature. 

Sample Code:
```python

import pandas as pd
from sklearn.impute import KNNImputer

# Sample data with missing values
data = {'Name': ['John', 'Emma', 'Ryan', 'Sarah', 'David'],
        'Age': [30, None, 25, None, 35],
        'City': ['London', 'Paris', 'Paris', 'Madrid', 'Berlin']}

df = pd.DataFrame(data)

# KNN Imputation (using only numeric columns)
imputer = KNNImputer(n_neighbors=2)
df['Age_imputed'] = imputer.fit_transform(df[['Age']])

print(df)
```

4. **K-Nearest Neighbors (KNN) Imputation** is a technique for filling in missing values based on the values of nearby data points. It uses a distance-based algorithm to identify the K-nearest neighbors of an instance with missing data, then imputes the missing value using the average (or most common value) of those neighbors. This method is particularly useful when there are complex relationships in the data, and it is common in situations where the data is continuous or has well-defined groups.  You want to impute missing values based on the similarity between rows (neighbors) in the dataset. The KNN imputation algorithm fills in missing values by looking at the values of the nearest neighbors (similar rows) and imputing based on a majority or average.

Sample Code:
```python 

import pandas as pd
from sklearn.impute import KNNImputer

# Sample data with missing values
data = {'Name': ['John', 'Emma', 'Ryan', 'Sarah', 'David'],
        'Age': [30, None, 25, 35, 32],
        'Height': [175, 160, 180, 165, 170],
        'Weight': [70, 60, 80, 65, 75]}

df = pd.DataFrame(data)

# KNN Imputation
imputer = KNNImputer(n_neighbors=2)
df[['Age']] = imputer.fit_transform(df[['Age', 'Height', 'Weight']])

print(df)
```

5. **Multiple Imputation** is an advanced technique for handling missing data that addresses the inherent uncertainty in missing values by creating multiple complete datasets through imputation. Instead of imputing a single value for each missing data point, Multiple Imputation generates several plausible values for each missing entry, resulting in multiple versions of the dataset. These datasets are then analyzed separately, and the results are combined to provide more robust and accurate estimates, reflecting the uncertainty about what the true missing values might be. This technique is particularly useful when the missing data mechanism is complex (e.g., data is missing not at random), as it allows for statistical inferences that account for uncertainty about the missing values. You want a more robust method that accounts for uncertainty in missing data. Multiple imputation creates several imputed datasets, analyzes each one, and combines the results, reducing bias and variance in the imputation process.

Sample Code:

```python 
import pandas as pd
from fancyimpute import IterativeImputer

# Sample data with missing values
data = {'Name': ['John', 'Emma', 'Ryan', 'Sarah', 'David'],
        'Age': [30, None, 25, None, 35],
        'Height': [175, 160, 180, 165, 170],
        'Weight': [70, 60, 80, 65, 75]}

df = pd.DataFrame(data)

# Apply Multiple Imputation (MICE method)
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df[['Age', 'Height', 'Weight']])

# Convert the imputed data back into a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=['Age', 'Height', 'Weight'])

print(df_imputed)
```

6.  **Forward Fill and Backward Fill** are simple, widely used techniques for imputing missing values in time series or sequential data. These methods fill in missing values by using the values from adjacent data points, making them particularly useful when the data exhibits a time-dependent structure or ordering. The data is time-series, and the values are missing in a sequential order. Use the previous (forward fill) or next (backward fill) available value to fill in the missing data.

Sample Code:
```python
import pandas as pd

# Sample DataFrame with missing values
data = {'Time': ['T1', 'T2', 'T3', 'T4'],
        'Value': [10, None, None, 15]}

df = pd.DataFrame(data)

# Forward Fill
df_forward_fill = df.fillna(method='ffill')

# Backward Fill
df_backward_fill = df.fillna(method='bfill')

print("Forward Fill:")
print(df_forward_fill)
print("\nBackward Fill:")
print(df_backward_fill)
```
7. **Using Domain Knowledge** for Imputation is a sophisticated technique that involves leveraging expert knowledge about the subject matter of the data to inform the imputation of missing values. This method is particularly useful when the missing data is not randomly missing and its pattern can be explained by external factors or inherent relationships in the domain. By integrating insights from industry-specific understanding, real-world constraints, or logical relationships between variables, domain knowledge can guide more accurate and contextually meaningful imputations than traditional statistical methods. You have strong domain expertise that allows you to estimate reasonable values based on business knowledge. Impute missing values using predefined rules based on domain knowledge or expert input.

Sample Code:
```python 
import pandas as pd
import numpy as np

# Sample medical data with missing values
data = {'Age': [25, 45, 60, 32, 55],
        'Weight': [70, 80, 90, 60, 85],
        'Blood Pressure': [120, None, None, 110, None]}

df = pd.DataFrame(data)

# Using domain knowledge (e.g., a simple rule: Older individuals tend to have higher blood pressure)
df['Blood Pressure'] = df.apply(
    lambda row: 130 if pd.isna(row['Blood Pressure']) and row['Age'] > 50 else row['Blood Pressure'],
    axis=1
)

print(df)
```

8. **Interpolation** is a technique used to estimate missing or unknown values within a range of known values in a dataset. It works under the assumption that the missing data points can be reasonably estimated by looking at the surrounding data points. This method is especially useful when dealing with time series data, sequential data, or any data where the values are expected to follow a continuous pattern. The missing values are numerical, and the data is ordered or sequential (e.g., time-series data). Interpolation fills missing values by estimating based on surrounding data points. Linear, polynomial, or spline interpolation can be used.


Sample Code:
```python 
import pandas as pd
import numpy as np

# Create a sample time series data with missing values
data = {'Time': [1, 2, 3, 4, 5],
        'Value': [10, np.nan, np.nan, 40, 50]}

df = pd.DataFrame(data)

# Perform linear interpolation
df['Interpolated Value'] = df['Value'].interpolate(method='linear')

print(df)
```

**Dimensionality** refers to the number of features (or variables) in a dataset. In engineering and machine learning, dimensionality can play a crucial role in model performance, and techniques for dimensionality reduction are often employed to improve the efficiency and accuracy of models, especially when dealing with high-dimensional data.

Dimensionality in Engineering Techniques:

1. **High Dimensionality**
High dimensionality means that your dataset has many features or variables (e.g., hundreds or thousands). This can be problematic for several reasons:

- Overfitting: With more features, models can easily become too complex and fit to noise in the data.

- Increased computational cost: More dimensions require more processing power and memory.

- Curse of Dimensionality: As the number of features increases, the volume of the feature space grows exponentially, and the distance between points becomes more uniform, making models less effective.

For example, if you’re working with a dataset that has 1,000 features, your model has to process and analyze all of those features. This increases the computational burden and can make the model prone to overfitting, especially if the number of observations is low relative to the number of features.


2. **Principal Component Analysis (PCA)** is a dimensionality reduction technique used in data preprocessing and exploratory data analysis. It transforms a large set of variables into a smaller one that still contains most of the information in the original dataset. This is achieved by identifying new uncorrelated variables, called principal components, which are linear combinations of the original variables. You want to reduce the dimensionality of a dataset by transforming features into a smaller set of uncorrelated components. PCA identifies the directions (principal components) in which the data varies the most and projects the data onto these components. The components are ordered by the amount of variance they explain.

Sample code:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'Feature1': [2, 4, 6, 8],
    'Feature2': [1, 3, 5, 7],
    'Feature3': [5, 7, 9, 11]
})

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Show explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```


3. **Linear Discriminant Analysis (LDA)** is a supervised machine learning technique used for both dimensionality reduction and classification. Unlike Principal Component Analysis (PCA), which is unsupervised and focuses on capturing maximum variance, LDA takes class labels into account and seeks to maximize class separability. The goal is to reduce dimensionality while maintaining the class separability. LDA works by finding a linear combination of features that best separates different classes in the data. Unlike PCA, which is unsupervised, LDA is supervised and uses class labels to guide the dimensionality reduction.


Sample Code:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualize the results
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection of Iris Dataset')
plt.show()
```

4. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a non-linear dimensionality reduction technique that is primarily used for data visualization. It transforms high-dimensional data into a low-dimensional space (usually 2D or 3D) while preserving the local structure of the data, meaning it keeps similar points close together. You need to reduce the dimensionality of the data for visualization purposes. t-SNE focuses on preserving the local structure of the data by minimizing the divergence between probability distributions that represent pairwise similarities. It is particularly effective for visualizing high-dimensional data in 2D or 3D.

Sample Code:

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load example data
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("t-SNE Visualization of Iris Dataset")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
```


5. **Autoencoders** are a special type of artificial neural network used for unsupervised learning, primarily for dimensionality reduction, feature learning, denoising, and data compression. They work by learning to reconstruct their input after compressing it into a lower-dimensional representation, also known as a latent space or bottleneck layer. You have non-linear data and want a more sophisticated method for dimensionality reduction. Autoencoders are a type of neural network that learn to compress the input into a lower-dimensional latent space and then reconstruct the original data. The encoder part of the autoencoder reduces the data to a lower-dimensional representation.

Sample Code:
```python
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
import numpy as np

# Load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), 784))

# Encoder
input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)

# Decoder
decoded = Dense(784, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
```



6. **Feature selection** is the process of selecting a subset of relevant features (variables, predictors) from your dataset that contribute the most to the predictive power of your model. It's a key step in data preprocessing and model optimization, especially in high-dimensional datasets. You want to reduce dimensionality by selecting a subset of the most important features rather than transforming them. Feature selection techniques evaluate the relevance of features and select a subset that contributes the most to the model performance.

- Filter methods: Use statistical tests to select features (e.g., Chi-Square, ANOVA).

- Wrapper methods: Use a machine learning model to evaluate subsets of features (e.g., Recursive Feature Elimination).

- Embedded methods: Perform feature selection during model training (e.g., Lasso, Decision Trees).

Sample Code:
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load sample data
X, y = load_iris(return_X_y=True)

# Apply RFE
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(X, y)

print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
```



7. **Independent Component Analysis (ICA)** is a dimensionality reduction technique used to separate a multivariate signal into additive, independent components. It is particularly useful when you assume that the observed data is a mixture of statistically independent non-Gaussian signals. ICA is widely used in signal processing, image separation, and feature extraction, especially when the goal is to uncover hidden sources. You need to find independent components in the data rather than uncorrelated components, like in PCA. ICA attempts to decompose the dataset into statistically independent components, which can be useful when the features are non-Gaussian.

Sample code:
```python
from sklearn.decomposition import FastICA
import numpy as np

# Simulated mixed signals
X = np.random.rand(1000, 2)

# Apply ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # Estimated sources
A_ = ica.mixing_           # Estimated mixing matrix

# Reconstruct signals (optional)
X_reconstructed = S_.dot(A_.T)

```
