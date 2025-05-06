**Data Flow Diagram (DFD)** is a visual representation of how data moves through a system. It shows the input, processing, storage, and output of data, making it easier to understand the structure and flow of information within a process or system.
Components of a Data Flow Diagram

1. **Processes**

- A process represents an action or transformation performed on data within the system.

- It is typically drawn as a circle or a rounded rectangle in a DFD.

- Each process should have at least one input and one output data flow.
- Example: A process like "Validate Payment" checks the payment details and returns a validation result.

2.  **Data Stores**
- A data store is a place where data is held for future use or reference.

- It is depicted as an open-ended rectangle or two parallel lines.

- Data stores are used by processes to either retrieve or save data.

- Example: A "User Database" might store login credentials and user profiles.


3. **External Entities (Terminators)**
-An external entity is a person, system, or organisation that interacts with the system but exists outside its boundaries.

- It is shown as a rectangle in a DFD.

- External entities either provide data to the system or receive data from it.

- Example: A "Customer" may submit an order or receive a shipping confirmation.


4. **Data Flows**
- A data flow shows the movement of data between processes, data stores, and external entities.

- It is represented by a labeled arrow in a DFD.

- Each arrow should clearly indicate what data is moving and in which direction.

- Example: "Order Details" might flow from the Customer to the "Process Order" component.

Data Flow Diagram Code  in Python using Graphviz:
```python
from graphviz import Digraph

# Create a new directed graph
dfd = Digraph("DataFlowDiagram", filename="dfd_example", format="png")

# External entities
dfd.node("User", shape="rectangle")
dfd.node("Admin", shape="rectangle")

# Processes
dfd.node("Login", shape="ellipse")
dfd.node("Validate User", shape="ellipse")
dfd.node("Show Dashboard", shape="ellipse")

# Data Stores
dfd.node("UserDB", shape="cylinder")

# Data Flows
dfd.edge("User", "Login", label="Enter Credentials")
dfd.edge("Login", "Validate User", label="Send Info")
dfd.edge("Validate User", "UserDB", label="Query User Data")
dfd.edge("UserDB", "Validate User", label="Return User Data")
dfd.edge("Validate User", "Show Dashboard", label="Access Granted")
dfd.edge("Show Dashboard", "User", label="Show UI")
dfd.edge("Admin", "UserDB", label="Manage Users")

# Render diagram
dfd.render()
```

**Graphviz** in Python is commonly used to visualize decision trees in machine learning, especially with libraries like scikit-learn. It helps convert tree-based models into a clear, human-readable graph format (usually .dot or image files like .png).

Graphviz sample code:
```python
# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Step 2: Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 3: Train a decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Step 4: Export the trained decision tree as DOT data
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=iris.feature_names,
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True)

# Step 5: Create a Graphviz object and render the tree
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # Saves to iris_decision_tree.pdf
graph.view()  # Opens the PDF in your default viewer
```

The primary purpose of a **Data Flow Diagram (DFD)** is to visually represent how data moves through a system, making it easier to understand the structure, functions, and interactions within that system. DFDs help analysts, designers, and stakeholders identify how information is input, processed, stored, and output, without delving into technical implementation details. They are especially useful during the early stages of system design, as they provide a clear and concise way to model data processes and interactions. By using DFDs, teams can identify inefficiencies, redundancies, or missing components in a workflow, which supports better system planning and communication. Whether used for documenting existing systems or designing new ones, DFDs serve as an effective tool for aligning business and technical perspectives.

**Partitions** refer to how a dataset is divided into subsets to train, validate, and test a model. This is a crucial step for evaluating and improving model performance. Here are the common types of partitions:

1. **Training Set**
- Purpose: Used to train the model—i.e., help it learn patterns from the data.

- Typical Size: ~60–80% of the total dataset.

2. **Validation Set**
- Purpose: Used during training to fine-tune hyperparameters and prevent overfitting.

- Typical Size: ~10–20% of the dataset.

- Not always used when employing techniques like cross-validation.

3. **Test Set**
- Purpose: Used only after training is complete to evaluate final model performance.

- Typical Size: ~10–20% of the dataset.

- Important: The model never sees this data during training or validation.

4. **Cross-Validation**
- A technique where the dataset is split into k equal parts (folds), and the model is trained and validated k times, each time with a different fold as the validation set.

- Helps reduce variability and provides a more reliable estimate of model performance.

Sample code for Partition using Scikit-learn:
```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Example dataset (replace with your real data)
data = pd.read_csv("your_dataset.csv")  # Load your data
X = data.drop("target", axis=1)         # Features
y = data["target"]                      # Target/label

# Step 1: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional Step 2: Further split train into Train/Validation (e.g., 60% train, 20% validation)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Result:
# X_train_final: 60% of data
# X_val: 20% of data
# X_test: 20% of data

print("Training set size:", X_train_final.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)
```

**Random sampling** in machine learning is selecting a subset of data randomly from a larger dataset. It’s used to ensure that the sample is representative of the entire population, avoiding bias and improving generalization.

Common Uses of Random Sampling in ML:

Train/Test Split

Randomly split the dataset into training and test sets.

Helps ensure that both sets are representative and not biased toward specific data patterns.

Cross-Validation

Folds are created using random sampling, ensuring variety in validation sets.

Bootstrapping

Random samples with replacement are drawn to create multiple datasets for methods like bagging (used in Random Forests).

Stochastic Gradient Descent (SGD)

Random samples (mini-batches) are used during training to reduce computation time and improve convergence.


**Training data** is the subset of a dataset used to train a model, that is, to teach the algorithm how to make predictions or classify data based on input features.


Characteristics of Training Data:

Labeled (in supervised learning):

Each example includes both input features (X) and the correct output/label (y).

Example: In image classification, the training data includes pictures (features) and their correct categories (labels).

Unlabeled (in unsupervised learning):

Only input features are given; the algorithm tries to learn patterns without guidance.

Example: In clustering, the training data contains just the data points.

Used for Model Fitting:

The algorithm finds patterns, weights, or rules in the training data that minimize prediction error.


**High variance** in machine learning refers to a model's tendency to perform exceptionally well on training data but poorly on unseen or test data. This happens because the model becomes too complex and starts to learn not just the underlying patterns in the data, but also the random noise and minor fluctuations. As a result, it essentially memorizes the training data instead of generalizing from it. This leads to overfitting, where the model’s predictions are highly accurate on the training set but inaccurate when applied to new data. High variance typically arises from using overly flexible models, having too many features relative to the number of training samples, or working with a small or noisy dataset. To combat high variance, practitioners often simplify the model, use regularization techniques, gather more data, or apply ensemble methods like bagging to improve generalization and reduce overfitting.

**Low variance** in machine learning means that a model's predictions are consistent across different training datasets, and it doesn't fluctuate significantly when exposed to new or slightly varied data. This stability indicates that the model is not overly sensitive to small changes in the input data, which is a desirable trait for generalization. A low variance model typically avoids overfitting, meaning it doesn't memorize the training data but instead captures the underlying patterns in a way that allows it to perform well on unseen data. While low variance is generally beneficial, if combined with high bias, it can still result in underfitting, where the model is too simplistic to accurately capture complex relationships in the data. Ideally, a well-performing model balances both low variance and low bias to achieve high accuracy and generalization.

**High bias** in machine learning occurs when a model is too simplistic to accurately capture the underlying patterns in the data, leading to underfitting. This means the model makes strong assumptions about the structure of the data and lacks the flexibility needed to learn complex relationships. As a result, it performs poorly not only on new, unseen data but also on the training data itself. High bias can be caused by using an overly simple algorithm, such as linear regression on a nonlinear dataset, or by ignoring important features. To address high bias, one might use a more complex model, incorporate additional relevant features, reduce regularization, or apply better feature engineering. In essence, a high-bias model cannot learn from the data effectively, resulting in consistently inaccurate predictions.

**Low bias** in machine learning refers to a model's ability to accurately capture the underlying patterns in the training data, without making overly simplistic assumptions. A model with low bias is flexible enough to learn complex relationships between input features and target outputs, leading to better performance on the training set. This is particularly important for tasks involving intricate patterns, where a more complex model (such as a decision tree or neural network) might be necessary. While low bias allows the model to perform well on the training data, it must be balanced with low variance to ensure the model generalizes well to unseen data. If low bias is achieved at the cost of high variance, the model may overfit, memorizing the training data and failing to predict new data accurately. Thus, achieving low bias is key to ensuring that a model 


**Polynomial regression** is a type of regression analysis in machine learning where the relationship between the independent variable x and the dependent variable y is modeled as an nth-degree polynomial. Polynomial regression allows for curved (non-linear) relationships.

Polynomial Regression formula: 



Where:
y is the target/output variable


x is the input/feature variable


β0,β1,...,βn​ are coefficients learned during training


n is the degree of the polynomial


ϵ is the error term



1. **Transforms input features**: Polynomial regression doesn't require a fundamentally new algorithm; it uses linear regression on a transformed version of the features 
2. **Captures non-linear patterns**: Useful when the data shows curvature that a straight line cannot capture.
3. **Still a linear model**: Despite modeling non-linear relationships, it's linear in terms of the parameters (the coefficients).


Polynomial Regression Sample Code with scikit-learn:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature values between 0 and 10
y = 2 + 0.5 * X**2 - X + np.random.randn(100, 1) * 5  # Quadratic relationship with noise

# Transform features to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit a linear regression model to the polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Predict using the model
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# Plot results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Polynomial Regression Fit')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```
**Polynomial regression** is a valuable extension of linear regression that enables machine learning models to capture non-linear relationships between variables. By introducing polynomial terms, it provides greater flexibility in modeling complex data patterns that a straight line cannot accurately represent. While it remains a linear model in terms of parameters, its ability to fit curved trends makes it especially useful in scenarios where the relationship between input and output is not strictly linear.
However, this flexibility comes with trade-offs. Higher-degree polynomials can lead to overfitting, where the model fits the training data too closely and fails to generalize well to new data. To use polynomial regression effectively, it's important to choose an appropriate degree based on the complexity of the data and to validate the model using techniques like cross-validation. Overall, polynomial regression is a simple yet powerful tool when applied with care and proper evaluation.
