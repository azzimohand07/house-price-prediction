# =========================

# Libraries

# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# =========================

# Load Dataset

# =========================

data = pd.read_csv("data/housing.csv")

# Remove missing values

data.dropna(inplace=True)

# Display dataset information

data.info()

# =========================

# Encode Categorical Features

# =========================

data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True, dtype=int)

# =========================

# Feature / Target Split

# =========================

x = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# =========================

# Train-Test Split

# =========================

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# =========================

# Exploratory Data Analysis

# =========================

train_data = x_train.join(y_train)

# Feature distributions

train_data.hist(figsize=(15, 8))
plt.show()

# Correlation heatmap

plt.figure(figsize=(15, 8))

sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

plt.title("Feature Correlation Heatmap")
plt.show()

# =========================

# Log Transform Skewed Features

# =========================

log_columns = ["total_rooms", "total_bedrooms", "population", "households"]

for col in log_columns:
    x_train[col] = np.log(x_train[col] + 1)
    x_test[col] = np.log(x_test[col] + 1)

# =========================

# Feature Scaling

# =========================

numeric_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                   "population", "households", "median_income"]

scaler = StandardScaler()

x_train[numeric_columns] = scaler.fit_transform(x_train[numeric_columns])

x_test[numeric_columns] = scaler.transform(x_test[numeric_columns])

# =========================

# Geographical Price Visualization

# =========================

plt.figure(figsize=(15, 8))

sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")

plt.title("House Prices by Location")
plt.show()

# =========================

# Linear Regression Model

# =========================

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

predictions = linear_model.predict(x_test)

# =========================

# Model Evaluation

# =========================

r2 = r2_score(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# =========================

# Cross Validation

# =========================

cv_scores = cross_val_score(LinearRegression(), x_train, y_train, cv=5, scoring="r2")

print("Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# =========================

# Model Parameters

# =========================

weights = linear_model.coef_
bias = linear_model.intercept_

print("Weights:", weights)
print("Bias:", bias)

# =========================

# SGD Learning Curve

# =========================

train_losses = []
validation_losses = []

sgd = SGDRegressor(max_iter=1, warm_start=True, learning_rate="constant", eta0=0.00001, random_state=42)

n_epochs = 100

for epoch in range(n_epochs):
    sgd.fit(x_train, y_train)
    train_predictions = sgd.predict(x_train)
    validation_predictions = sgd.predict(x_test)
    train_loss = mean_squared_error(y_train, train_predictions)
    validation_loss = mean_squared_error(y_test, validation_predictions)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

# =========================

# Plot Learning Curve

# =========================

plt.figure(figsize=(15, 8))

plt.plot(train_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("SGD Learning Curve")

plt.legend()
plt.show()

sgd_pred = sgd.predict(x_test)

print("SGD R²:", r2_score(y_test, sgd_pred))
print("SGD RMSE:", np.sqrt(mean_squared_error(y_test, sgd_pred)))

print("Linear R²:", r2_score(y_test, predictions))
print("Linear RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))



