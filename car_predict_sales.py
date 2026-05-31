import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("car data.csv")

print(data.head())

print(data.info())

print(data.columns)


print(data.isnull().sum())

data = data.dropna()


le = LabelEncoder()

data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'])
data['Selling_type'] = le.fit_transform(data['Selling_type'])
data['Transmission'] = le.fit_transform(data['Transmission'])


X = data[['Year',
          'Present_Price',
          'Driven_kms',
          'Fuel_Type',
          'Selling_type',
          'Transmission',
          'Owner']]


y = data['Selling_Price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Predicted Prices:")
print(y_pred[:5])


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

plt.show()