import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("housing-prices.csv")

print(df.head())
print(df.info())

# Features
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

# Target
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

model = LinearRegression()

model.fit(X_train, y_train)

coefficients_df = pd.DataFrame(data=model.coef_, index=X.columns, columns=['Coeff'])
print(coefficients_df)

predictions = model.predict(X_test)

# Evaluating the model
plt.scatter(y_test, predictions)
plt.show()

print("\nMEAN ABSOLUTE ERROR: ", metrics.mean_absolute_error(y_test, predictions))
print("MEAN ROOT SQUARED ERROR: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
