import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# medv -- Median Value of Owner-Occupied Homes
# ptratio -- Pupil-Teacher Ratio
# lstat -- Percentage of Lower Status of the Population
# rm -- Average Number of Rooms per Dwelling

# loading dataset
file_path = 'bostonhousingdata/housing.csv'  # Update this path to your actual file path
boston = pd.read_csv(file_path)

# display first few rows of the dataset
print(boston.head())

# Perform basic EDA
sns.pairplot(boston)
plt.show()

# Prepare the data for training
X = boston.drop('medv', axis=1)
y = boston['medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
