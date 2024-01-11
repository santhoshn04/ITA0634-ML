import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load the dataset
dataset = pd.read_csv(r"C:\Users\Raju\Downloads\DATASET\DATASET\Salary_Data.csv")

# Split the dataset into independent variables (X) and dependent variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the salaries for the test data
y_pred = model.predict(X_test)

#model good or not
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


#Visualising the Training set results Here scatter plot is used to visualize the results. 

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


