import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Load dataset
data = pd.read_csv(r"C:\Users\sujat\OneDrive\Desktop\goldpriceprediction\gold_price_data.csv")

print("Dataset:")
print(data.head())

# Select Features
X = data[['Open','High','Low']]
y = data['Close']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# AI Model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Predicted Prices:", predictions)

# Accuracy
accuracy = metrics.r2_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Gold Price Prediction")
plt.show()