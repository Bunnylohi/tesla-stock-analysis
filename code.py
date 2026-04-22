# Tesla Stock Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("Tesla Stock Dataset.csv")   # replace with your file name
print(df.head())
print(df.info())
print(df.describe())

# -----------------------------
# Step 2: Exploratory Data Analysis (EDA)
# -----------------------------
# Closing price trend
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.title("Tesla Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.show()

# Volume distribution
sns.histplot(df['Volume'], bins=30, kde=True)
plt.title("Trading Volume Distribution")
plt.show()

# Correlation heatmap
corr = df[['Open','High','Low','Close','Adj Close','Volume']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Step 3: Feature Engineering
# -----------------------------
# Daily returns
df['Daily_Return'] = df['Close'].pct_change()

# Moving averages
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

# Plot moving averages
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['MA20'], label='MA20')
plt.plot(df['Date'], df['MA50'], label='MA50')
plt.title("Tesla Closing Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Step 4: Predictive Modeling
# -----------------------------
# Drop NaN rows created by rolling averages
df = df.dropna()

# Features and target
X = df[['Open','High','Low','Volume','MA20','MA50']]
y = df['Close']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.title("Actual vs Predicted Tesla Close Price")
plt.show()
