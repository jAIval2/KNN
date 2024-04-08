import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib

matplotlib.use('Agg')  # Explicitly set backend

import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data.csv")

# Treat null values - fill with forward fill (extrapolate)
data.ffill(inplace=True)

# Feature Engineering - Calculate moving averages
data['9_ema'] = data['close'].ewm(span=9, min_periods=0, adjust=False).mean()
data['21_ema'] = data['close'].ewm(span=21, min_periods=0, adjust=False).mean()
data['50_ema'] = data['close'].ewm(span=50, min_periods=0, adjust=False).mean()
data['200_ema'] = data['close'].ewm(span=200, min_periods=0, adjust=False).mean()

# Feature Engineering - Rate of Change (ROC)
data['roc'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * 100

# Feature Engineering - Average True Range (ATR)
high_low = data['high'] - data['low']
high_close_prev = np.abs(data['high'] - data['close'].shift(1))
low_close_prev = np.abs(data['low'] - data['close'].shift(1))
ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
data['atr'] = ranges.max(axis=1)

# Feature Engineering - Stochastic Oscillator (%K and %D)
period = 14
data['stoch_%k'] = ((data['close'] - data['low'].rolling(window=period).min()) / (
            data['high'].rolling(window=period).max() - data['low'].rolling(window=period).min())) * 100
data['stoch_%d'] = data['stoch_%k'].rolling(window=3).mean()

# Feature Engineering - Relative Strength Index (RSI)
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['rsi'] = 100 - (100 / (1 + rs))

# Map the target variable to numerical values
signal_mapping = {'none': 0, 'buy': 1, 'sell': 2}
data['signal'] = data['signal'].map(signal_mapping)

# Exclude non-numeric columns before scaling
X = data.drop(['datetime', 'signal'], axis=1)  # Excluding datetime and signal columns
y = data['signal']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the imputed features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Model Selection - K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
print("\nK-Nearest Neighbors:")
print("Accuracy Score:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))
print("Classification Report:")
print(classification_report(y_test, knn_pred, zero_division=1))

# Model Selection - Support Vector Classifier
svc = SVC()
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)
print("\nSupport Vector Classifier:")
print("Accuracy Score:", accuracy_score(y_test, svc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svc_pred))
print("Classification Report:")
print(classification_report(y_test, svc_pred, zero_division=1))

# Model Selection - Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print("\nRandom Forest Classifier:")
print("Accuracy Score:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred, zero_division=1))

# Ensure datetime is in the correct format
data['datetime'] = pd.to_datetime(data['datetime'])

plt.figure(figsize=(14, 7))
plt.plot(data['datetime'], data['close'], label='BTC Close Price', color='skyblue', linewidth=2)

# Reset index of the data DataFrame
data_reset_index = data.reset_index(drop=True)

# Split the dataset into train and test sets
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# Reset index of the test dataset
data_test_reset_index = data_test.reset_index(drop=True)

# Assuming 1 represents 'buy' and 2 represents 'sell' for each classifier
# Highlight buy signals for K-Nearest Neighbors
buy_signals_knn = data_test_reset_index[data_test_reset_index.index < len(knn_pred)][knn_pred == 1]
plt.scatter(buy_signals_knn['datetime'], buy_signals_knn['close'], label='KNN Buy Signal', marker='^', color='green', alpha=1, s=100)

# Highlight sell signals for K-Nearest Neighbors
sell_signals_knn = data_test_reset_index[data_test_reset_index.index < len(knn_pred)][knn_pred == 2]
plt.scatter(sell_signals_knn['datetime'], sell_signals_knn['close'], label='KNN Sell Signal', marker='v', color='red', alpha=1, s=100)

# Repeat the same process for other classifiers (Support Vector Classifier and Random Forest Classifier)

plt.title('BTC Price with Predicted Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('BTC Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('BTC_Predictions3.png')  # Save the plot as an image
