import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score

# Load the dataset

diabetes_data = pd.read_csv('dataset.csv')

# Data Preprocessing
# Check for missing values
print("Missing values in each column:")
print(diabetes_data.isnull().sum())

# As the dataset doesn't contain any missing values, we can proceed with scaling the features

# Split the data into independent and dependent variables using iloc
X = diabetes_data.iloc[:, :-1]  # Independent variables (features)
y = diabetes_data.iloc[:, -1]  # Dependent variable (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
# Initialize the K-Nearest Neighbors model with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Model Testing
# Predict the target values for the test set
y_pred = knn.predict(X_test_scaled)


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
