# datascience
SEVEN7CODE TECHNOLOGIES-DATA SCIENCE CODE
Certainly, here's the complete code including dataset download, preprocessing, and model training:
python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download the dataset from the provided URL
url = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
print("Please download the dataset from:", url)
print("After downloading, save it as 'titanic-dataset.csv' in the same directory as this script.")
input("Press Enter to continue once you've downloaded the dataset...")

# Load the dataset
data = pd.read_csv("titanic-dataset.csv")

# Data preprocessing
data.drop a(subset=['Age', 'Embarked'], inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Selecting features and the target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}"
