
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Convert text to numbers
data['parent_edu'] = data['parent_edu'].map({'High School':0, 'Bachelor':1, 'Master':2})
data['gender'] = data['gender'].map({'Male':0, 'Female':1})

# Features and target
X = data[['study_hours', 'attendance', 'parent_edu', 'gender']]
y = data['grade']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to use in app.py
with open('model/student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
