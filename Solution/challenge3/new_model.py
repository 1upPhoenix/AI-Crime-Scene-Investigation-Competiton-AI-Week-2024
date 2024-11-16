from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib

# Load and preprocess the data
file_path = "C:\\Users\\ahmed\\OneDrive\\Desktop\\AIQU\\AI Crime Scene Investigation\\challenge3\\online_activity3.csv"
data = pd.read_csv(file_path)

# Basic feature engineering
data['day'] = pd.to_datetime(data['date']).dt.day
data['month'] = pd.to_datetime(data['date']).dt.month
data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Custom flag for keywords related to safety/escape
keywords = ["escape", "safety", "paranoia", "trust", "trapped", "emergency"]
data['safety_related'] = data['search_term'].apply(lambda x: 1 if any(word in x.lower() for word in keywords) else 0)

# Separate the label column
y = data['label']

# Apply TF-IDF to search terms
tfidf_vectorizer = TfidfVectorizer()
search_term_tfidf = tfidf_vectorizer.fit_transform(data['search_term'])

# Convert TF-IDF matrix to DataFrame and merge with other features
search_term_df = pd.DataFrame(search_term_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
data = pd.concat([data[['day', 'month', 'day_of_week', 'is_weekend', 'safety_related']], search_term_df], axis=1)

# Define features and labels for model training
X = data  # Features
y = y  # Labels

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up Gradient Boosting with expanded hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}

gb_classifier = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predictions and accuracy with a lower threshold
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.4  # Lower threshold for anomaly detection sensitivity
y_pred = (y_pred_proba >= threshold).astype(int)

# Classification report for better evaluation
print(classification_report(y_test, y_pred))

# Save the model and encoders
joblib.dump(best_model, 'anomaly_detector.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
