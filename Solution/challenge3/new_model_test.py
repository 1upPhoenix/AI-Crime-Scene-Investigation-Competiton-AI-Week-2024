import pandas as pd
import joblib
import numpy as np

# Load the model, scaler, and TF-IDF vectorizer
anomaly_detector = joblib.load('anomaly_detector.pkl')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Keywords related to safety or distress
keywords = ["escape", "safety", "paranoia", "trust", "trapped", "emergency"]

# Function for Real-Time Testing
def analyze_new_activity():
    print("Enter a search term to test against Sarah's known behavior patterns.")
    
    while True:
        # Take search query input
        search_query = input("Enter search query (e.g., 'how to bake brownies') or type 'exit' to quit: ").lower()
        if search_query == 'exit':
            print("Exiting the program.")
            break

        # Create TF-IDF features for the new search term
        search_term_tfidf = tfidf_vectorizer.transform([search_query])
        tfidf_features = pd.DataFrame(search_term_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Date-based features
        day = pd.to_datetime('now').day
        month = pd.to_datetime('now').month
        day_of_week = pd.to_datetime('now').dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0

        # Create safety-related flag
        safety_related = 1 if any(word in search_query for word in keywords) else 0

        # Combine all features into a DataFrame
        new_data_point = pd.concat(
            [
                pd.DataFrame([[day, month, day_of_week, is_weekend, safety_related]],
                             columns=['day', 'month', 'day_of_week', 'is_weekend', 'safety_related']),
                tfidf_features
            ],
            axis=1
        )

        # Ensure the feature order matches that of the training data
        new_data_point = new_data_point.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale the features
        new_data_point_scaled = scaler.transform(new_data_point)

        # Predict Normal/Anomaly for the new data point
        is_anomaly = anomaly_detector.predict(new_data_point_scaled)[0] == 1  # Assuming 1 means anomaly

        # Display Feedback
        if is_anomaly:
            print("This search query deviates from Sarah's typical behavior.")
        else:
            print("This search query aligns with Sarah's known behavior.")

# Run the function to test the model interactively
analyze_new_activity()
