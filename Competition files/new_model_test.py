import pandas as pd
import joblib

# Load the model, scaler, and TF-IDF vectorizer
anomaly_detector = joblib.load('anomaly_detector.pkl')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Do not edit
keywords = ["escape", "safety", "paranoia", "trust", "trapped", "emergency"]

# Do not edit
def analyze_new_activity():
    print("Enter a search term to test against Sarah's known behavior patterns.")
    
    while True:
        
        search_query = input("Enter search query (e.g., 'how to bake brownies') or type 'exit' to quit: ").lower()
        if search_query == 'exit':
            print("Exiting the program.")
            break

        search_term_tfidf = tfidf_vectorizer.transform([search_query])
        tfidf_features = pd.DataFrame(search_term_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        day = pd.to_datetime('now').day
        month = pd.to_datetime('now').month
        day_of_week = pd.to_datetime('now').dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0

        safety_related = 1 if any(word in search_query for word in keywords) else 0

        new_data_point = pd.concat(
            [
                pd.DataFrame([[day, month, day_of_week, is_weekend, safety_related]],
                             columns=['day', 'month', 'day_of_week', 'is_weekend', 'safety_related']),
                tfidf_features
            ],
            axis=1
        )

        new_data_point = new_data_point.reindex(columns=scaler.feature_names_in_, fill_value=0)

        new_data_point_scaled = scaler.transform(new_data_point)

        is_anomaly = anomaly_detector.predict(new_data_point_scaled)[0] == 1  # Assuming 1 means anomaly

        if is_anomaly:
            print("This search query deviates from Sarah's typical behavior.")
        else:
            print("This search query aligns with Sarah's known behavior.")

analyze_new_activity()
