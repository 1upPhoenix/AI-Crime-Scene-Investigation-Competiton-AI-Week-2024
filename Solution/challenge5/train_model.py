# train_model.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Define possible scenarios with feature combinations for each location
scenario_data = [
    # Small Cabin scenarios
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 0, 'Confusion': 0, 'Fatigue': 0, 'Fear': 0, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 0},
    {'Hunger': 0, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 0, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 0, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 0},
    #fajr
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 0},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 0},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 0},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 0},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 0},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 0, 'Fear': 1, 'Dizziness': 1, 'Sunny': 0, 'Windy': 0, 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 0},

    # Abandoned Hotel scenarios
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 0, 'Confusion': 0, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    #fajr
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 0, 'Confusion': 0, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 1, 'Warm': 0, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 0, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 0, 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 1},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 0, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 1, 'Warm': 1, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 1},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 0, 'Fear': 0, 'Dizziness': 1, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 1},

    # Restaurant scenarios
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 0, 'Fear': 0, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 1, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2},
    #fajr
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 0, 'Fear': 0, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 1, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 0, 'Fear': 1, 'Dizziness': 0, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 2},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 1, 'Warm': 0, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2}, 
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 1, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 0, 'Warm': 1, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 1, 'Destination': 2},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 0, 'Sunny': 1, 'Windy': 0, 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 0, 'Confusion': 0, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 0, 'Clear': 1, 'Warm': 0, 'Cold': 1, 'Rain': 0, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 0, 'NearWater': 0, 'Destination': 2},

    # Gas Station scenarios
    {'Hunger': 0, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 1, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 1, 'NearWater': 0, 'Destination': 3},
    #fajr
    {'Hunger': 0, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 1, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 1, 'NearWater': 0, 'Destination': 3},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 1, 'NearWater': 1, 'Destination': 3},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 1, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 0, 'Warm': 0, 'Cold': 1, 'Rain': 0, 'TimeOfDay': 1, 'NearWater': 0, 'Destination': 3},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 1, 'Confusion': 0, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 1, 'Foggy': 1, 'Clear': 1, 'Warm': 0, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 1, 'NearWater': 0, 'Destination': 3},
    {'Hunger': 0, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 1, 'Sunny': 1, 'Windy': 1, 'Foggy': 0, 'Clear': 1, 'Warm': 0, 'Cold': 0, 'Rain': 1, 'TimeOfDay': 1, 'NearWater': 1, 'Destination': 3},
    {'Hunger': 1, 'Injury': 0, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 0, 'Fear': 1, 'Dizziness': 1, 'Sunny': 0, 'Windy': 1 , 'Foggy': 1, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 1,'NearWater': 1, 'Destination': 3},
    {'Hunger': 1, 'Injury': 1, 'Dehydration': 0, 'Confusion': 1, 'Fatigue': 1, 'Fear': 0, 'Dizziness': 0, 'Sunny': 1, 'Windy': 0, 'Foggy': 0, 'Clear': 0, 'Warm': 1, 'Cold': 0, 'Rain': 0, 'TimeOfDay': 1, 'NearWater': 1, 'Destination': 3},
    {'Hunger': 0, 'Injury': 1, 'Dehydration': 1, 'Confusion': 1, 'Fatigue': 1, 'Fear': 1, 'Dizziness': 0, 'Sunny': 0, 'Windy': 0, 'Foggy': 1, 'Clear': 1, 'Warm': 0, 'Cold': 1, 'Rain': 1, 'TimeOfDay': 1, 'NearWater': 1, 'Destination': 3}
]



# Create a DataFrame from the scenarios
df_scenarios = pd.DataFrame(scenario_data)

# Split data into features and target
X = df_scenarios.drop(columns=['Destination'])
y = df_scenarios['Destination']

# Train a decision tree classifier on the scenarios
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Save the trained model to a file
joblib.dump(clf, 'location_model.joblib')
print("Model trained and saved as 'location_model.joblib'")
