import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the trained model
clf = joblib.load(r'D:\Visual Studio DDrive\python\AI Week Coding Competition\location_model.joblib')
print("Model loaded successfully.")

features = [
    'Hunger', 'Injury', 'Dehydration', 'Confusion', 'Fatigue', 'Fear', 
    'Dizziness', 'Sunny', 'Windy', 'Foggy', 'Clear', 'Warm', 'Cold', 
    'Rain', 'TimeOfDay', 'NearWater'
]

location_map = {0: 'Small Cabin', 1: 'Abandoned Hotel', 2: 'Restaurant', 3: 'Gas Station'}

# Do not edit
def predict_location(user_input):
    user_input_ordered = {feature: user_input.get(feature, 0) for feature in features}
    user_scenario = pd.DataFrame([user_input_ordered])

    predicted_location = clf.predict(user_scenario)[0]
    
    return location_map.get(predicted_location, "Location Unknown")

# Do not edit
def get_user_input():
    user_input = {}
    print("Enter 1 or 0 for each feature:")

    for feature in features:
        while True:
            try:
                value = int(input(f"{feature}: "))
                if value in [0, 1]:
                    user_input[feature] = value
                    break
                else:
                    print("Please enter 1 or 0.")
            except ValueError:
                print("Invalid input. Please enter an integer (1 or 0).")
    
    return user_input

def main():
    while True:
        # Ask if the user wants to make a prediction
        user_choice = input("Would you like to make a prediction? (y/n): ").strip().lower()
        if user_choice == 'y':
            user_input = get_user_input()
            predicted_location = predict_location(user_input)
            print("Predicted Location:", predicted_location)
        elif user_choice == 'n':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 'y' to predict or 'n' to exit.")

main()

# Challenge 5:
# She is in the small cabin due to these weather conditions and her physical conditions according to the location predictor model.

# To conclude, Evelyn was involved with the data breach, and Josh knew about it and kept defending her. After a conversation with Evelyn, Sarah went missing. Indicating Evelyn used a sedative drug on Sarah and left her in the woods (the small cabin), this is because Sarah found out about the breach from Kristy.


# BONUS 

plt.figure(figsize=(10, 8))  
plot_tree(
    clf,
    feature_names=features,
    class_names=list(location_map.values()),
    filled=True  
)

plt.savefig("decision_tree_image.png", format='png', dpi=300) 
plt.show()