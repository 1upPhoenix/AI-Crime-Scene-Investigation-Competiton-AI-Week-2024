# import pandas as pd
# import joblib

# # Load the trained model
# clf = joblib.load(r'C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge5\location_model.joblib')
# print("Model loaded successfully.")

# # Define the feature names
# features = [
#     'Hunger', 'Injury', 'Dehydration', 'Confusion', 'Fatigue', 'Fear', 
#     'Dizziness', 'Sunny', 'Windy', 'Foggy', 'Clear', 'Warm', 'Cold', 
#     'Rain', 'TimeOfDay', 'NearWater'
# ]

# # Location mapping dictionary
# location_map = {0: 'Small Cabin', 1: 'Abandoned Hotel', 2: 'Restaurant', 3: 'Gas Station'}

# def predict_location(user_input):
#     # Ensure input dictionary has all features
#     for feature in features:
#         user_input.setdefault(feature, 0)  # Default missing features to 0
    
#     # Convert input to DataFrame format
#     user_scenario = pd.DataFrame([user_input])

#     # Predict location
#     predicted_location = clf.predict(user_scenario)[0]
#     return location_map.get(predicted_location, "Location Unknown")

# # For interactive use
# if __name__ == "__main__":
#     user_input = {}
#     print("Enter 1 or 0 for each feature:")

#     for feature in features:
#         while True:
#             try:
#                 value = int(input(f"{feature}: "))
#                 if value in [0, 1]:
#                     user_input[feature] = value
#                     break
#                 else:
#                     print("Please enter 1 or 0.")
#             except ValueError:
#                 print("Invalid input. Please enter an integer (1 or 0).")

#     print("Predicted Location:", predict_location(user_input))

# # Example usage with predefined input
# example_input = {'Hunger': 1, 'TimeOfDay': 1, 'Sunny': 1, 'Fatigue': 0}
# print("Predicted Location with example input:", predict_location(example_input))



import pandas as pd
import joblib

# Load the trained model
clf = joblib.load(r'C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge5\location_model.joblib')
print("Model loaded successfully.")

# Define the feature names
features = [
    'Hunger', 'Injury', 'Dehydration', 'Confusion', 'Fatigue', 'Fear', 
    'Dizziness', 'Sunny', 'Windy', 'Foggy', 'Clear', 'Warm', 'Cold', 
    'Rain', 'TimeOfDay', 'NearWater'
]

# Function to predict location based on a dictionary input, enforcing feature order
def predict_location(user_input):
    # Reorder user input to match the feature list
    user_input_ordered = {feature: user_input.get(feature, 0) for feature in features}
    user_scenario = pd.DataFrame([user_input_ordered])


    # Predict location
    predicted_location = clf.predict(user_scenario)[0]
    location_map = {0: 'Small Cabin', 1: 'Abandoned Hotel', 2: 'Restaurant', 3: 'Gas Station'}
    
    return location_map.get(predicted_location, "Location Unknown")

# Function to get user input and predict location
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

# Main loop for continuous user input
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


# Start the main loop
main()


