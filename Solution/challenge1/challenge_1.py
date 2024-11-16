import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy

# Load the English model for Named Entity Recognition (NER)
nlp = spacy.load('en_core_web_sm')

# Load the CSV file from the specified path with defined column names
file_path = r'C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge1\testimonies.csv'
df = pd.read_csv(file_path, header=None, names=['Name', 'Statement'])

# Combine all columns into a single 'Testimony' column
df['Testimony'] = df.apply(lambda row: row['Statement'], axis=1)

# Vectorize the text using TF-IDF for keyword extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Testimony'])

# Extract top 5 keywords for each testimony
def extract_top_keywords(tfidf_matrix, feature_names, top_n=5):
    keywords_list = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]  # Get top N indices
        top_keywords = [feature_names[index] for index in top_indices]
        keywords_list.append(top_keywords)
    return keywords_list

# Get the top keywords per testimony
feature_names = vectorizer.get_feature_names_out()
df['Keywords'] = extract_top_keywords(X, feature_names)

# Sentiment Analysis using TextBlob
df['Sentiment'] = df['Testimony'].apply(lambda x: TextBlob(x).sentiment)

# Named Entity Recognition using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['Entities'] = df['Testimony'].apply(extract_entities)

def detect_lying(sentiment):
    if sentiment.polarity < -0.18:
        return "Potentially Lying"
    elif -0.18 <= sentiment.polarity <= 0.2:
        return "Neutral"
    else:
        return "Likely True"

# Update the 'Potential_Lying' column to reflect these categories
df['Lie_Detection_Status'] = df['Sentiment'].apply(detect_lying)

# Adjust display function to reflect the new 'Lie_Detection_Status' label
def display_testimony_info(selected_indices):
    for index in selected_indices:
        row = df.iloc[index]
        # Display name and statement separately
        print(f"\nName: {row['Name']}")  # Name of the person giving the statement
        print(f"Statement: {row['Statement']}")  # The statement itself
        print(f"Keywords: {row['Keywords']}")
        print(f"Sentiment: {row['Sentiment']}")
        print(f"Entities: {row['Entities']}")
        print(f"Lie Detection Status: {row['Lie_Detection_Status']}")

# User Input to Select Testimonies
def user_select_testimonies():
    while True:
        print("\nAvailable testimonies:")
        for i in range(len(df)):
            print(f"{i}: {df['Statement'].iloc[i][:50]}...")  # Show first 50 characters of the statement

        selected_indices = input("Enter the indices of the testimonies you want to see (comma-separated), or 'exit' to quit: ")
        if selected_indices.lower() == 'exit':
            print("Exiting the program.")
            break

        try:
            selected_indices = [int(index.strip()) for index in selected_indices.split(',')]
            display_testimony_info(selected_indices)
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid indices.")

# Run the user selection function
user_select_testimonies()
