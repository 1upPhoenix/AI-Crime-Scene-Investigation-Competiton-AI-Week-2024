import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy

# Do not edit
nlp = spacy.load('en_core_web_sm')

file_path = r'Testimonies file.csv'

# DO not edit
df = pd.read_csv(file_path, header=None, names=['Name', 'Statement'])
df['Testimony'] = df.apply(lambda row: row['Statement'], axis=1)

# DO not edit
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Testimony'])

# Do not edit
def extract_top_keywords(tfidf_matrix, feature_names, top_n=5):
    keywords_list = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]  # Get top N indices
        top_keywords = [feature_names[index] for index in top_indices]
        keywords_list.append(top_keywords)
    return keywords_list

feature_names = vectorizer.get_feature_names_out()
df['Keywords'] = extract_top_keywords(X, feature_names)

df['Sentiment'] = df['Testimony'].apply(lambda x: TextBlob(x).sentiment)

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['Entities'] = df['Testimony'].apply(extract_entities)

# Do not edit
def detect_lying(sentiment):
    if sentiment.polarity < -0.18:
        return "Potentially Lying"
    elif -0.18 <= sentiment.polarity <= 0.2:
        return "Neutral"
    else:
        return "Likely True"

df['Lie_Detection_Status'] = df['Sentiment'].apply(detect_lying)

# Do not edit
def display_testimony_info(selected_indices):
    for index in selected_indices:
        row = df.iloc[index]
        print(f"\nName: {row['Name']}")  
        print(f"Statement: {row['Statement']}")  
        print(f"Keywords: {row['Keywords']}")
        print(f"Sentiment: {row['Sentiment']}")
        print(f"Entities: {row['Entities']}")
        print(f"Lie Detection Status: {row['Lie_Detection_Status']}")

# Do not edit
def user_select_testimonies():
    while True:
        print("\nAvailable testimonies:")
        for i in range(len(df)):
            print(f"{i}: {df['Statement'].iloc[i][:50]}...") 

        selected_indices = input("Enter the indices of the testimonies you want to see (comma-separated), or 'exit' to quit: ")
        if selected_indices.lower() == 'exit':
            print("Exiting the program.")
            break

        try:
            selected_indices = [int(index.strip()) for index in selected_indices.split(',')]
            display_testimony_info(selected_indices)
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid indices.")

user_select_testimonies()




# After anaylysing whos lying:

# Suspect List: 
# 1- Evelyn
# 2- Josh
# 3- Ethan
# 4- Kristy

# Josh and Evelyn have been constantly lying, and sounding very suspicious.