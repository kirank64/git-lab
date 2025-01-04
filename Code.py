import pandas as pd
import streamlit as st
import os

# Check if nltk is installed
try:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    st.write("NLTK is installed, and required data files are downloaded!")
except ModuleNotFoundError:
    st.error("NLTK is not installed. Please check your dependencies.")
#Display installed packages
st.write("Installed packages:")
os.system("pip freeze")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # For saving the model
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# --- Data Preparation ---
data = {
    'text': [
        "Hi there!", "Hello!", "Good morning", "Good evening", "Hey!",  # Greeting
        "Tell me something funny", "Make me laugh", "Do you know any jokes?", "Tell a joke", "I need a laugh",  # Joke
        "I want to book a flight", "Reserve a table at a restaurant", "Can you book a cab for me?", "I need to book a hotel", "Schedule an appointment",  # Booking
        "What's the weather like today?", "Tell me the weather forecast", "Is it going to rain today?", "What's the temperature outside?", "Check the weather in New York",  # Weather
        "Play some music for me", "Can you play a song?", "I want to listen to something relaxing", "Start the playlist", "Play my favorite song",  # Music
        "Can you assist me?", "I need some help", "Could you help me with this?", "What can you help me with?", "I need assistance",  # Help
        "Find me a place to eat", "Show me restaurants nearby", "Any good restaurants around?", "Where can I have dinner?", "Suggest a good place for lunch",  # Restaurants
        "What time is it?", "Tell me the current time", "Can you check the time?", "What's the time now?", "Give me the time, please",  # Time
        "Goodbye", "See you later", "Bye!", "Talk to you soon", "Catch you later",  # Farewell
        "How's it going?", "What's up?", "How are you feeling?", "What are you up to?", "How's your day?"  # Small Talk
    ],
    'intent': [
        "greeting", "greeting", "greeting", "greeting", "greeting",  # Greeting
        "joke", "joke", "joke", "joke", "joke",  # Joke
        "booking", "booking", "booking", "booking", "booking",  # Booking
        "weather", "weather", "weather", "weather", "weather",  # Weather
        "music", "music", "music", "music", "music",  # Music
        "help", "help", "help", "help", "help",  # Help
        "restaurants", "restaurants", "restaurants", "restaurants", "restaurants",  # Restaurants
        "time", "time", "time", "time", "time",  # Time
        "farewell", "farewell", "farewell", "farewell", "farewell",  # Farewell
        "small_talk", "small_talk", "small_talk", "small_talk", "small_talk"  # Small Talk
    ]
}
df = pd.DataFrame(data)

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Feature Engineering
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['intent']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model Training and Evaluation ---
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Bagging': BaggingClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=10, random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
}

model_accuracies = {}

for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    avg_accuracy = scores.mean()
    model_accuracies[model_name] = avg_accuracy

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# --- Save the Best Model and Vectorizer ---
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")

# --- Deployment with Streamlit ---
def streamlit_code():
    with open("streamlit_app.py", "w") as file:
        file.write('''\
import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words("english")]
    return " ".join(tokens)

# Chatbot response function
def chatbot_response(user_input):
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    intent = model.predict(vectorized_input)[0]
    responses = {
        "greeting": "Hi there! How can I assist you?",
        "joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "booking": "Sure! Let me help you with that booking.",
        "weather": "Today's weather is sunny with a chance of awesomeness!",
        "music": "Playing your favorite tunes!",
        "help": "I'm here to help! What do you need?",
        "restaurants": "Looking for restaurants nearby...",
        "time": "It's time to shine! The current time is... (check your watch!)",
        "farewell": "Goodbye! Have a great day ahead!",
        "small_talk": "I'm just a chatbot, but I'm doing great. Thanks for asking!"
    }
    return responses.get(intent, "I didn't quite catch that. Could you try rephrasing?")

# Streamlit app interface
st.title("Chatbot using NLP")
user_input = st.text_input("You:", placeholder="Type something...")
if user_input:
    st.write(f"Bot: {chatbot_response(user_input)}")
''')

streamlit_code()
