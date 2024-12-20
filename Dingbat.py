import nltk
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define some predefined patterns and responses (rule-based)
pairs = [
    (r"Hi|hi|Hello|hello|Hey|hey", ["Hello!", "Hey there!", "Hi!"]),
    (r"How are you?", ["I'm good, thank you!", "I'm doing well, how about you?"]),
    (r"What is your name?", ["I'm Dingbat!"]),
    (r"quit", ["Bye! Have a great day!"]),
]

# Sample training data for machine learning (questions and their categories)
training_sentences = [
    "Hello",
    "Hi",
    "How are you?",
    "What is your name?",
    "Goodbye",
    "Bye",
    "Can you help me with the weather?",
    "Tell me about the weather",
    "What is the temperature today?",
    "How is the weather in New York?",
]

training_labels = [
    "greeting",  # Hello, Hi, How are you?
    "greeting",
    "greeting",
    "greeting",
    "goodbye",   # Goodbye, Bye
    "goodbye",
    "weather",   # Weather-related questions
    "weather",
    "weather",
    "weather",
]

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
classifier = MultinomialNB()
classifier.fit(X_train, training_labels)

# Function to predict intent using the machine learning model
def predict_intent(user_input):
    user_input_vector = vectorizer.transform([user_input])
    intent = classifier.predict(user_input_vector)
    return intent[0]

# Function to process the conversation
def Dingbat():
    print("Hi! I'm Dingbat! Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # First check with rule-based matching
        for pattern, responses in pairs:
            if nltk.re.search(pattern, user_input):
                print("Dingbat:", np.random.choice(responses))
                break
        else:
            # If no match is found in rule-based, predict using ML
            intent = predict_intent(user_input)
            if intent == "greeting":
                print("Dingbat: Hello! How can I assist you today?")
            elif intent == "goodbye":
                print("Dingbat: Goodbye! Have a great day!")
                break
            elif intent == "weather":
                print("Dingbat: I can help with weather information. What location are you interested in?")
            else:
                print("Dingbat: I'm sorry, I didn't understand that.")

if __name__ == "__main__":
    Dingbat()