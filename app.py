from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    cleaned_text = ' '.join(tokens)
    return cleaned_text

app = Flask(__name__)

# Load your pre-trained model
model = load_model(r"C:\Users\dream\OneDrive\Desktop\Fake_news\Model\my_model.h5")

# Pre-defined max length for padding
max_length = 50

# Define the API route
@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.get_json()
    data = json_data.get("text")

    # Clean the text input
    new_data = clean_text(data)

    # Tokenization and sequence padding
    tokenizer = Tokenizer()  # Load a pre-trained tokenizer if available
    tokenizer.fit_on_texts([new_data])
    sequence = tokenizer.texts_to_sequences([new_data])
    padded_sequence = pad_sequences(sequence, padding="pre", maxlen=max_length)

    # Make predictions
    prediction = model.predict(padded_sequence)
    binary_prediction = np.round(prediction).astype(int)

    # Return the response
    response = {"prediction": int(binary_prediction[0][0])}
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
