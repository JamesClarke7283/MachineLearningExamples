import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pretrained model
model_path = './models/sentiment_model_LSTM_7e_86acc_0.36loss.keras'
model = load_model(model_path)

# Load the tokenizer
with open('./tokenizers/sentiment_analysis/imdb_lstm_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def encode_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=250, padding='post', truncating='post')

text = "The movie was great! The characters were well-developed and the plot was thrilling.".lower()

# Convert the text into tokenized and padded sequence
encoded_text = encode_text(text)

# Make predictions
prediction = model.predict(encoded_text)

sentiment = "positive" if prediction[0][0] >= 0.5 else "negative"

print(f"Sentiment: {sentiment}, Probability: {prediction[0][0]:.2f}")