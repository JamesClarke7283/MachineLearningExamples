import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Define the path to the dataset and the output file
INPUT_FILE = './datasets/raw/imdb_reviews.csv'
OUTPUT_FILE = './datasets/preprocessed/preproc_imdb_reviews.csv'

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<br />", " ", text)  # Remove <br /> tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    return text

# Load the dataset
data = pd.read_csv(INPUT_FILE)

# Clean the reviews
data['text'] = data['text'].apply(clean_text)

# Tokenize the reviews
vocab_size = 10000
max_length = 250

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad the sequences
padded_data = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Convert the preprocessed data to a DataFrame
preprocessed_data = pd.DataFrame({
    'text': [' '.join(map(str, seq)) for seq in padded_data],
    'sentiment': data['sentiment']
})

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv(OUTPUT_FILE, index=False)

TOKENIZER_FILE = './tokenizers/sentiment_analysis/imdb_lstm_tokenizer.pickle'
with open(TOKENIZER_FILE, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
