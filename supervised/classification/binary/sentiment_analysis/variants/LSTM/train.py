import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from metrics import metrics

# Load preprocessed data
data_path = './datasets/preprocessed/preproc_imdb_reviews.csv'
data = pd.read_csv(data_path)

# Split data into training and test sets
data_train, data_test, label_train, label_test = train_test_split(
    data['text'].apply(lambda x: list(map(int, x.split()))), 
    data['sentiment'], 
    test_size=0.2, 
    random_state=42
)

# Define constants
VOCAB_SIZE = 10000
MAX_LENGTH = 250

# Convert data into a format suitable for training
data_train = pad_sequences(data_train, padding='post', maxlen=MAX_LENGTH)
data_test = pad_sequences(data_test, padding='post', maxlen=MAX_LENGTH)

# Create the model
model = Sequential([
    Embedding(VOCAB_SIZE, 16, input_length=MAX_LENGTH),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(data_train, label_train, epochs=7, batch_size=512, validation_split=0.2)

model_name = 'LSTM_7e'

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(data_test, label_test)

# Convert the accuracy to a whole number percentage
accuracy_percentage = int(test_accuracy * 100)

# Format the loss up to two decimal places
formatted_loss = "{:.2f}".format(test_loss)

# Construct the filename
filename = f'{model_name}_{accuracy_percentage}acc_{formatted_loss}loss'

# Save the model with the constructed filename
model.save(f'./models/sentiment_model_{filename}.keras')

# Visualize the metrics
metrics(history, model_name)
