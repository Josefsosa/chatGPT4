import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
import numpy as np

# Define the vocabulary of the model
vocab = ['hello', 'world', 'how', 'are', 'you']

# Define the input and output sequences
inputs = ['hello', 'how', 'are', 'you', 'world']
outputs = ['world', 'are', 'you', 'hello', 'how']

# Create a dictionary to map words to indices
word_to_index = {word: index for index, word in enumerate(vocab)}

# Convert input and output sequences to integer sequences
inputs_seq = [word_to_index[word] for word in inputs]
outputs_seq = [word_to_index[word] for word in outputs]

# Create input and output arrays for training the model
inputs_array = np.array(inputs_seq).reshape(1, -1, 1)
outputs_array = np.array(outputs_seq).reshape(1, -1, 1)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=5))
model.add(LSTM(units=32))
model.add(Dense(units=len(vocab), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(inputs_array, outputs_array, epochs=100)

# Test the model
test_input = ['hello', 'how']
test_input_seq = [word_to_index[word] for word in test_input]
test_input_array = np.array(test_input_seq).reshape(1, -1, 1)
predictions = model.predict(test_input_array)
predicted_word_index = np.argmax(predictions, axis=2)
predicted_word = vocab[predicted_word_index[0][0]]

print('Input:', test_input)
print('Output:', predicted_word)
