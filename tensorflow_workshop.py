
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Sample dataset
reviews = ["Very bad product", "This product is great!", "I hated it", "Best purchase ever", "Worst quality", "Absolutely love it!", "This product is amazing", "So disappointed", "Would not recommend", "Best decision ever", "Fantastic product!", "Really bad", "Terrible", "I enjoy using it!", "Not a great product"]
labels = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Converting labels  NumPy array
labels = np.array(labels, dtype=np.float32)


# Tokenize text
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=5)

print("Input shape:", padded_sequences.shape)
print("Labels shape:", labels.shape)
print("Padded Sequences: ", padded_sequences)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 16, input_length=5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, labels, epochs=150)

# Test with a new review
new_reviews = ["I recommend this product", "I love it", "I do not like it"]
new_seq = tokenizer.texts_to_sequences(new_reviews)
padded_new_seq = pad_sequences(new_seq, maxlen=5)
prediction = model.predict(padded_new_seq)
for i in range(len(new_reviews)):
  print(f"Sentiment: {'Positive' if prediction[i][0] > 0.5 else 'Negative'}")