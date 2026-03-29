import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Simple LSTM model (lightweight, compatible)
model = Sequential()
model.add(Embedding(input_dim=120000, output_dim=64, input_length=200))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Save model
model.save("my_model.h5")

print("✅ my_model.h5 created successfully")