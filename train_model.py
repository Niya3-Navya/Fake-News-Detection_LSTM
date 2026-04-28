

# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# import re

# # 1. Cleaning Function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'^.*?\(reuters\) - ', '', text) 
#     text = re.sub(r'[^a-z ]', '', text)
#     return text

# # Load data
# fake = pd.read_csv("Fake.csv")
# true = pd.read_csv("True.csv")

# fake['label'] = 0
# true['label'] = 1

# df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# df['text'] = df['text'].apply(clean_text)

# X = df['text']
# y = df['label']

# max_words = 10000
# max_len = 300 
# tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
# tokenizer.fit_on_texts(X)

# X_seq = tokenizer.texts_to_sequences(X)
# X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')

# X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# # 3. Enhanced LSTM Model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(max_words, 128, input_length=max_len),
#     tf.keras.layers.SpatialDropout1D(0.2),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # --- YAHAN CHANGE HAI ---
# # History variable mein training data save karein
# history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

# # Model aur Tokenizer save karein
# model.save("my_model.h5")
# with open('tokenizer.pkl', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Training History ko save karein (Graph ke liye)
# with open('train_history.pkl', 'wb') as f:
#     pickle.dump(history.history, f)

# print("✅ Model, Tokenizer, and History saved successfully!")



import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Cleaning Function
def clean_text(text):
    text = text.lower()
    # Reuters header hatane ke liye (e.g., "WASHINGTON (Reuters) - ")
    text = re.sub(r'^.*?\(reuters\) - ', '', text) 
    text = re.sub(r'[^a-z ]', '', text)
    return text

# Load data
# Ensure Fake.csv and True.csv are in the same folder
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 0
true['label'] = 1

# Data ko combine aur shuffle karna
df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Text cleaning apply karein
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# 2. Tokenizer Setup
max_words = 10000
max_len = 300 
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')

# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2)

# 3. Enhanced LSTM Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 128, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
# History variable training metrics ko store karta hai
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

# --- CONFUSION MATRIX LOGIC ---
print("Generating Confusion Matrix...")
# Test data par predictions nikalna
y_pred = (model.predict(X_test) > 0.5).astype("int32")
# Matrix calculate karna
cm = confusion_matrix(y_test, y_pred)

# --- SAVING ALL ASSETS ---
# 1. Model save karein
model.save("my_model.h5")

# 2. Tokenizer save karein
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Training History save karein (Line Graph ke liye)
with open('train_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# 4. Confusion Matrix save karein (Report ke liye)
with open('confusion_matrix.pkl', 'wb') as f:
    pickle.dump(cm, f)

print("✅ Model, Tokenizer, History, and Confusion Matrix saved successfully!")