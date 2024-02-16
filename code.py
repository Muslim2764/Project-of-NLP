#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout


df = pd.read_csv("isear.csv")

# Drop unnecessary columns
df = df.drop(columns=["ID"])  # Assuming "ID" column is unnecessary

# Display the first few rows of the dataset
print(df.head())

# Data Analysis and Visualization

# Distribution of sentiments
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['content'])
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
X = tokenizer.texts_to_sequences(df['content'])
X_pad = pad_sequences(X, maxlen=maxlen)

# Encoding sentiments
encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Model Architecture
model = Sequential([
    Embedding(vocab_size, 100, input_length=maxlen),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(np.max(y) + 1, activation='softmax')  # Adjust output units based on number of unique sentiments
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




