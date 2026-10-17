import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---- CONFIG (must match collect_data.py) ----
SIGNS = ['hello', 'thank_you', 'yes', 'no', 'please']
SEQUENCES_PER_SIGN = 30
SEQUENCE_LENGTH = 30
DATA_PATH = 'data'
# ----------------------------------------------

label_map = {sign: idx for idx, sign in enumerate(SIGNS)}

sequences, labels = [], []

for sign in SIGNS:
    for seq_num in range(SEQUENCES_PER_SIGN):
        file_path = os.path.join(DATA_PATH, sign, f'{seq_num}.npy')
        window = np.load(file_path)
        sequences.append(window)
        labels.append(label_map[sign])

X = np.array(sequences)          # shape: (num_samples, 30, 126)
y = to_categorical(labels).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(SIGNS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.save('sign_model.h5')
print("Model saved as sign_model.h5")

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")