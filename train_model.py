import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import os

#ignoRE CORUPT SUFFFFFFF
df = pd.read_csv("datafoeAI1.csv", on_bad_lines='skip', engine='python')
gaze_labels = ['Left', 'Right', 'Up', 'Down', 'Center']
df = df[df['Label'].isin(gaze_labels)]
#this is a very very simnple ai using ML did not have time to use CNN
# 1. LOAD CSV DATA
df = df.dropna()

# Input features
featuers_needed = df[["iris_x_norm", "iris_y_norm", "eye_width", "eye_height"]].values
#filted labels
# Output labels
labels_putputted= df["Label"].values

# 2. ENCODE LABELS
encode_for_x = LabelEncoder()
y_encoded = encode_for_x.fit_transform(labels_putputted)

# 3. TRAIN/TEST SPLIT ( now splits the features, not the encoder)
X_train, X_test, y_train, y_test = train_test_split(
    featuers_needed, y_encoded, test_size=0.2, random_state=42
)

# 4. NORMALIZE INPUTS
scaler = StandardScaler()
X_train_yes = scaler.fit_transform(X_train)
X_test_no = scaler.transform(X_test)

# 5. BUILD MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)), 
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),  
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(encode_for_x.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#cus cool
history = model.fit(
    X_train_yes, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

#EVALUATE cuz cooool
loss, acccc = model.evaluate(X_test_no, y_test)
print(f"\nTest Accuracy: {acccc*100:.2f}%")

# 8. SAVE MODEL + LABELS + SCALER+cooool
model.save("gaze_direction_model.keras")
np.save("gaze_label_classes.npy", encode_for_x.classes_)
np.save("gaze_scaler_mean.npy", scaler.mean_)
np.save("gaze_scaler_scale.npy", scaler.scale_)

#80% accuracy sheeesh
