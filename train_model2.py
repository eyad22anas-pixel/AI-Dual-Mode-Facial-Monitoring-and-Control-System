import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# 1. LOAD CSV DATA
df = pd.read_csv("datafoeAI1.csv")

# Drop rows with missing values in the relevant columns (features + label)
df = df.dropna(subset=[
    'Label', 'Blink_Rate', 'Avg_Blink_Duration', 'EAR_Mean', 'EAR_Variance', 'Longest_Eye_Closure'
])

# 2. SELECT FEATURES AND LABELS
features = df[[
    "Blink_Rate", "Avg_Blink_Duration", "EAR_Mean", "EAR_Variance", "Longest_Eye_Closure"
]].values

labels = df["Label"].values

# 3. ENCODE LABELS
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    features, y_encoded, test_size=0.2, random_state=42
)

# 5. NORMALIZE INPUTS
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. BUILD MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. TRAIN MODEL
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 8. EVALUATE MODEL
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 9. SAVE MODEL, LABELS, SCALER
model.save("blink_ear_aggregate_model.keras")
np.save("blink_ear_label_classes.npy", label_encoder.classes_)
np.save("blink_ear_scaler_mean.npy", scaler.mean_)
np.save("blink_ear_scaler_scale.npy", scaler.scale_)

#Test Accuracy: 100% (sheeeeeeeeeesh crazy)