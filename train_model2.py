import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
#deleate corupterd data
df = pd.read_csv("datafoeAI1.csv", on_bad_lines='skip', engine='python')

# laod data
#limit the data to what we need 
drowsiness_labels = ['Normal', 'Tired', 'Very Tired']
df = df[df['Label'].isin(drowsiness_labels)]

# Drop rows with missing values
df = df.dropna(subset=[
    'Label', 'Blink_Rate', 'Avg_Blink_Duration', 'EAR_Mean', 'EAR_Variance', 'Longest_Eye_Closure'
])

# labels
features = df[[
    "Blink_Rate", "Avg_Blink_Duration", "EAR_Mean", "EAR_Variance", "Longest_Eye_Closure"
]].values

labels = df["Label"].values

# ENCODE LABELS (so ai understand)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# split the data into 8-2 ratio to train and test
X_train, X_test, y_train, y_test = train_test_split(
    features, y_encoded, test_size=0.2, random_state=42
)

# normalizing input this is needed for some reason
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# make acteull model
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

# train the thing
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# test accuracy
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 9. SAVE MODEL and the stuff that it does
model.save("blink_ear_aggregate_model.keras")
np.save("blink_ear_label_classes.npy", label_encoder.classes_)
np.save("blink_ear_scaler_mean.npy", scaler.mean_)
np.save("blink_ear_scaler_scale.npy", scaler.scale_)

#Test Accuracy: 81.82%% (sheeeeeeeeeesh crazy)
