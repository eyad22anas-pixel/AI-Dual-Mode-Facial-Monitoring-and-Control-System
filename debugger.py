import pandas as pd
#i used these to uynderstand the bugs dont mind them
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

model_path = "gaze_direction_model.keras"

print("Current working directory:", os.getcwd())

if os.path.isfile(model_path):
    print(f"Model file '{model_path}' found. Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
else:
    print(f"Model file '{model_path}' NOT found.")
    print("Make sure the file is in the current working directory or provide the full path.")