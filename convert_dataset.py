import pandas as pd
import os
import cv2
import numpy as np

# Path to the FER-2013 dataset CSV file
csv_path = r'C:\Users\lenovo\Documents\BBC Projects\Emotion_detection\fer2013'

# Create directories for train and validation datasets
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

for emotion in range(7):
    os.makedirs(f'data/train/{emotion}', exist_ok=True)
    os.makedirs(f'data/validation/{emotion}', exist_ok=True)

# Load the dataset
data = pd.read_csv(csv_path)

# Split the data into training and validation sets and save the images
for index, row in data.iterrows():
    emotion = row['emotion']
    img = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if index % 5 == 0:
        cv2.imwrite(f'data/validation/{emotion}/{index}.jpg', img)
    else:
        cv2.imwrite(f'data/train/{emotion}/{index}.jpg', img)
