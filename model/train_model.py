# model/train_model.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load and preprocess dataset
def load_data(data_dir, num_classes=6):
    data, labels = [], []
    for i in range(num_classes):
        path = os.path.join(data_dir, str(i))
        for img_name in os.listdir(path):
            try:
                img = Image.open(os.path.join(path, img_name)).resize((224, 224))
                data.append(np.array(img) / 255.0)
                labels.append(i)
            except Exception as e:
                print("Error loading image:", e)
    return np.array(data, dtype=np.float32), np.array(labels)

data_dir = 'YOUR_LOCAL_TRAIN_PATH'  # Replace this with your path
data, labels = load_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=32)
y_train = to_categorical(y_train, 6)
y_test = to_categorical(y_test, 6)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True,
                             fill_mode='nearest')
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))

model.save("modeles.h5")
print("✅ Model saved to modeles.h5")

# Save accuracy/loss plots
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("../images/accuracy_plot.png")

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../images/loss_plot.png")
