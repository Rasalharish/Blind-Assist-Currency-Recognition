# model/predict_sample.py

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("modeles.h5")
img_path = "sample.jpg"  # Replace with actual image path

img = Image.open(img_path).resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

label_map = {
    0: '10 rupee',
    1: '20 rupee',
    2: '50 rupee',
    3: '100 rupee',
    4: '200 rupee',
    5: '500 rupee'
}

print("Predicted:", label_map.get(predicted_class, "Unknown"))

plt.imshow(img)
plt.title(f"Prediction: {label_map.get(predicted_class)}")
plt.axis('off')
plt.show()
