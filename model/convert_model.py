# model/convert_model.py

import tensorflow as tf

model = tf.keras.models.load_model("modeles.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modeles.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite: modeles.tflite")
