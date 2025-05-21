# 💵 Blind Assist for Currency Recognition

A deep learning–based system designed to empower visually impaired individuals by helping them recognize currency notes through image recognition and audio feedback.

## 👁️‍🗨️ Overview

Visually impaired individuals often face challenges when dealing with cash transactions, risking fraud or misidentification of currency. This project addresses this issue by developing a CNN-based currency recognition system that integrates voice feedback, promoting independence and safety in financial dealings.

---

## 🧠 Problem Statement

Develop a reliable and efficient solution using machine learning and deep learning techniques for visually impaired people to:
- Recognize currency notes via image input.
- Provide audio feedback for real-time support.

---

## 🎯 Objectives

- Train a CNN model to classify currency notes using a custom dataset.
- Build a mobile application with real-time image capture and voice output.
- Evaluate the system for high classification accuracy and performance.

---

## 🛠️ Methodology

1. **Data Collection & Preprocessing**  
   - Indian currency dataset (2.74k images, ~2 GB).
   - Standardized preprocessing (resizing, normalization, augmentation).

2. **Data Splitting**  
   - Dataset divided into training and testing subsets.

3. **Model Building**  
   - Architecture: Sequential CNN.
   - Layers: Convolutional + MaxPooling + Dense + Dropout.
   - Framework: TensorFlow and Keras.

4. **Model Evaluation**  
   - Metrics: Accuracy, Confusion Matrix, Loss Curve.
   - Achieved **83.03% test accuracy**.

---

## 🚧 Gaps Identified

- Requires internet connectivity (no offline support).
- Cannot detect fake/counterfeit currency.
- Struggles with worn or wrinkled notes.
- Hardware alternatives like banknote readers are bulky and not portable.

---

## 💡 Proposed System

- **Model**: Sequential CNN with 3 convolutional layers and 2 dense layers.
- **Data Augmentation**: Rotation, zoom, shear, and flips using `ImageDataGenerator`.
- **Layers**:
  - Convolutional Layers with Max Pooling
  - Flatten Layer
  - Dense Layers with Dropout to prevent overfitting

---

## 🖼️ Architecture

The architecture follows a standard CNN pipeline:  
`Input → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output`

---

## 📦 Dataset

- **Currency**: Indian Rupee (₹)
- **Images**: 2740 labeled images of various denominations
- **Source**: [Mendeley Data](https://data.mendeley.com/)
- **Format**: PNG
- **Size**: ~2 GB

---

## 📊 Results

- **Test Accuracy**: 83.03%
- Includes:
  - Accuracy graph
  - Loss curve
  - Confusion matrix

---

## 🔚 Conclusion

This project enhances accessibility and inclusivity by enabling blind individuals to identify currency easily and independently. It’s a promising assistive technology for real-world deployment.

---

## 🔭 Future Scope

- ✅ Add **fake currency detection** using image forensics.
- ✅ Develop **offline-capable models**.
- ✅ Integrate with **wearable devices** (e.g., smart glasses).
- ✅ Embed into **financial transaction systems** (POS, ATM).

## 🤝 Connect with Me

📧 rasalharish585@gmail.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/rasalharish/)


