# Pneumonia Detection Using Deep Learning

An AI-powered application that analyzes chest X-ray images and predicts whether an image is classified as **Normal** or **Pneumonia** using a trained deep learning model.

## Overview

This project uses deep learning and computer vision to classify chest X-ray images. It provides an interactive interface built with Streamlit, allowing users to upload an X-ray image and view the model's prediction.

## Features

- Upload chest X-ray images.
- Automated image preprocessing and resizing.
- Deep learning-based image classification.
- Prediction confidence display.
- Interactive Streamlit interface.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Deep Learning
- Convolutional Neural Networks (CNN)

## Project Structure

```text
Pneumonia_Detection_App/
├── app.py
├── models/
├── assets/
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone the repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Navigate to the project folder:

   ```bash
   cd Pneumonia_Detection_App
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Disclaimer

This project is developed for educational and demonstration purposes. Its predictions are not a medical diagnosis and should not replace professional medical evaluation.

## Author

Developed as an academic deep learning project.
