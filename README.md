# Multiclass-Fish-Image-Classification
# 🐟 Multiclass Fish Image Classification

A deep learning project for classifying fish images into multiple categories using **Convolutional Neural Networks (CNN)** and **Transfer Learning**. The project also includes a **Streamlit** web application for real-time predictions.

---

## 📌 Features
- **Data Preprocessing & Augmentation** using `ImageDataGenerator`
- **CNN Model** trained from scratch
- **Transfer Learning** with pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)
- **Model Evaluation** with accuracy, precision, recall, F1-score, and confusion matrix
- **Deployment** with a Streamlit web app for real-time fish species prediction
- Supports **11 Fish Categories**:
  - animal fish
  - animal fish bass
  - fish sea_food black_sea_sprat
  - fish sea_food gilt_head_bream
  - fish sea_food hourse_mackerel
  - fish sea_food red_mullet
  - fish sea_food red_sea_bream
  - fish sea_food sea_bass
  - fish sea_food shrimp
  - fish sea_food striped_red_mullet
  - fish sea_food trout

---

---

## ⚙️ Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification
Create and activate virtual environment (optional but recommended)

bash
Copy
Edit
conda create -n fishenv python=3.9
conda activate fishenv
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
📊 Dataset
The dataset contains fish images organized in folders by category and split into:

train/ (training set)

val/ (validation set)

test/ (unseen test set)

Example Structure:

kotlin
Copy
Edit
data/
    train/
        animal fish/
        fish sea_food trout/
        ...
    val/
        ...
    test/
        ...
🚀 Training the Model
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open fish_classifier.ipynb and execute the cells:

Phase 1: Data Preprocessing & Augmentation

Phase 2: CNN Model Training

Phase 3: Model Evaluation

Phase 4: Deployment (optional)

🧪 Testing the Model
Single Image Prediction
python
Copy
Edit
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = "path_to_your_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = list(train_gen.class_indices.keys())[np.argmax(pred)]
print(f"Predicted: {pred_class}")
📈 Evaluation Metrics
The model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Example:

Metric	Score
Accuracy	0.92
Precision	0.91
Recall	0.90
F1-score	0.90

🌐 Deployment
To run the Streamlit app locally:

bash
Copy
Edit
streamlit run streamlit_app/app.py
📜 Requirements
See requirements.txt:

nginx
Copy
Edit
tensorflow
pillow
numpy
matplotlib
seaborn
scikit-learn
streamlit

