# 🧠 COVID-19 X-Ray Classification using Deep Learning & Machine Learning

Final Project – CPS 4801  
Kean University  

This project implements multiple approaches for classifying chest X-ray images as **COVID-19 positive** or **Normal** using:

- Transfer Learning (ResNet50)
- Feature Extraction + Random Forest
- Image Preprocessing with Adaptive Thresholding
- GPU acceleration (Google Colab T4)

---

# 📊 Project Overview

The goal of this project was to compare deep learning and classical machine learning approaches for medical image classification.

We implemented three models:

1️⃣ ResNet50 Transfer Learning (RGB images)  
2️⃣ ResNet50 Feature Extraction + Random Forest  
3️⃣ ResNet50 with Adaptive Thresholding preprocessing  

Dataset used:
- COVID X-ray images (Positive cases)
- Control X-ray images (Normal cases)

Data split:
- 80% Training
- 20% Testing

---

# 🧠 Model 1 – ResNet50 Transfer Learning

### Approach:
- Pretrained ResNet50 (ImageNet weights)
- Final fully connected layer modified for binary classification
- CrossEntropyLoss
- Adam optimizer
- 3 epochs

### Results:

Accuracy: **90%**

```
Non-COVID Precision: 1.00
COVID Precision:     0.82
Overall Accuracy:    0.90
```

---

# 🌲 Model 2 – Feature Extraction + Random Forest

### Approach:
- Remove final layer of ResNet50
- Extract deep features
- Flatten feature vectors
- Train RandomForestClassifier (100 trees)

### Results:

Accuracy: **78%**

```
Overall Accuracy: 0.78
```

This demonstrates the performance difference between:
- End-to-end deep learning
- Feature extraction + classical ML

---

# 🧪 Model 3 – Adaptive Thresholding + ResNet50

### Preprocessing:
- Convert image to grayscale
- Apply adaptive thresholding (OpenCV)
- Convert back to RGB
- Resize to 224x224

### Model Architecture:
- ResNet50 backbone
- Added:
  - Linear layer (256 neurons)
  - ReLU
  - Dropout (0.3)
  - Final binary classification layer

### Results:

Accuracy: **90%**

```
Normal Precision: 0.85
COVID Precision:  0.95
Overall Accuracy: 0.90
```

This experiment tested whether preprocessing improves classification performance.

---

# 🏗️ Architecture Overview

```
Dataset
   ↓
Preprocessing (Resize / Thresholding)
   ↓
ResNet50 (Transfer Learning)
   ↓
Binary Classification
   ↓
Evaluation (Precision / Recall / F1)
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- scikit-learn
- Google Colab (T4 GPU)
- PIL
- tqdm

---

# 📂 Project Structure

```
Final-Project-CPS-4801-01/
│
└── Code_for_Final_Project.ipynb
```

---

# ▶️ How to Run

This project was developed in **Google Colab**.

### Steps:

1. Upload notebook to Colab
2. Enable GPU:
   Runtime → Change Runtime Type → GPU
3. Mount Google Drive
4. Ensure dataset folder structure:

```
MyDrive/
 └── Cov19 Dataset/
      ├── Early Cov/
      └── Control/
```

5. Run all cells

---

# 📈 Evaluation Metrics

Models evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report (sklearn)

---

# 🧠 Key Concepts Demonstrated

- Transfer Learning
- Convolutional Neural Networks
- Feature Extraction
- Classical ML comparison
- Image preprocessing techniques
- GPU acceleration
- Model evaluation
- Binary classification
- Medical image analysis

---

# 🎯 Learning Outcomes

This project demonstrates:

- Ability to implement pretrained CNN architectures
- Understanding of model fine-tuning
- Knowledge of ML vs DL tradeoffs
- Image preprocessing experimentation
- Real-world dataset handling
- Evaluation using scientific metrics

---

# 🔮 Future Improvements

- Cross-validation
- ROC curve analysis
- Confusion matrix visualization
- Hyperparameter tuning
- Larger dataset
- Data augmentation
- Deployment via web interface
- Model explainability (Grad-CAM)

---

# 💼 Why This Project Is Portfolio-Strong

This project shows:

✔ Deep Learning experience  
✔ PyTorch proficiency  
✔ GPU usage  
✔ ML experimentation  
✔ Model comparison analysis  
✔ Medical imaging application  
✔ Research-oriented thinking  

Highly relevant for:

- Machine Learning Internships
- AI Research roles
- Computer Vision positions
- Data Science internships
- AI in Healthcare roles

---

# 👤 Authors

Angel Picon  
Leonel Torres Garcia  

Kean University – CPS 4801  

---

# ⚠️ Disclaimer

This project is for educational and research purposes only.  
It is not intended for medical diagnosis.
