# Retinal Eye Disease Prediction 👁️

## Executive Summary
A production-oriented deep learning project that classifies retinal OCT images into four categories — **CNV, DME, Drusen, and Normal** — using a lightweight transfer-learning backbone (MobileNet) and Keras preprocessing. The solution is designed for reproducibility, model interpretability, and potential deployment in clinical screening or telemedicine workflows.

## Overview
This project is a deep learning–based retinal disease classification system that analyzes retinal OCT (Optical Coherence Tomography) images and automatically detects whether the retina is healthy or affected by one of three major retinal conditions:

- **CNV (Choroidal Neovascularization)**
- **DME (Diabetic Macular Edema)**
- **Drusen**
- **Normal (healthy retina)**

The system leverages **MobileNet** as a lightweight yet effective CNN architecture and uses **Keras preprocessing utilities** to build a robust training pipeline.

## Key Highlights
- **Model Architecture:** MobileNet (ImageNet pretrained, fine-tuned for this task)
- **Preprocessing:** Image resizing, normalization, and augmentation using Keras preprocessing utilities (e.g., `tf.keras.preprocessing` / `tf.keras.layers.Resizing`, `Rescaling`, `RandomFlip`, `RandomRotation`)
- **Classification:** Four categories — CNV, DME, Drusen, Normal
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, and per-class metrics
- **Reproducibility:** Full pipeline implemented in a single Jupyter Notebook (`Training_model.ipynb`)

## Technologies Used
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras (MobileNet, Keras preprocessing)
- **Data Handling:** NumPy, Pandas
- **Image Processing:** OpenCV (optional for additional image ops)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## Dataset
The project uses a retinal OCT dataset containing labeled images for the four categories: CNV, DME, Drusen, and Normal. Images are preprocessed with Keras utilities (resizing, normalization, augmentation).  
*(Dataset source example: Kermany et al., OCT Images - Kaggle)*

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Retinal-Eye-Disease-Prediction.git
cd Retinal-Eye-Disease-Prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch Jupyter Notebook and open:
```bash
jupyter notebook Training_model.ipynb
```
4. Run the notebook cells to preprocess data, train MobileNet, and evaluate results.

## Results
- Model trained using **MobileNet** achieved **X% accuracy** on the test dataset.  
- Performance validated with confusion matrix and a detailed classification report.  
- Training progress is visualized through accuracy and loss curves.  
*(Replace X% with your final accuracy score and optionally add per-class metrics.)*

## Example Code Snippet (Model setup)
```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # fine-tune later if needed

inputs = tf.keras.Input(shape=(224,224,3))
x = layers.Resizing(224,224)(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Applications
- Automated screening for retinal diseases in clinical and community settings
- AI-assisted decision support for ophthalmologists
- Deployment in telemedicine/remote health programs

## Future Scope
- Deploy as a web or mobile application for real-time use
- Experiment with other transfer learning architectures (EfficientNet, InceptionV3)
- Improve robustness with larger, more diverse datasets and domain adaptation techniques
- Add model explainability (Grad-CAM / saliency maps) to support clinical acceptance

## Notes for Interviewers
- The notebook (`Training_model.ipynb`) contains the full pipeline: data loading, Keras-based preprocessing, model definition (MobileNet), training loop, evaluation, and figures.  
- Key choices (MobileNet, Keras preprocessing) were made to balance performance and inference efficiency for potential deployment scenarios.  
- Open to questions about dataset selection, handling class imbalance, augmentation strategy, fine-tuning vs. feature-extraction decisions, and model explainability.

## Author
**Your Name**  
Email: your.email@example.com  
GitHub: https://github.com/yourusername

---

*To personalize:* update the `Your Name`, contact details, GitHub URL, dataset source link, and final metrics (X%) before sharing with interviewers.
