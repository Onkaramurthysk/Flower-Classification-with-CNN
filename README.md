# 🌸 Flower Classification with CNN

## 📌 Project Overview
This project uses a **Convolutional Neural Network (CNN)** based on **ResNet-18** to classify images of flowers from the **Flowers102 dataset**. The model is trained and deployed using **Streamlit** for real-time predictions.

## 🚀 Features
✅ **Deep Learning Model**: ResNet-18 fine-tuned for **102 flower categories**  
✅ **Real-time Classification**: Upload an image to get **Top-3 predictions with confidence scores**  
✅ **Optimized Training**: Trained using **Adam optimizer** and **CrossEntropyLoss** on **GPU**  
✅ **Interactive Web App**: Built with **Streamlit** for easy user interaction  

## 🛠️ Technologies Used
- **Python** 🐍  
- **PyTorch** (Deep Learning)  
- **Torchvision** (Image Processing)  
- **Streamlit** (Model Deployment)  
- **PIL (Pillow)** (Image Handling)  
- **Scikit-Learn** (Data Processing)  

## 📥 Installation
### 🔹 1. Clone the Repository  
git clone https://github.com/your-repo/flower-classification-cnn.git
cd flower-classification-cnn

### 🔹 2. Install Dependencies
pip install -r requirements.txt

### 🔹 3. Train the Model
Before running the web app, train the model by executing the following command:
python train_model.py


###  🔹 4. Run the Streamlit App
streamlit run app.py

## 📊 Model Training
* The model is trained using ResNet-18, fine-tuned for 102 classes.
* Data is preprocessed using transforms (Resize, Normalize, ToTensor).
* Trained on GPU for faster convergence.
* Saved model weights as model.pth.

## 📷 Usage
* Open the Streamlit web app.
* Upload an image of a flower.
* The model will predict the Top-3 possible flowers with confidence scores.

## 🚀 Future Enhancements
* Improve accuracy with EfficientNet or ViT (Vision Transformers)
* Deploy on AWS/GCP for cloud-based accessibility
* Add explainability with Grad-CAM visualization

## 👨‍💻 Author
[Onkaramurthy S K] 
