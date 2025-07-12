# 🌿 Crop Yield Prediction System - Cotton Disease Predictor

**Developed by Nandhini S**  
**Department of Artificial Intelligence and Data Science**  
**Dr. N. G. P. Institute of Technology**  
**Email: nandhinisenthil1920@gmail.com**

---

An AI-powered web application that predicts cotton leaf diseases using deep learning. The application can detect four different conditions: Bacterial Blight, Curl Virus, Fusarium Wilt, and Healthy plants.

## 🚀 Features

- **AI-Powered Disease Detection**: Uses ResNet50-based CNN for accurate disease prediction
- **Beautiful Web Interface**: Modern, responsive design with intuitive user experience
- **Real-time Image Analysis**: Upload images and get instant predictions
- **Detailed Disease Information**: Comprehensive symptoms, treatments, and prevention tips
- **Confidence Scoring**: Shows prediction confidence levels for better decision making
- **Mobile-Friendly**: Works perfectly on smartphones and tablets

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Your cotton disease dataset in the `cotton/` folder

## 🛠️ Installation

1. **Clone or download this project** to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify your dataset structure**:
   ```
   cotton/
   ├── bacterial_blight/
   ├── curl_virus/
   ├── fussarium_wilt/
   └── healthy/
   ```

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Train the model** (first time only):
   - Click the "Train Model" button in the top-right corner
   - Wait for the training to complete (this may take several minutes)

4. **Start predicting**:
   - Upload a cotton leaf image
   - Click "Analyze Disease" to get predictions

## 📁 Project Structure

```
cotton disease predction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface
├── cotton/               # Your dataset
│   ├── bacterial_blight/
│   ├── curl_virus/
│   ├── fussarium_wilt/
│   └── healthy/
└── uploads/              # Temporary upload folder (auto-created)
```

## 🔧 API Endpoints

- `GET /` - Main web interface
- `POST /train` - Train the AI model
- `POST /predict` - Predict disease from uploaded image
- `GET /model-status` - Check if model is trained

## 🧠 Model Details

- **Architecture**: ResNet50 with custom classification layers
- **Input Size**: 224x224 pixels
- **Classes**: 4 (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy)
- **Training**: Transfer learning with ImageNet weights

## 🎨 Features

### Disease Detection
- **Bacterial Blight**: Angular leaf spots, water-soaked lesions
- **Curl Virus**: Upward leaf curling, yellowing veins
- **Fusarium Wilt**: Yellowing and wilting, vascular discoloration
- **Healthy**: Normal green leaves, proper growth

### User Interface
- Drag & drop image upload
- Real-time image preview
- Confidence scoring with visual indicators
- Detailed disease information and recommendations
- Responsive design for all devices

## 🔍 Troubleshooting

### Common Issues

1. **"Model not trained" error**:
   - Click "Train Model" button and wait for completion
   - Ensure your dataset is properly structured

2. **Training fails**:
   - Check that all image files are valid
   - Ensure sufficient disk space
   - Verify Python dependencies are installed

3. **Prediction errors**:
   - Use supported image formats (JPG, PNG, GIF)
   - Keep file size under 10MB
   - Ensure image shows clear cotton leaves

### Performance Tips

- **Faster training**: Reduce epochs in `app.py` (line 108)
- **Better accuracy**: Increase training data
- **Memory issues**: Reduce batch size in training

## 🌟 Future Enhancements

- [ ] Grad-CAM visualization for explainable AI
- [ ] Multi-language support (Tamil, Hindi, etc.)
- [ ] Batch prediction for multiple images
- [ ] Export predictions to PDF reports
- [ ] Integration with IoT sensors
- [ ] Mobile app version

## 📊 Dataset Information

The application uses your local dataset with the following structure:
- **Bacterial Blight**: Images showing angular leaf spots and lesions
- **Curl Virus**: Images showing leaf curling and yellowing
- **Fusarium Wilt**: Images showing wilting and vascular issues
- **Healthy**: Images of normal, healthy cotton leaves

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the UI/UX
- Adding more disease classes

## 📄 License

This project is open source and available under the MIT License.

## 👨‍🎓 Academic Information

This project was developed as part of academic research in the field of Artificial Intelligence and Data Science. The system demonstrates the application of deep learning techniques in agricultural technology for crop disease prediction and yield optimization.

### Research Focus
- **Domain**: Agricultural Technology
- **Technology**: Deep Learning, Computer Vision
- **Application**: Crop Disease Detection and Yield Prediction
- **Dataset**: Cotton Leaf Disease Dataset

## 🙏 Acknowledgments

- Built with TensorFlow and Flask
- Uses ResNet50 architecture for transfer learning
- Modern web design with Tailwind CSS
- Font Awesome icons for enhanced UI
- Academic guidance from Dr. N. G. P. Institute of Technology

---

**Happy Farming! 🌾**

For support or questions, please contact: **nandhinisenthil1920@gmail.com**

This project repository serves as a demonstration of AI applications in agricultural technology. 