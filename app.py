from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import cv2
import base64
import io
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = 224

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease classes
DISEASE_CLASSES = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']

# Global model variable
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path):
    """Load and preprocess image for model prediction"""
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def create_model():
    """Create and return the ResNet-based model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(DISEASE_CLASSES), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the model on the cotton disease dataset"""
    global model
    
    print("Loading and preprocessing dataset...")
    
    # Data loading and preprocessing
    X_train = []
    y_train = []
    
    # Load images from each class
    for class_idx, class_name in enumerate(DISEASE_CLASSES):
        class_path = os.path.join('cotton', class_name)
        if os.path.exists(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    try:
                        img_path = os.path.join(class_path, filename)
                        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = image.img_to_array(img)
                        img_array = preprocess_input(img_array)
                        
                        X_train.append(img_array)
                        y_train.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    if len(X_train) == 0:
        raise ValueError("No images found in the dataset!")
    
    X_train = np.array(X_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(DISEASE_CLASSES))
    
    print(f"Dataset loaded: {len(X_train)} images")
    print(f"Class distribution: {np.sum(y_train, axis=0)}")
    
    # Create and train model
    model = create_model()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model.save('cotton_disease_model.h5')
    print("Model trained and saved successfully!")
    
    return history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to train the model"""
    try:
        history = train_model()
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'epochs': len(history.history['accuracy'])
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict disease from uploaded image"""
    global model
    
    if model is None:
        # Try to load existing model
        try:
            model = tf.keras.models.load_model('cotton_disease_model.h5')
        except:
            return jsonify({
                'success': False,
                'message': 'Model not trained. Please train the model first.'
            }), 400
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image file provided'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No image selected'
        }), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            
            # Preprocess image
            img_array = load_and_preprocess_image(filename)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get disease information
            disease_name = DISEASE_CLASSES[predicted_class]
            disease_info = get_disease_info(disease_name)
            
            # Clean up uploaded file
            os.remove(filename)
            
            return jsonify({
                'success': True,
                'prediction': {
                    'disease': disease_name,
                    'confidence': confidence,
                    'class_id': int(predicted_class),
                    'all_probabilities': predictions[0].tolist()
                },
                'disease_info': disease_info
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Prediction failed: {str(e)}'
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'Invalid file format'
    }), 400

def get_disease_info(disease_name):
    """Return information about the detected disease"""
    disease_info = {
        'bacterial_blight': {
            'name': 'Bacterial Blight',
            'description': 'A serious bacterial disease caused by Xanthomonas axonopodis pv. malvacearum.',
            'symptoms': [
                'Angular leaf spots with dark brown margins',
                'Water-soaked lesions on leaves',
                'Black cankers on stems',
                'Wilting and death of young plants'
            ],
            'treatments': [
                'Remove and destroy infected plants',
                'Use copper-based bactericides',
                'Practice crop rotation',
                'Use disease-resistant varieties',
                'Avoid overhead irrigation'
            ],
            'prevention': [
                'Plant certified disease-free seeds',
                'Maintain proper plant spacing',
                'Control weeds and insects',
                'Avoid working in wet fields'
            ]
        },
        'curl_virus': {
            'name': 'Cotton Leaf Curl Virus',
            'description': 'A viral disease transmitted by whiteflies, causing severe leaf curling and stunting.',
            'symptoms': [
                'Upward curling of leaves',
                'Yellowing of leaf veins',
                'Stunted plant growth',
                'Reduced boll formation',
                'Leaf thickening and brittleness'
            ],
            'treatments': [
                'Control whitefly populations',
                'Use systemic insecticides',
                'Remove infected plants',
                'Plant virus-resistant varieties',
                'Use reflective mulches'
            ],
            'prevention': [
                'Monitor whitefly populations',
                'Use yellow sticky traps',
                'Plant early to avoid peak whitefly activity',
                'Maintain field hygiene'
            ]
        },
        'fussarium_wilt': {
            'name': 'Fusarium Wilt',
            'description': 'A fungal disease caused by Fusarium oxysporum f. sp. vasinfectum, affecting the vascular system.',
            'symptoms': [
                'Yellowing and wilting of leaves',
                'Brown discoloration of vascular tissue',
                'Stunted growth',
                'Premature defoliation',
                'Reduced yield'
            ],
            'treatments': [
                'Use fungicide seed treatments',
                'Apply soil fumigants',
                'Use resistant varieties',
                'Practice crop rotation',
                'Improve soil drainage'
            ],
            'prevention': [
                'Use certified disease-free seeds',
                'Maintain proper soil pH',
                'Avoid over-irrigation',
                'Practice crop rotation with non-host crops'
            ]
        },
        'healthy': {
            'name': 'Healthy Plant',
            'description': 'The cotton plant appears to be healthy with no visible disease symptoms.',
            'symptoms': [
                'Normal green leaf color',
                'Proper leaf shape and size',
                'No spots or lesions',
                'Healthy stem and branches',
                'Normal growth pattern'
            ],
            'treatments': [
                'Continue regular monitoring',
                'Maintain proper irrigation',
                'Apply balanced fertilizers',
                'Control pests if needed',
                'Practice good field hygiene'
            ],
            'prevention': [
                'Regular field monitoring',
                'Proper nutrient management',
                'Timely irrigation',
                'Pest and disease monitoring',
                'Good agricultural practices'
            ]
        }
    }
    
    return disease_info.get(disease_name, {})

@app.route('/model-status')
def model_status():
    """Check if model is trained and available"""
    global model
    
    if model is None:
        try:
            model = tf.keras.models.load_model('cotton_disease_model.h5')
            return jsonify({
                'trained': True,
                'message': 'Model loaded successfully'
            })
        except:
            return jsonify({
                'trained': False,
                'message': 'Model not available'
            })
    
    return jsonify({
        'trained': True,
        'message': 'Model is ready'
    })

if __name__ == '__main__':
    print("Starting Cotton Disease Prediction App...")
    print("Available endpoints:")
    print("- GET  / : Main application")
    print("- POST /train : Train the model")
    print("- POST /predict : Predict disease from image")
    print("- GET  /model-status : Check model status")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 