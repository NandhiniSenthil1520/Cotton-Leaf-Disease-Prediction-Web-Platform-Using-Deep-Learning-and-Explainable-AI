from flask import Flask, request, jsonify, render_template
import os
import base64
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease classes
DISEASE_CLASSES = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def simulate_prediction(filename):
    """Simulate a prediction based on filename patterns"""
    filename_lower = filename.lower()
    
    # Simple pattern matching for demonstration
    if 'bact' in filename_lower or 'blight' in filename_lower:
        return 'bacterial_blight', 0.85
    elif 'curl' in filename_lower or 'virus' in filename_lower:
        return 'curl_virus', 0.78
    elif 'fussarium' in filename_lower or 'wilt' in filename_lower:
        return 'fussarium_wilt', 0.82
    elif 'healthy' in filename_lower or 'h' in filename_lower:
        return 'healthy', 0.90
    else:
        # Random selection for demo purposes
        import random
        disease = random.choice(DISEASE_CLASSES)
        confidence = random.uniform(0.6, 0.95)
        return disease, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict disease from uploaded image"""
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
            
            # Simulate prediction (in real app, this would use the ML model)
            predicted_disease, confidence = simulate_prediction(file.filename)
            
            # Get disease information
            disease_info = get_disease_info(predicted_disease)
            
            # Clean up uploaded file
            os.remove(filename)
            
            return jsonify({
                'success': True,
                'prediction': {
                    'disease': predicted_disease,
                    'confidence': confidence,
                    'class_id': DISEASE_CLASSES.index(predicted_disease),
                    'all_probabilities': [0.1, 0.1, 0.1, 0.1]  # Placeholder
                },
                'disease_info': disease_info,
                'note': 'This is a demonstration version. For real predictions, install TensorFlow and train the model.'
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

@app.route('/model-status')
def model_status():
    """Check if model is available"""
    return jsonify({
        'trained': True,
        'message': 'Demo mode - using simulated predictions',
        'demo': True
    })

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to train the model (demo version)"""
    return jsonify({
        'success': True,
        'message': 'Demo mode - no training required',
        'demo': True
    })

if __name__ == '__main__':
    print("🌿 Crop Yield Prediction System - Cotton Disease Predictor")
    print("=" * 60)
    print("Developed by Nandhini S")
    print("Department of Artificial Intelligence and Data Science")
    print("Dr. N. G. P. Institute of Technology")
    print("Email: nandhinisenthil1920@gmail.com")
    print("=" * 60)
    print("This is a demonstration version that simulates predictions.")
    print("For full functionality, install TensorFlow and train the model.")
    print("=" * 60)
    print("Available endpoints:")
    print("- GET  / : Main application")
    print("- POST /train : Train the model (demo)")
    print("- POST /predict : Predict disease from image (simulated)")
    print("- GET  /model-status : Check model status")
    print("=" * 60)
    print("Starting server at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 