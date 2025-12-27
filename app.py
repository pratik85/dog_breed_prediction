"""
Dog Breed Prediction Flask Application

This Flask application provides a web interface for dog breed classification.
It uses a pre-trained Convolutional Neural Network (CNN) model to predict
the breed of dogs from uploaded images.

Features:
    - Upload dog images through a web interface
    - Real-time prediction of dog breed
    - Confidence score display
    - Responsive UI with styling

Author: ML Project
Date: December 2025
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize Flask application
app = Flask(__name__)

# Configuration settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads folder if it doesn't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model path (use local model file inside the project's `model/` folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'dog_breed_model.h5')

# Image preprocessing parameters
IMG_SIZE = 224
IMG_SCALE = 1.0 / 255.0

# Dog breed class names
# These should match the order of classes used during model training
CLASS_NAMES = [
    'saint_bernard',
    'scottish_deerhound',
    'siberian_husky',
    'silky_terrier',
    'yorkshire_terrier'
]

# ============================================================================
# MODEL LOADING
# ============================================================================

try:
    """
    Load the pre-trained dog breed classification model.
    The model is a CNN trained on the Mini Dog Breed dataset.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction.
    
    Steps:
        1. Read image using OpenCV
        2. Convert BGR to RGB color space
        3. Resize to model input size (224x224)
        4. Normalize pixel values to [0, 1]
        5. Add batch dimension
        
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (preprocessed_image, original_image) or (None, None) if error occurs
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None, None
        
        # Store original for reference
        original_img = img.copy()
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to required dimensions
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values
        img = img.astype('float32') * IMG_SCALE
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info(f"Image preprocessed successfully: {image_path}")
        return img, original_img
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, None


def predict_breed(image_path):
    """
    Predict the dog breed from the uploaded image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing:
            - 'breed': Predicted dog breed
            - 'confidence': Confidence score (0-1)
            - 'all_predictions': All class predictions with probabilities
            - 'error': Error message if prediction fails
    """
    if model is None:
        logger.error("Model not loaded")
        return {
            'error': 'Model not loaded. Please check the model path.',
            'breed': None,
            'confidence': 0.0
        }
    
    try:
        # Preprocess image
        processed_img, original_img = preprocess_image(image_path)
        
        if processed_img is None:
            return {
                'error': 'Failed to process image.',
                'breed': None,
                'confidence': 0.0
            }
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_breed = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        # Create dictionary of all predictions
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        logger.info(
            f"Prediction - Breed: {predicted_breed}, Confidence: {confidence:.2f}"
        )
        
        return {
            'breed': predicted_breed,
            'confidence': round(confidence, 4),
            'all_predictions': all_predictions,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            'error': f'Prediction error: {str(e)}',
            'breed': None,
            'confidence': 0.0
        }


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the home page and handle form POST uploads.

    GET: render the upload form
    POST: accept uploaded file, run prediction, and render results
    """
    if request.method == 'POST':
        # Form upload path (input name expected to be 'file')
        if 'file' not in request.files:
            logger.warning("No file provided in form submit")
            return render_template('index.html', error='No file provided')

        file = request.files['file']

        if file.filename == '':
            logger.warning("No file selected in form submit")
            return render_template('index.html', error='No file selected')

        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return render_template('index.html', error=f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}')

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File uploaded via form: {filename}")

            result = predict_breed(filepath)

            # If prediction returned an error, show it
            if result.get('error'):
                return render_template('index.html', error=result.get('error'))

            # Render a dedicated result page showing image and breed
            return render_template('result.html', breed=result.get('breed'), confidence=result.get('confidence'), image_path=filename, all_predictions=result.get('all_predictions'))

        except Exception as e:
            logger.error(f"Error processing upload on index route: {str(e)}")
            return render_template('index.html', error=str(e))

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return breed prediction.
    
    Request:
        - file: Image file (multipart/form-data)
        
    Returns:
        JSON response containing:
        - breed: Predicted dog breed
        - confidence: Confidence score
        - all_predictions: Predictions for all breeds
        - error: Error message if any
    """
    # Check if file is present in request
    if 'file' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File uploaded: {filename}")
        
        # Make prediction
        result = predict_breed(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file: {str(e)}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify application status.
    
    Returns:
        JSON response with application status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 - Not Found errors."""
    logger.error(f"404 error: {error}")
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 - Internal Server errors."""
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Run the Flask application.
    
    Configuration:
        - Debug mode: True (set to False in production)
        - Host: 0.0.0.0 (accessible from any network interface)
        - Port: 5000
        - Threaded: True (handle multiple requests)
    """
    logger.info("Starting Dog Breed Prediction Flask Application")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
