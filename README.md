# Dog Breed Prediction

A modern Flask web application that predicts dog breeds from uploaded images using a pre-trained Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Features

- **Image Upload**: Upload dog images in PNG, JPG, JPEG, GIF, or BMP format
- **Breed Classification**: Real-time prediction of dog breed using a trained deep learning model
- **Confidence Scoring**: Display prediction confidence score with visual progress bar
- **All Predictions**: View probability distribution across all 5 dog breed classes
- **Modern Dark UI**: Sleek dark-gradient interface with glassmorphism design
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Error Handling**: Comprehensive error handling and user-friendly messages

## Supported Dog Breeds

The model can classify the following dog breeds:
1. Saint Bernard
2. Scottish Deerhound
3. Siberian Husky
4. Silky Terrier
5. Yorkshire Terrier

## Technical Stack

- **Backend**: Flask 2.3.3
- **Deep Learning**: TensorFlow 2.18.0, Keras 3.13.0
- **Image Processing**: OpenCV, scikit-image, Pillow
- **Server**: Gunicorn
- **Frontend**: HTML, CSS (modern dark gradient theme)

## Prerequisites

- Windows 10 / 11 (or Linux/macOS)
- Python 3.12.4+
- pip (Python package manager)

## Installation & Setup

### 1. Clone/Navigate to Project
```bash
cd dog_breed_prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access in Browser
- **URL**: http://localhost:5000/
- **Upload** a dog image to get a breed prediction
- **View Results** on the dedicated result page with confidence score

### 6. Stop Server
Press `CTRL + C` in the terminal

## Project Structure

```
2.dog_breed_prediction/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── model/
│   └── dog_breed_model.h5     # Pre-trained CNN model
├── static/
│   └── styles.css             # Modern dark-gradient UI styling
├── templates/
│   ├── index.html             # Upload form page
│   └── result.html            # Prediction results page
└── uploads/                   # Temporary uploaded images (created at runtime)
```

## How It Works

1. **User uploads an image** through the web interface
2. **Image preprocessing** converts image to 224x224 pixels and normalizes pixel values
3. **Model prediction** runs the pre-trained CNN on the processed image
4. **Results displayed** showing predicted breed, confidence score, and all class probabilities
5. **File cleanup** uploaded file is removed after processing

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224 x 224 pixels
- **Output Classes**: 5 dog breeds
- **Framework**: TensorFlow/Keras
- **Model File**: `model/dog_breed_model.h5`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Render upload form |
| `/` | POST | Handle image upload and prediction |
| `/uploads/<filename>` | GET | Serve uploaded images |
| `/predict` | POST | API endpoint for predictions (JSON) |
| `/health` | GET | Health check endpoint |

## File Upload Limits

- **Max File Size**: 16 MB
- **Allowed Formats**: PNG, JPG, JPEG, GIF, BMP

## Configuration

Edit `app.py` to modify:
- `UPLOAD_FOLDER`: Location for temporary uploads
- `MAX_CONTENT_LENGTH`: Maximum file upload size
- `ALLOWED_EXTENSIONS`: Supported image formats
- `IMG_SIZE`: Model input image size (default: 224)
- `CLASS_NAMES`: List of dog breed classes

## Logging

The application logs important events such as:
- Model loading status
- Image upload events
- Prediction results
- Errors and exceptions

Logs are displayed in the console during execution.

## Tips for Best Results

- Use **clear, well-lit images** of dogs
- **Full body or close-up shots** work best
- Images should show the **dog clearly** without obstructions
- **Single dog per image** for accurate predictions
- Confidence score above **60%** indicates high confidence

## Troubleshooting

### Model Not Loading
Ensure `model/dog_breed_model.h5` exists in the `model/` folder.

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead
```

### Image Upload Fails
Check that:
- Image format is supported (PNG, JPG, JPEG, GIF, BMP)
- File size is under 16 MB
- Browser allows file uploads

## Future Enhancements

- [ ] Support for more dog breeds
- [ ] Batch image processing
- [ ] Image augmentation for better accuracy
- [ ] Model fine-tuning
- [ ] REST API documentation (Swagger)
- [ ] Database integration for prediction history
- [ ] User authentication

## License

This project is provided as-is for educational purposes.

## Author

ML Project - December 2025

---

**Happy Breed Predicting! 🐕**

---

## Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Git initialized
- Heroku account

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Deploy to Heroku**
   ```bash
   git push heroku main
   ```

4. **View Logs**
   ```bash
   heroku logs --tail
   ```

5. **Access Your App**
   ```bash
   heroku open
   ```

### Important Files for Heroku
- `Procfile` - Specifies how Heroku runs the app (uses gunicorn)
- `runtime.txt` - Specifies Python version (3.10.13)
- `requirements.txt` - Lists all dependencies with pinned versions

### Troubleshooting Heroku Issues

**App crashes after deployment?**
- Check logs: `heroku logs --tail`
- Verify PORT environment variable is used (app.py handles this)
- Ensure `Procfile` uses `gunicorn app:app`

**Static files not loading?**
- Flask serves static files in `/static` folder
- Ensure CSS and JS are in the correct directory

**Reference image missing?**
- Original PAN image must be at: `model/original_pan.jpg`
- Verify during deployment: `git push heroku main --verbose`

# Folder Structure

pan_tampering_app/
│
├── app.py
├── model/
│   └── original_pan.jpg        # genuine PAN template
│
├── static/
│   ├── styles.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── requirements.txt
└── README.md

# How to Run (Windows)

```bash
cd pan_tampering_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```