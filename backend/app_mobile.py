# Modified Flask app for mobile access
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import socket

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = r"C:\Users\Admin\Desktop\AI_Poultry_Disease_Diagnosis\models\best_lightweight_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def get_local_ip():
    """Get the local IP address of this computer"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def preprocess_image(image_file):
    """Preprocess image for model prediction"""
    try:
        # Open and convert image
        image = Image.open(image_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (128x128 as per your training)
        image = image.resize((128, 128))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Mobile-friendly HTML template
MOBILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐔 AI Poultry Disease Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 500px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 14px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background-color: #f0f4ff;
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .upload-text {
            font-size: 16px;
            color: #333;
            margin-bottom: 5px;
        }
        
        .upload-subtext {
            font-size: 12px;
            color: #999;
        }
        
        #imageInput {
            display: none;
        }
        
        .preview-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .analyze-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
        }
        
        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result.healthy {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .result.diseased {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence-bar {
            background-color: rgba(255,255,255,0.3);
            border-radius: 10px;
            height: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .confidence-fill.healthy {
            background-color: #28a745;
        }
        
        .confidence-fill.diseased {
            background-color: #dc3545;
        }
        
        .recommendation {
            margin-top: 15px;
            padding: 15px;
            background-color: rgba(255,255,255,0.5);
            border-radius: 8px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐔 AI Poultry Disease Detector</h1>
            <p>Newcastle Disease Detection using Deep Learning</p>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Tap to select chicken image</div>
            <div class="upload-subtext">or drag and drop here</div>
        </div>
        
        <input type="file" id="imageInput" accept="image/*" capture="environment">
        
        <div class="preview-container" id="previewContainer" style="display: none;">
            <img id="previewImage" class="preview-image" alt="Preview">
        </div>
        
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
            🔍 Analyze Image
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Analyzing image...</div>
        </div>
        
        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div>Confidence: <span id="confidence"></span>%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceFill"></div>
            </div>
            <div class="recommendation" id="recommendation"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('analyzeBtn').disabled = false;
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedFile = file;
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('previewImage').src = e.target.result;
                        document.getElementById('previewContainer').style.display = 'block';
                        document.getElementById('analyzeBtn').disabled = false;
                    };
                    reader.readAsDataURL(file);
                }
            }
        });
        
        async function analyzeImage() {
            if (!selectedFile) return;
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const formData = new FormData();
                formData.append('image', selectedFile);
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show results
                displayResult(data);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                alert('Error analyzing image. Please try again.');
            }
            
            document.getElementById('analyzeBtn').disabled = false;
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            const isHealthy = !data.is_diseased;
            const confidence = Math.round(data.confidence * 100);
            
            // Set result class
            resultDiv.className = 'result ' + (isHealthy ? 'healthy' : 'diseased');
            
            // Set content
            document.getElementById('resultTitle').textContent = 
                isHealthy ? '✅ Healthy Chicken' : '🚨 Newcastle Disease Detected';
            document.getElementById('confidence').textContent = confidence;
            
            // Set confidence bar
            const confidenceFill = document.getElementById('confidenceFill');
            confidenceFill.className = 'confidence-fill ' + (isHealthy ? 'healthy' : 'diseased');
            confidenceFill.style.width = confidence + '%';
            
            // Set recommendation
            document.getElementById('recommendation').textContent = isHealthy
                ? 'The chicken appears healthy. Continue regular monitoring and maintain good hygiene practices.'
                : 'Newcastle disease detected. Isolate the bird immediately and consult a veterinarian. This is a highly contagious viral disease.';
            
            // Show result
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the mobile-friendly interface"""
    return render_template_string(MOBILE_HTML)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict Newcastle disease from uploaded image"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Interpret results
        if prediction > 0.5:
            result = "Newcastle Disease"
            confidence = float(prediction)
            is_diseased = True
        else:
            result = "Healthy"
            confidence = float(1 - prediction)
            is_diseased = False
        
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'is_diseased': is_diseased,
            'raw_prediction': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("🚀 Starting Newcastle Disease Detection API for Mobile...")
    print(f"📱 Access from your phone: http://{local_ip}:5000")
    print(f"💻 Access from this computer: http://localhost:5000")
    print(f"🌐 Make sure your phone is on the same WiFi network!")
    print("=" * 60)
    
    # Run on all interfaces so it's accessible from other devices
    app.run(debug=True, host='0.0.0.0', port=5000)