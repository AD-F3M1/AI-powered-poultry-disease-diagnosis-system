from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Update this path to your actual model location
MODEL_PATH = r"C:\Users\Admin\Desktop\AI_Poultry_Disease_Diagnosis\models\best_lightweight_model.h5"

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def predict_image(image_file):
    """Make prediction on uploaded image"""
    try:
        # Open and process image
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match training size
        image = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Interpret result
        if prediction > 0.5:
            result = "Newcastle Disease Detected"
            confidence = prediction
            is_diseased = True
        else:
            result = "Healthy"
            confidence = 1 - prediction
            is_diseased = False
        
        return {
            'result': result,
            'confidence': float(confidence),
            'is_diseased': is_diseased
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    """Main page with upload form"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐔 AI Poultry Disease Detector</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .upload-area { border: 2px dashed #007bff; padding: 40px; margin: 20px 0; border-radius: 10px; text-align: center; cursor: pointer; transition: all 0.3s; }
            .upload-area:hover { border-color: #0056b3; background: #f8f9fa; }
            .btn { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn:hover { background: #0056b3; }
            .btn:disabled { background: #ccc; cursor: not-allowed; }
            .result { margin: 20px 0; padding: 20px; border-radius: 10px; }
            .healthy { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .diseased { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .loading { display: none; text-align: center; color: #007bff; }
            #preview { max-width: 300px; margin: 10px auto; display: block; border-radius: 8px; }
            .confidence-bar { background: #e9ecef; height: 20px; border-radius: 10px; margin: 10px 0; overflow: hidden; }
            .confidence-fill { height: 100%; transition: width 0.5s ease; }
            .healthy-fill { background: #28a745; }
            .diseased-fill { background: #dc3545; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐔 AI Poultry Disease Detector</h1>
            <p style="text-align: center; color: #666;">Upload a chicken image to detect Newcastle Disease using AI</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <p>📸 Click here to select a chicken image</p>
                    <p style="font-size: 14px; color: #666;">Supported formats: JPG, PNG, GIF</p>
                    <input type="file" id="imageInput" name="image" accept="image/*" style="display: none;" onchange="previewImage()">
                </div>
                <img id="preview" style="display: none;">
                <br>
                <div style="text-align: center;">
                    <button type="submit" class="btn" id="analyzeBtn">🔍 Analyze Image</button>
                </div>
            </form>
            
            <div id="loading" class="loading">
                <p>🔄 Analyzing image with AI...</p>
                <p style="font-size: 14px;">This may take a few seconds</p>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            function previewImage() {
                const file = document.getElementById('imageInput').files[0];
                const preview = document.getElementById('preview');
                
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                }
            }

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const imageFile = document.getElementById('imageInput').files[0];
                const analyzeBtn = document.getElementById('analyzeBtn');
                
                if (!imageFile) {
                    alert('Please select an image first!');
                    return;
                }
                
                formData.append('image', imageFile);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🔄 Analyzing...';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🔍 Analyze Image';
                    
                    // Show result
                    if (data.error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="result diseased">❌ <strong>Error:</strong> ' + data.error + '</div>';
                    } else {
                        const resultClass = data.is_diseased ? 'diseased' : 'healthy';
                        const emoji = data.is_diseased ? '🚨' : '✅';
                        const confidence = Math.round(data.confidence * 100);
                        const fillClass = data.is_diseased ? 'diseased-fill' : 'healthy-fill';
                        
                        document.getElementById('result').innerHTML = 
                            '<div class="result ' + resultClass + '">' +
                            '<h3>' + emoji + ' ' + data.result + '</h3>' +
                            '<div class="confidence-bar">' +
                            '<div class="confidence-fill ' + fillClass + '" style="width: ' + confidence + '%"></div>' +
                            '</div>' +
                            '<p><strong>Confidence:</strong> ' + confidence + '%</p>' +
                            (data.is_diseased ? 
                                '<p><strong>⚠️ Recommendation:</strong> Isolate the bird immediately and consult a veterinarian. Newcastle disease is highly contagious.</p>' :
                                '<p><strong>✅ Recommendation:</strong> Bird appears healthy. Continue regular monitoring and maintain good hygiene practices.</p>') +
                            '</div>';
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🔍 Analyze Image';
                    document.getElementById('result').innerHTML = 
                        '<div class="result diseased">❌ <strong>Error:</strong> Failed to connect to AI model. Please try again.</div>';
                }
            });
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    if model is None:
        return jsonify({'error': 'AI model not loaded properly'})
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    result = predict_image(image_file)
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 Starting AI Poultry Disease Detector...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)