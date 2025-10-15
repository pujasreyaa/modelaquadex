from flask import Flask, request, jsonify, send_from_directory, make_response
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 🔥 Model paths
MODEL_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FSRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "fsrcnn_superres.pth")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_arcface.pth")

DEVICE = torch.device('cpu')

yolo_model = None
fsrcnn_model = None
classifier_model = None

# --- FSRCNN ---
class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, d, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(d, s, kernel_size=1)
        self.conv3 = nn.Conv2d(s, d, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.deconv(x)
        return x

# --- Classifier ---
class MobileNetV2ArcFace(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=512):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return logits

# --- Load Models ---
try:
    print("⏳ Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(DEVICE)
    print("✅ YOLOv8 loaded.")

    print("⏳ Loading FSRCNN model...")
    fsrcnn_model = FSRCNN(scale_factor=4)
    state_dict = torch.load(FSRCNN_MODEL_PATH, map_location=DEVICE, weights_only=True)
    fsrcnn_model.load_state_dict(state_dict)
    fsrcnn_model.eval().to(DEVICE)
    print("✅ FSRCNN loaded.")

    print("⏳ Loading Classifier model...")
    classifier_model = MobileNetV2ArcFace(num_classes=4, embedding_dim=512)
    state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE, weights_only=True)
    classifier_model.load_state_dict(state_dict)
    classifier_model.eval().to(DEVICE)
    print("✅ Classifier loaded.")

except Exception as e:
    print("❌ MODEL LOADING FAILED:")
    print(traceback.format_exc())
    yolo_model = None
    fsrcnn_model = None
    classifier_model = None

# --- Constants ---
CLASS_NAMES = ["Ceratium", "Coscinodiscus", "Dinophysis", "Euglena"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

def preprocess_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(DEVICE)

def enhance_crop_with_fsrcnn(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr_tensor = fsrcnn_model(tensor)
        sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

# --- Routes ---
@app.route('/')
def analysis_page():
    html_content = '''
<!DOCTYPE html>
<html lang="en" class="dark-theme">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plankton Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="icon" type="image/png" href="/logo.png">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #00334e;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 50px;
            background-color: #002a44;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
            box-sizing: border-box;
        }
        .navbar .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 20px;
            color: #00bcd4;
            flex: 1;
        }
        .navbar .logo img {
            height: 42px;
            width: 42px;
            border-radius: 50%;
        }
        .navbar nav {
            display: flex;
            gap: 25px;
            flex: 9;
            justify-content: center;
        }
        .navbar nav a {
            text-decoration: none;
            color: #ffffff;
            font-weight: 500;
            transition: color 0.3s;
        }
        .navbar nav a:hover {
            color: #00bcd4;
        }
        .navbar nav a.active {
            border-bottom: 2px solid #00bcd4;
            padding-bottom: 4px;
            color: #00bcd4;
        }
        .navbar .auth-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .container {
            max-width: 1200px;
            margin: 100px auto 20px auto;
            padding: 0 20px;
        }
        .card {
            background: #e6f0fa;
            color: #00334e;
            border-radius: 12px;
            box-shadow: 0 6px 14px rgba(0,0,0,0.15);
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #002a44;
            margin-top: 0;
            margin-bottom: 20px;
        }
        .upload-section .upload-box {
            border: 2px dashed #00bcd4;
            border-radius: 8px;
            text-align: center;
            padding: 40px 20px;
            color: #37474f;
        }
        .upload-icon {
            margin-bottom: 10px;
        }
        .upload-box p {
            margin: 0;
            font-size: 1rem;
            color: #00334e;
        }
        .or-text {
            font-size: 0.875rem;
        }
        .support-text {
            font-size: 0.75rem;
            color: #78909c;
            margin-top: 15px;
        }
        .btn-primary {
            background-color: #002a44;
            color: #ffffff;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            margin-top: 15px;
            transition: background 0.3s ease;
        }
        .btn-primary:hover {
            background: #00bcd4;
            color: #002a44;
        }
        .results-section {
            display: none;
        }
        .results-section.show {
            display: block;
        }
        .processed-container {
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }
        .processed-image-box {
            flex: 1 1 50%;
            background-color: #002a44;
            border-radius: 8px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        #processedImage {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            display: none;
        }
        .image-placeholder {
            text-align: center;
            color: #cfd8dc;
            padding: 20px;
        }
        .image-placeholder p {
            font-size: 0.875rem;
            margin-top: 10px;
        }
        .species-detected {
            flex: 1 1 40%;
            min-width: 300px;
            text-align: left;
        }
        .table-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #002a44;
            margin-bottom: 10px;
        }
        .species-detected table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .species-detected th, .species-detected td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 0.875rem;
        }
        .species-detected th {
            background-color: #f0f0f0;
            font-weight: 600;
            color: #00334e;
        }
        .species-detected tr:hover {
            background-color: #f9f9f9;
        }
        .species-detected td {
            color: #00334e;
        }
        .species-detected td:first-child {
            width: 24px;
        }
        .species-detected td:first-child img {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            object-fit: cover;
            border: 1px solid #ddd;
        }
        .table-summary {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            font-size: 0.8rem;
            color: #78909c;
        }
        @media (max-width: 768px) {
            .processed-container {
                flex-direction: column;
            }
            .processed-image-box {
                min-height: 250px;
            }
        }
    </style>
</head>
<body>

    <header class="navbar">
        <div class="logo">
            <img src="/logo.png" alt="AquaDex Logo">
            <span>AquaDex</span>
        </div>
          
        <nav>
            <a href="/home">Home</a>
            <a href="/" class="active">Analysis</a>
            <a href="/database">Database</a>
            <a href="/history">History</a>
        </nav>
        <div id="auth-container" class="auth-container"></div>
    </header>

    <div class="container">
        <div class="card upload-section">
            <h3 class="card-title">Upload Media</h3>
            <div class="upload-box" id="drop-area">
                <div class="upload-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="#00bcd4" viewBox="0 0 24 24">
                        <path d="M19 13v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2zM12 2l-6 6h4v4h4V8h4l-6-6z"/>
                    </svg>
                </div>
                <p>Drag and drop your files here</p>
                <p class="or-text">or click to browse</p>
                <input type="file" id="fileInput" multiple accept="image/jpeg,image/png" style="display: none;">
                <button class="btn-primary" id="chooseFileBtn">Choose Files</button>
                <p class="support-text">Supports: JPG, PNG (max 50MB)</p>
            </div>
        </div>

        <div class="card results-section" id="results-card">
            <h3 class="card-title">Analysis Results</h3>
            <div class="processed-container">
                <div class="processed-image-box">
                    <img id="processedImage" alt="Processed image">
                    <div class="image-placeholder" id="imagePlaceholder">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="#00bcd4" viewBox="0 0 24 24">
                            <path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                        </svg>
                        <p>Processed image with detected species will appear here</p>
                    </div>
                </div>
                
                <div class="species-detected">
                    <h4 class="table-title">Species Detected</h4>
                    <table id="speciesTable">
                        <thead>
                            <tr>
                                <th>Image</th>
                                <th>Species</th>
                                <th>Predicted</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    </table>
                    <div class="table-summary">
                        <span id="totalSpecies">Total species detected: 0</span>
                        <span id="totalOrganisms">Total organisms counted: 0</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const CLASS_NAMES = ["Ceratium", "Coscinodiscus", "Dinophysis", "Euglena"];
        const COLORS = [[255,0,0], [0,255,0], [0,0,255], [255,255,0]];

        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('fileInput');
        const chooseFileBtn = document.getElementById('chooseFileBtn');
        const resultsCard = document.getElementById('results-card');
        const processedImage = document.getElementById('processedImage');
        const imagePlaceholder = document.getElementById('imagePlaceholder');
        const speciesTableBody = document.querySelector('#speciesTable tbody');
        const totalSpeciesSpan = document.getElementById('totalSpecies');
        const totalOrganismsSpan = document.getElementById('totalOrganisms');
        const authContainer = document.getElementById('auth-container');

        function checkLoginStatus() {
            const user = localStorage.getItem('loggedInUser');
            if (user) {
                authContainer.innerHTML = `<span class="welcome-text">Welcome, ${user}</span><a href="#" class="log-out-btn">Log Out</a>`;
                document.querySelector('.log-out-btn').addEventListener('click', (e) => {
                    e.preventDefault();
                    localStorage.removeItem('loggedInUser');
                    window.location.reload();
                });
            }
        }

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); });
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'));
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'));
        });

        dropArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            handleFiles(files);
        });

        chooseFileBtn.addEventListener('click', () => fileInput.click());
        dropArea.addEventListener('click', (e) => {
            if (!e.target.closest('#chooseFileBtn')) fileInput.click();
        });

        fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

        async function handleFiles(files) {
            if (!files || files.length === 0) return;
            const file = files[0];
            if (!file.type.match('image.*')) {
                alert("Please upload an image (JPG/PNG).");
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                resultsCard.classList.add('show');
                processedImage.src = 'data:image/jpeg;base64,' + data.processed_image;
                processedImage.onload = () => {
                    imagePlaceholder.style.display = 'none';
                    processedImage.style.display = 'block';
                };
                processedImage.onerror = () => {
                    alert("Failed to load processed image. Check server logs.");
                };

                speciesTableBody.innerHTML = '';
                const counts = {};
                data.detections.forEach(detection => {
                    counts[detection.label] = (counts[detection.label] || 0) + 1;
                });
                
                Object.entries(counts).forEach(([species, count]) => {
                    const detection = data.detections.find(d => d.label === species);
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <!-- ✅ SHOW ACTUAL CROPPED IMAGE -->
                        <td><img src="data:image/jpeg;base64,${detection.crop_base64}" alt="${species}"></td>
                        <td>${species}</td>
                        <td>${(detection.confidence * 100).toFixed(1)}%</td>
                        <td>${count}</td>
                    `;
                    speciesTableBody.appendChild(row);
                });
                
                totalSpeciesSpan.textContent = `Total species detected: ${Object.keys(counts).length}`;
                totalOrganismsSpan.textContent = `Total organisms counted: ${data.detections.length}`;
            } catch (error) {
                console.error('Upload failed:', error);
                alert('Upload failed. Check console for details.');
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            checkLoginStatus();
        });
    </script>
</body>
</html>
'''
    return html_content

@app.route('/upload', methods=['POST'])
def upload_file():
    if yolo_model is None or fsrcnn_model is None or classifier_model is None:
        return jsonify({'error': 'Models not loaded properly. Check server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        image_stream = BytesIO()
        file.save(image_stream)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        results = yolo_model(image, conf=0.25, imgsz=640, device=DEVICE)
        boxes = results[0].boxes
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = preprocess_crop(crop)
            with torch.no_grad():
                logits = classifier_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, cls_id = torch.max(probs, dim=1)
                confidence = confidence.item()
                cls_id = cls_id.item()

            if confidence < 0.7:
                enhanced_crop = enhance_crop_with_fsrcnn(crop)
                input_tensor_enhanced = preprocess_crop(enhanced_crop)
                with torch.no_grad():
                    logits_enhanced = classifier_model(input_tensor_enhanced)
                    probs_enhanced = torch.softmax(logits_enhanced, dim=1)
                    confidence_enhanced, cls_id_enhanced = torch.max(probs_enhanced, dim=1)
                    if confidence_enhanced > confidence:
                        confidence = confidence_enhanced.item()
                        cls_id = cls_id_enhanced.item()

            # ✅ Encode crop to base64 for frontend
            _, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            crop_base64 = base64.b64encode(buffer).decode('utf-8')

            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cls_id': cls_id,
                'label': CLASS_NAMES[cls_id],
                'confidence': confidence,
                'crop_base64': crop_base64  # ← Added for frontend
            })

        if not detections:
            return jsonify({'error': 'No plankton detected.'}), 400

        result_image = image.copy()
        for obj in detections:
            color = COLORS[obj['cls_id']]
            cv2.rectangle(result_image, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), color, 2)
            label = f"{obj['label']} {obj['confidence']:.2f}"
            cv2.putText(result_image, label, (obj['x1'], obj['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', result_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': jpg_as_text,
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logo.png')
def logo():
    if os.path.exists('logo.png'):
        return send_from_directory('.', 'logo.png')
    else:
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="#00bcd4"/>
            <text x="12" y="16" text-anchor="middle" fill="white" font-size="10">A</text>
        </svg>'''
        response = make_response(svg)
        response.headers['Content-Type'] = 'image/svg+xml'
        return response

@app.route('/home')
@app.route('/database')
@app.route('/history')
def dummy_pages():
    return '<h1>Page under construction</h1><a href="/">Back to Analysis</a>'

if __name__ == '__main__':
    print("🚀 Starting AquaDex server...")
    print(f"Models dir: {os.path.abspath(MODEL_DIR)}")
    app.run(host='0.0.0.0', port=5000, debug=True)