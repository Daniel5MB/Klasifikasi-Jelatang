from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Path folder project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder upload file gambar dan model
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model_saya(terbaru).keras')

# Pastikan folder uploads ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model Keras
model = load_model(MODEL_PATH)

# Daftar kelas sesuai urutan di training datagen di Colab (sama urutan dengan model output)
class_labels = ['Bukan Jelatang', 'Jelatang Ayam', 'Jelatang Gajah', 'Jelatang Niru' ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', prediction='Tidak ada file yang dipilih.')

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Simpan gambar yang diupload
    file.save(file_path)

    # Preprocessing gambar sesuai Colab:
    # - Resize ke (150, 150)
    # - Rescale 1./255 (sama seperti di ImageDataGenerator)
    # - Expand dims untuk batch size 1
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi dengan model
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    confidence = np.max(prediction)
    label = class_labels[pred_index]

    # Threshold confidence bisa kamu sesuaikan jika mau (misal > 0.6)
    if confidence < 0.5:
        label = "Tidak dikenali sebagai salah satu jenis jelatang (confidence rendah)"

    return render_template('index.html',
                           prediction=label,
                           confidence=round(confidence * 100, 2),
                           filename=filename)

# Route untuk akses gambar upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
