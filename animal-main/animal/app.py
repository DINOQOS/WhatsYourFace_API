import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 모델 불러오기
model = tf.keras.models.load_model('keras_model.h5')

# 클래스 레이블 정의
labels = ['dog', 'dolphin', 'duck', 'elephant', 'fox', 'goldfish', 'horse', 'lion', 'mouse', 'ox', 'pig', 'sheep', 'snake', 'swan', 'wolf', 'zebra']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 전송 받은 파일이 이미지 파일 형태가 아니면 400을 반환
    if 'image' not in request.files:
        return jsonify({'error': 'no file upload'}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    image = Image.open(f)
    image = image.resize((224, 224))  # 모델에 맞는 크기로 조정

    image = image.convert('RGB')

    # 이미지를 numpy 배열로 변환
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

    # 모델에 이미지 데이터 전달
    pred = model.predict(image_array)
    pred_class = np.argmax(pred, axis=-1)[0]
    prediction_label = labels[pred_class]

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)