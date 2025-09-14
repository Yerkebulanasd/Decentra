# app.py
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io


MODEL_PATH = 'model.h5'
IMG_SIZE = (224,224)


app = Flask(__name__)


# Загрузим модель
model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(file_bytes):
img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
img = img.resize(IMG_SIZE)
arr = np.array(img).astype('float32') / 255.0
return np.expand_dims(arr, axis=0)


@app.route('/')
def index():
return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
if 'file' not in request.files:
return jsonify({'error':'no file'}), 400
f = request.files['file']
img_bytes = f.read()
x = prepare_image(img_bytes)


# модель возвращает два выхода: dirty, damaged
preds = model.predict(x)
# model.predict может вернуть list of arrays или dict depending on saved model.
if isinstance(preds, list) and len(preds) == 2:
dirty_p = float(preds[0][0][0])
damaged_p = float(preds[1][0][0])
elif isinstance(preds, dict):
dirty_p = float(preds['dirty'][0][0])
damaged_p = float(preds['damaged'][0][0])
else:
# если модель вернула одномерный массив [p_dirty, p_damaged]
arr = np.array(preds).ravel()
dirty_p, damaged_p = float(arr[0]), float(arr[1])


out = {
'dirty_score': dirty_p,
'dirty': int(dirty_p > 0.5),
'damaged_score': damaged_p,
'damaged': int(damaged_p > 0.5)
}
return jsonify(out)


if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000, debug=True)
