'''
created by rknguyen
'''

import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from sklearn import neighbors
from helper import confirm_checkin
from modules.models import ArcFaceModel
from classifier import knn_init, add_embeds
from modules.utils import load_yaml, l2_norm
from flask import Flask, render_template, request, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load ResNet50 model
print('[*] Model is importing')
model = ArcFaceModel(
    size=112, backbone_type='ResNet50', training=False)
model.load_weights(tf.train.latest_checkpoint('./checkpoints/arc_res50'))

# Create daemon & init KNN
clf = knn_init()
app = Flask(__name__)
app.secret_key = "@#!ILMHSOMUCH*@@!@"


def get_embeds(base64_image):
    global model
    image = base64.b64decode(base64_image)
    numpy_image = np.frombuffer(image, dtype=np.uint8)
    face = cv2.imdecode(numpy_image, 1)
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.
    img = np.expand_dims(face, 0)
    embeds = l2_norm(model(img))
    return embeds


@app.route("/", methods=["GET"])
def index():
    return jsonify({"success": True})


@app.route("/register", methods=["POST"])
def register():
    user_id = request.form["user_id"]
    base64_image = request.form["image"]

    embeds = get_embeds(base64_image)
    add_embeds(embeds, user_id)

    global clf
    clf = knn_init()

    return jsonify({
        "success": True,
        "message": "Thêm khuôn mặt thành công"
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    base64_image = request.form["image"]

    embeds = get_embeds(base64_image)

    try:
        predict = clf.predict(embeds)[0]
    except:
        return jsonify({
            "error": True,
            "message": "Không thể nhận diện được khuôn mặt"
        })

    confirm_checkin(predict, base64_image)
    return jsonify({
        "success": True,
        "predict": predict
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="1210")
