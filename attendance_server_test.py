from flask import Flask, request, jsonify
import numpy as np
import cv2
import matplotlib.pyplot as plt



app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def receive_image():

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    print("image received")
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image from esp32cam")
    plt.show()  

    return jsonify({
        "status": "received"
    })

if __name__ == "main":
    app.run(host = "0.0.0.0", port=5000)