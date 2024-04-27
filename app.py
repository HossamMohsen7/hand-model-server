import base64
import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request
import json

# from flask_socketio import SocketIO, emit
import joblib
from flask_sock import Sock

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"

# sock = Sock()
# sock.init_app(app)
# socketio = SocketIO(app, cors_allowed_origins="*")


class handDetector:
    def __init__(
        self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplex,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
        return lmlist, bbox


def extract_features(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:  # Check if image is loaded
        img = cv2.resize(img, size)  # Resize image
        return img.flatten()  # Flatten the image to a vector
    else:
        return None


def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]

    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)

    # fh = open("imageToSave.png", "wb")
    # fh.write(image_bytes)
    # fh.close()

    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# @socketio.on("connect")
# def test_connect():
#     print("Connected")
#     emit("my_response", {"data": "Connected"})

categories = [
    ["fine", "بخير"],
    ["hello", "مرحبا, كيف حالك؟"],
    ["stop", "قف"],
    ["yes", "نعم"],
]
model_en = joblib.load("hand_gesture_model.pkl")
detector = handDetector()


def receive_image(image, lang):
    try:
        # Decode the base64-encoded image data
        print("Got image")
        img = base64_to_image(image)

        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)
        if bbox:
            print("Found hand")
            x, y, x2, y2 = bbox
            hand_img = img[y:y2, x:x2]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(
                hand_img, (64, 64)
            )  # Ensure the image is resized as per training
            hand_img_flat = hand_img.flatten().reshape(1, -1)

            prediction = model_en.predict(hand_img_flat)
            probability = model_en.predict_proba(hand_img_flat).max()

            sequence = prediction[0]

            if lang == "arabic":
                for category in categories:
                    if category[0] == prediction[0]:
                        sequence = category[1]
                        break

            # # Encode the processed image as a JPEG-encoded base64 string
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            # result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
            # processed_img_data = base64.b64encode(frame_encoded).decode()

            # # Prepend the base64-encoded string with the data URL prefix
            # b64_src = "data:image/jpg;base64,"
            # processed_img_data = b64_src + processed_img_data

            # Send the processed image back to the client
            # emit("processed_image", processed_img_data)
            return {"data": sequence, "probability": probability}
            # emit("prediction", {"data": prediction[0], "probability": probability})
    except Exception as e:
        print(e)


@app.route("/process", methods=["POST"])
def process():
    request_data = request.get_json()

    image = request_data["image"]
    lang = request_data["lang"]

    prediction = receive_image(image, lang)

    if prediction:
        return prediction
    else:
        return {"data": "No hand detected"}


import asyncio
import http
import signal

import websockets


async def health_check(path, request_headers):
    if path == "/healthz":
        return http.HTTPStatus.OK, [], b"OK\n"


async def echo(websocket):
    async for message in websocket:
        # json
        data = json.loads(message)
        image = data["image"]
        lang = data["lang"]
        prediction = receive_image(image, lang)
        if prediction:
            await websocket.send(json.dumps(prediction))
        else:
            await websocket.send(json.dumps({"data": "No hand detected"}))


async def main():
    # Set the stop condition when receiving SIGTERM.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    async with websockets.serve(
        echo,
        host="",
        port=8080,
        max_size=10 * 1024 * 1024,
        process_request=health_check,
    ):
        await stop


if __name__ == "__main__":
    asyncio.run(main())


# socketio.on_event("image", receive_image, namespace="/")

# if __name__ == "__main__":
#     app.run(debug=True, port=5000, host="0.0.0.0")
# socketio.run(app, debug=True, port=5000, host="0.0.0.0")
