from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load EfficientDet-D0 from TensorFlow Hub
efficientdet_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
efficientdet = hub.load(efficientdet_url)

# Define class labels
class_labels = {
     1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    45: 'plate',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    66: 'mirror',
    67: 'dining table',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    91: 'hair brush',
}

uploads_dir = 'uploads'
results_dir = 'static/result'

# Ensure the uploads and results directories exist
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def process_image(file_path):
    # Load and preprocess the image for EfficientDet
    image = cv2.imread(file_path)
    input_image = tf.convert_to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=tf.uint8)
    input_image = tf.expand_dims(input_image, axis=0)

    # Run inference
    detections = efficientdet(input_image)

    # Post-process the detections
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    # Draw bounding boxes and labels on the image
    h, w, _ = image.shape
    for box, score, class_id in zip(boxes, scores, classes):
        if score > 0.5:  # Adjust the confidence threshold as needed
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Get the class label from the mapping
            label = class_labels.get(int(class_id), 'Unknown')

            # Draw bounding box and label
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the result image
    result_path = os.path.join(results_dir, 'result.jpg')
    cv2.imwrite(result_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # If it's a POST request, process the uploaded image
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)

            # Process the image
            process_image(file_path)

            # Redirect to the result page
            return redirect(url_for('result'))

    # If it's a GET request, render the upload page
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image
        # file_path = os.path.join(uploads_dir, file.filename)
        file_path = os.path.join(results_dir, file.filename)
        file.save(file_path)

        # Process the image
        process_image(file_path)

        # Redirect to the result page
        return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('result.html', object_count=10, filename='result.jpg')

if __name__ == '__main__':
    app.run(debug=True)
