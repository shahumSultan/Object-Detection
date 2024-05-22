import numpy as np
import random
import cv2


def getObjects(uploaded_file):
    # Loading Yolo
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = net.getUnconnectedOutLayersNames()

    # Convert the uploaded file to a OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize the image for faster processing
    image = cv2.resize(image, None, fx=0.4, fy=0.4)

    # Image Conversion and Network
    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detecting all the objects in the image with score > 70%
    detected_labels = []
    colors = {}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.5:
                # Get the coordinates of the bounding box
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                # Calculate coordinates for the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Draw bounding box on the image
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                # Add label to the detected_labels list
                detected_labels.append(classes[class_id])
                # Store color for the class
                if classes[class_id] not in colors:
                    colors[classes[class_id]] = color

    # Get unique labels
    unique_labels = list(set(detected_labels))

    return unique_labels, image, colors
