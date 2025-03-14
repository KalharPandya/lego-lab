import torch
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import cv2
import numpy as np

# Global configuration
MODEL_PATH = "./best_model.pt"

# Preprocessing parameters
SPARAM1, SRANGE1 = 10, 30
SPARAM2, SRANGE2 = 10, 30  # Adjust if needed
TARGET_SIZE = (224, 224)

# Load the YOLO model using the Ultralytics API
model = YOLO(MODEL_PATH)  # This loads your YOLO trained model

def preprocess_image(pil_img, target_size=TARGET_SIZE):
    """
    Preprocess the input PIL image:
      - Convert to OpenCV BGR format
      - Resize to target_size
      - Convert to HSV and apply mean shift filtering twice
      - Convert back to RGB and return as a PIL image
    """
    # Convert PIL image (RGB) to NumPy array and then to BGR
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize image
    resized = cv2.resize(img, target_size)
    
    # Convert to HSV and apply mean shift filtering twice
    image_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    segmented = cv2.pyrMeanShiftFiltering(image_hsv, sp=SPARAM1, sr=SRANGE1)
    segmented = cv2.pyrMeanShiftFiltering(segmented, sp=SPARAM2, sr=SRANGE2)
    
    # Convert back to BGR then to RGB
    final_image = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(final_image)

def predict_lego_count(images):
    """
    Accepts one or more images (PIL format), preprocesses them,
    runs YOLO detection, rescales the bounding boxes to the original image size,
    draws bounding boxes, and returns both the annotated image(s) and a summary string.
    """
    # Ensure we have a list of images
    if not isinstance(images, list):
        images = [images]
        
    annotated_images = []
    summaries = []
    
    for img in images:
        # Store original dimensions
        orig_w, orig_h = img.size  # (width, height)
        
        # Preprocess the image (assumed to be resized to fixed dimensions, e.g. 640x640)
        preprocessed_img = preprocess_image(img)  # your preprocessing function
        
        # Run YOLO detection on the preprocessed image
        results = model(preprocessed_img, imgsz=224)
        if results and len(results) > 0:
            # Get normalized detection boxes (xyxy format, values between 0 and 1)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
            count = len(boxes)
            if count > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                avg_conf = confs.mean() * 100
            else:
                avg_conf = 0.0
            summary = f"Predicted LEGO pieces: {count} (avg conf: {avg_conf:.1f}%)"
        else:
            summary = "No detections."
            boxes = np.empty((0,4))
            count = 0
        
        summaries.append(summary)
        
        # Unnormalize bounding boxes from normalized coordinates to original image dimensions.
        boxes_unnorm = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Multiply by original image width and height.
            x1 = x1 * orig_w / TARGET_SIZE[0]
            y1 = y1 * orig_h / TARGET_SIZE[0]
            x2 = x2 * orig_w / TARGET_SIZE[0]
            y2 = y2 * orig_h / TARGET_SIZE[0]
            boxes_unnorm.append([x1, y1, x2, y2])
        boxes_unnorm = np.array(boxes_unnorm)
        
        # Convert original PIL image to NumPy array (RGB) and then to BGR for OpenCV
        orig_np = np.array(img)
        image_for_draw = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        
        # Draw each bounding box on the image
        for box in boxes_unnorm:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image_for_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert drawn image back from BGR to RGB and then to PIL format
        annotated_np = cv2.cvtColor(image_for_draw, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(annotated_np)
        annotated_images.append(annotated_img)
    
    if len(annotated_images) == 1:
        return annotated_images[0], summaries[0]
    return annotated_images, "\n".join(summaries)

# Define the Gradio Interface with two outputs: image(s) and textbox.
iface = gr.Interface(
    fn=predict_lego_count,
    inputs=gr.Image(type="pil", label="Upload Images"),
    outputs=[gr.Image(type="pil", label="Annotated Images"), gr.Textbox(label="Detection Summary")],
    title="LEGO Piece Counter - YOLO Model Tester",
    description="Upload one or more images (JPG/PNG) to predict the number of LEGO pieces using a YOLO detection model. The images are preprocessed and annotated with bounding boxes.",
)

iface.launch(server_port=7891, server_name="localhost")
