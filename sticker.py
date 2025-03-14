import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import gradio as gr

# Set up color for sticker border (white)
BORDER_COLOR = (255, 255, 255)

# COCO classes (used by Mask R-CNN)
CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# We are focusing on the person class for sticker creation.
TARGET_CLASSES = ['person']

def get_prediction(img, threshold=0.7):
    """Get model predictions filtered for person detection."""
    # Convert image to tensor
    transform = F.to_tensor(img)
    prediction = model([transform])
    
    masks = []
    boxes = []
    labels = []
    scores = []
    
    pred_classes = [CLASSES[i] for i in prediction[0]['labels']]
    pred_masks = prediction[0]['masks'].detach().cpu().numpy()
    pred_boxes = prediction[0]['boxes'].detach().cpu().numpy()
    pred_scores = prediction[0]['scores'].detach().cpu().numpy()
    
    for i, score in enumerate(pred_scores):
        if score > threshold and pred_classes[i] in TARGET_CLASSES:
            masks.append(pred_masks[i][0])
            boxes.append(pred_boxes[i])
            labels.append(pred_classes[i])
            scores.append(score)
            
    return masks, boxes, labels, scores

def create_sticker(img, mask, border_thickness=3):
    """
    Create a sticker image by applying the mask to the image.
    The area outside the mask is made transparent and a border is drawn.
    """
    # Create a binary mask
    mask_bin = (mask > 0.5).astype(np.uint8)
    
    # Convert the image to BGRA (add alpha channel)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = (mask_bin * 255).astype(np.uint8)
    sticker = cv2.merge([b_channel, g_channel, r_channel, alpha_channel])
    
    # Draw a border by finding and drawing contours on the binary mask.
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(sticker, [cnt], -1, BORDER_COLOR, border_thickness)
    
    return sticker

def process_image(input_img):
    """
    Process the uploaded image:
      - Run detection for persons
      - If found:
          * If one person is detected, create a sticker from that mask.
          * If multiple persons are detected, combine their masks and then create a sticker.
      - If no person is detected, return the original image with a notification.
    """
    # Ensure image is in the correct format (BGR)
    if input_img.shape[2] == 4:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
    
    with torch.no_grad():
        masks, boxes, labels, scores = get_prediction(input_img)
    
    if len(masks) == 0:
        # If no person is detected, add a notification to the image.
        print("No person detected")
        output_img = input_img.copy()
        cv2.putText(output_img, "No person detected.", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
        
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, "No person detected.", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return output_img
    elif len(masks) == 1:
        # Use the single detected person's mask.
        sticker = create_sticker(input_img, masks[0])
    else:
        # Combine multiple person masks via pixel-wise maximum.
        combined_mask = np.zeros_like(masks[0])
        for m in masks:
            combined_mask = np.maximum(combined_mask, m)
        sticker = create_sticker(input_img, combined_mask)
    
    return sticker

# Load pre-trained Mask R-CNN model
print("Loading Mask R-CNN model...")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Create a Gradio interface with an image upload widget
description = """
Upload an image containing one or more persons. The model will detect the person(s) and convert them into a sticker with a transparent background and a white border.
"""

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Sticker Output"),
    title="Person Sticker Maker",
    description=description,
    allow_flagging="never"
)

iface.launch(server_name='0.0.0.0', server_port=7890)
