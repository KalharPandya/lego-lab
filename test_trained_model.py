import os
import glob
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.ops import box_iou
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configuration: update these paths as needed
VAL_IMAGES_DIR = r"datasets/output_split/val/images"
VAL_LABELS_DIR = r"datasets/output_split/val/labels"
IOU_THRESHOLD = 0.5
MODEL_PATH="./best.pt"
def load_yolo_labels(label_path, img_width, img_height):
    """
    Reads a YOLO label file and converts normalized coordinates to absolute (xyxy) format.
    Returns a list of boxes (each as [xmin, ymin, xmax, ymax]) and a list of class ids.
    """
    boxes = []
    classes = []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, "r") as f:
        for line in f.read().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, w, h = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            w *= img_width
            h *= img_height
            xmin = x_center - w / 2
            ymin = y_center - h / 2
            xmax = x_center + w / 2
            ymax = y_center + h / 2
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(int(cls_id))
    return boxes, classes

def evaluate_detection_on_image(img_path, label_path, iou_thresh=IOU_THRESHOLD):
    """
    Loads an image and its corresponding YOLO-format label file, runs the YOLO model,
    and computes per-image detection results.
    
    Returns:
        - TP, FP, FN, number of predictions, number of GT boxes, average confidence,
        - detections: a list of tuples (score, is_tp) for each prediction.
    """
    # Load image and get dimensions
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size

    # Run the model; assumed that model() returns results with boxes in xyxy absolute coordinates.
    results = model(img, imgsz=224, verbose=False)
    # Get predicted boxes and confidences
    if results and len(results) > 0:
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        pred_confs = results[0].boxes.conf.cpu().numpy()    # shape: (N,)
        num_preds = len(pred_boxes)
    else:
        pred_boxes = np.empty((0, 4))
        pred_confs = np.empty((0,))
        num_preds = 0

    # Load ground truth boxes from label file
    gt_boxes, gt_classes = load_yolo_labels(label_path, img_width, img_height)
    gt_boxes = np.array(gt_boxes)
    num_gt = len(gt_boxes)

    # If no predictions or no ground truths, set metrics accordingly.
    if num_preds == 0:
        return 0, 0, num_gt, num_preds, num_gt, 0.0, []
    if num_gt == 0:
        return 0, num_preds, 0, num_preds, 0, float(pred_confs.mean()), [(conf, 0) for conf in pred_confs]

    # Compute IoU between predicted boxes and ground truths.
    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    ious = box_iou(pred_tensor, gt_tensor)  # shape: (num_preds, num_gt)

    matched_gt = set()
    TP = 0
    FP = 0
    detections = []  # List of tuples: (confidence, is_tp)
    # For each prediction, check if it has a matching ground truth
    for i in range(ious.shape[0]):
        max_iou, max_idx = torch.max(ious[i], dim=0)
        if max_iou.item() >= iou_thresh and max_idx.item() not in matched_gt:
            TP += 1
            matched_gt.add(max_idx.item())
            detections.append((pred_confs[i], 1))
        else:
            FP += 1
            detections.append((pred_confs[i], 0))
    FN = num_gt - len(matched_gt)
    avg_conf = float(pred_confs.mean()) if num_preds > 0 else 0.0
    return TP, FP, FN, num_preds, num_gt, avg_conf, detections

def compute_average_precision(detections, total_gt):
    """
    Given a list of detections (each as (score, is_tp)) across the validation set,
    compute the Average Precision (AP) at the defined IoU threshold.
    This is a simplified calculation: sort detections by score, compute cumulative precision and recall,
    and integrate the precision-recall curve.
    """
    if not detections:
        return 0.0
    # Sort by descending confidence
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    for score, is_tp in detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt if total_gt > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    # Compute AP using the trapezoidal rule over the recall-precision curve
    # (This is a simple approximation.)
    ap = 0.0
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    # Append sentinel values at the beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(1, len(recalls)):
        delta = recalls[i] - recalls[i-1]
        ap += delta * precisions[i]
    return ap

# -------------------------------
# Detailed Evaluation Function with Confusion Matrix and mAP
# -------------------------------
def detailed_evaluation(val_images_dir, val_labels_dir, iou_thresh=IOU_THRESHOLD):
    """
    Runs evaluation on the entire validation set.
    Computes overall TP, FP, FN, precision, recall, F1 score, and Average Precision (AP).
    Also displays a confusion matrix-like bar chart.
    """
    image_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    total_TP = total_FP = total_FN = 0
    total_preds = 0
    total_gt = 0
    all_detections = []  # To accumulate (score, is_tp) for AP calculation
    
    for img_path in tqdm(image_paths, desc="Evaluating images", unit="image"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_labels_dir, base + ".txt")
        TP, FP, FN, num_preds, num_gt, avg_conf, detections = evaluate_detection_on_image(img_path, label_path, iou_thresh)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_preds += num_preds
        total_gt += num_gt
        all_detections.extend(detections)
    
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute AP (mAP50) for this evaluation
    ap = compute_average_precision(all_detections, total_gt)
    
    print("Detailed Evaluation on Validation Set")
    print("--------------------------------------")
    print(f"Total Ground Truth Boxes: {total_gt}")
    print(f"Total Predictions: {total_preds}")
    print(f"True Positives (TP): {total_TP}")
    print(f"False Positives (FP): {total_FP}")
    print(f"False Negatives (FN): {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Precision (mAP50): {ap:.4f}")
    
    # Plot a confusion matrix-like bar chart (TP, FP, FN)
    labels = ["TP", "FP", "FN"]
    values = [total_TP, total_FP, total_FN]
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["green", "red", "orange"])
    plt.title("Detection Summary")
    plt.ylabel("Count")
    plt.show()
    
    # Return dictionary of metrics for further analysis.
    return {
        "Total GT": total_gt,
        "Total Predictions": total_preds,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": ap
    }

# -------------------------------
# Main Testing Code
# -------------------------------
def main():
    # Load the pretrained YOLO model
    global model
    model = YOLO(MODEL_PATH)
    print("Pretrained YOLO model loaded.")
    
    # Run detailed evaluation on the validation set
    metrics = detailed_evaluation(VAL_IMAGES_DIR, VAL_LABELS_DIR, iou_thresh=IOU_THRESHOLD)
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()
