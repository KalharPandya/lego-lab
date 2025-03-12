#!/usr/bin/env python3
"""
lego_detection.py

This script:
1. Checks the environment (OpenCV, PyTorch, CUDA).
2. Processes raw images via mean shift and caches them.
3. Splits the dataset into train/val/test.
4. Parses XML to extract bounding boxes for each LEGO piece.
5. Trains a Faster R-CNN model to detect LEGO pieces. 
6. Allows further training or evaluation on val/test sets.
"""

import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from PIL import Image

# --- For detection evaluation, we can rely on TorchVision references or do something ad-hoc:
from torchvision.ops import box_iou

torch.cuda.empty_cache()

# -------------------------------
# Configuration & Directory Paths
# -------------------------------
RAW_IMAGES_DIR = r"lego_dataset\dataset_20210629145407_top_600\images"
PROCESSED_IMAGES_DIR = r"output"
ANNOTATIONS_DIR = r"lego_dataset\dataset_20210629145407_top_600\annotations"
OUTPUT_SPLIT_DIR = r"output_split"
ANNOTATION_CACHE_FILE = "annotation_bboxes.npy"
MODEL_PATH = "./fasterrcnn_lego.pth"
TARGET_SIZE = (224, 224)


# Check if RAW_IMAGES_DIR and ANNOTATIONS_DIR exist
if not (os.path.exists(RAW_IMAGES_DIR) and os.path.exists(ANNOTATIONS_DIR)):
    print("Raw images or annotation directory not found. Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("dreamfactor/biggest-lego-dataset-600-parts")
    print("Path to dataset files:", path)
    # Optionally, you can update RAW_IMAGES_DIR and ANNOTATIONS_DIR based on the downloaded path.
    # For example:
    RAW_IMAGES_DIR = os.path.join(path, "images")
    ANNOTATIONS_DIR = os.path.join(path, "annotations")
else:
    print("Raw images and annotation directories found.")



# Mean shift parameters for image processing
SPARAM1, SRANGE1 = 10, 30
SPARAM2, SRANGE2 = 10, 40

# If you want to treat each distinct "part number" (the <name> in XML) as a unique class:
# Provide a mapping from the part names to integer class IDs.
# Example below is *just illustrative*. Populate with your actual known part names if you want multiple classes.
# If you want a single class (just "lego_piece"), set PART_NAME_TO_ID = {"lego_piece": 1}
PART_NAME_TO_ID = {
    # Suppose we found these part names frequently in the dataset:
    # '3029': 1,
    # '3706': 2,
    # '22484': 3,
    # '32523': 4,
    # ...
    # Or just unify them all under a single class:
    "lego_piece": 1
}
NUM_CLASSES = len(PART_NAME_TO_ID) + 1  # +1 for background class (0)

# -------------------------------
# Block 1: Environment Check
# -------------------------------
def check_environment():
    print("=== Environment Check ===")
    print("OpenCV Version:", cv2.__version__)
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected.")
    print()

# -------------------------------
# Block 2: Image Processing & Caching
# -------------------------------
def process_images(raw_dir=RAW_IMAGES_DIR, output_dir=PROCESSED_IMAGES_DIR, target_size=TARGET_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    processed_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    if processed_files:
        print(f"Processed images already exist in '{output_dir}' (found {len(processed_files)} files). Skipping processing.")
        return processed_files
    else:
        print("No processed images found. Processing raw images now...")

        def process_and_save(image_path, output_dir, target_size=(224, 224)):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to load {image_path}")
                    return None
                # Resize image
                resized = cv2.resize(image, target_size)
                # Convert to HSV and apply mean shift filtering twice
                image_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                segmented = cv2.pyrMeanShiftFiltering(image_hsv, sp=SPARAM1, sr=SRANGE1)
                segmented = cv2.pyrMeanShiftFiltering(segmented, sp=SPARAM2, sr=SRANGE2)
                # Convert back to BGR
                final_image = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
                # Save processed image
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, final_image)
                return output_path
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        raw_image_files = glob.glob(os.path.join(raw_dir, "*.jpg"))
        for img_path in tqdm(raw_image_files, desc="Processing images", unit="file"):
            process_and_save(img_path, output_dir, target_size=target_size)
        processed_files = glob.glob(os.path.join(output_dir, "*.jpg"))
        print(f"Processed and saved {len(processed_files)} images to '{output_dir}'\n")
        return processed_files

# -------------------------------
# Block 3: Dataset Splitting
# -------------------------------
def split_dataset(processed_dir=PROCESSED_IMAGES_DIR, split_dir=OUTPUT_SPLIT_DIR):
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")
    
    # Create directories if they do not exist
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Check if dataset is already split (e.g., train_dir has images)
    existing_train_files = glob.glob(os.path.join(train_dir, "*.jpg"))
    if existing_train_files:
        print(f"Dataset already split (found {len(existing_train_files)} train images). Skipping splitting.\n")
        return
    else:
        print("Splitting dataset into train/val/test...")
        all_files = glob.glob(os.path.join(processed_dir, "*.jpg"))
        if not all_files:
            raise ValueError("No images found. Please check your image processing step or input directory.")

        # First split into train+val and test (80/20)
        train_val_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        # Then split train_val into train and val (80/20 of train+val -> overall ~64/16/20)
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
        
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")
        
        def copy_files(file_list, target_dir):
            for file_path in tqdm(file_list, desc=f"Copying -> {os.path.basename(target_dir)}", unit="file"):
                shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))
        
        copy_files(train_files, train_dir)
        copy_files(val_files, val_dir)
        copy_files(test_files, test_dir)
        print("Dataset split and files copied successfully.\n")

# -------------------------------
# Block 4: XML Annotation Processing for Bounding Boxes
# -------------------------------
def parse_xml_bboxes(xml_file):
    """
    Parse the XML file to get bounding boxes and part labels.
    Return a list of (xmin, ymin, xmax, ymax, class_label).
    """
    bboxes = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_tag is None or bndbox is None:
                continue

            # Convert part name to ID if it exists in PART_NAME_TO_ID
            # If you want each distinct part name to be a class: 
            #   class_id = PART_NAME_TO_ID.get(name_tag.text, 0)  # 0 is background if unknown
            # If you want all pieces to be the same class:
            class_id = PART_NAME_TO_ID["lego_piece"]  # treat them all as class 1

            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bboxes.append((xmin, ymin, xmax, ymax, class_id))
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    
    return bboxes

def build_annotation_dict(annotations_dir=ANNOTATIONS_DIR, cache_file=ANNOTATION_CACHE_FILE):
    """
    Build a dictionary: { base_filename: list of bboxes }
       where each bbox is a tuple (xmin, ymin, xmax, ymax, class_id).
    """
    if os.path.exists(cache_file):
        print(f"Loading cached bounding boxes from '{cache_file}'...")
        data = np.load(cache_file, allow_pickle=True).item()
        annotation_dict = data["annotation_dict"]
    else:
        print("Parsing XML annotation files for bounding boxes...")
        xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
        annotation_dict = {}

        def process_single_xml(xml_path):
            base = os.path.splitext(os.path.basename(xml_path))[0]
            bboxes = parse_xml_bboxes(xml_path)
            return base, bboxes

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_single_xml, xf) for xf in xml_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="XML -> BBoxes"):
                base_name, bbox_list = future.result()
                annotation_dict[base_name] = bbox_list
        
        np.save(cache_file, {"annotation_dict": annotation_dict})
        print(f"Cached bounding boxes to '{cache_file}'.")
    
    # Print stats
    total_bboxes = sum(len(v) for v in annotation_dict.values())
    print(f"Total bounding boxes: {total_bboxes}")
    
    # Optional: show distribution of bounding boxes per image
    bbox_counts = [len(v) for v in annotation_dict.values()]
    plt.figure()
    bins = np.arange(0, max(bbox_counts) + 2) - 0.5
    plt.hist(bbox_counts, bins=bins, edgecolor='black')
    plt.title("Histogram of bounding boxes per image")
    plt.xlabel("# of bounding boxes")
    plt.ylabel("Frequency")
    plt.xticks(range(0, max(bbox_counts)+1))
    plt.show()
    
    return annotation_dict

# -------------------------------
# Dataset Class for Detection
# -------------------------------
class LegoDetectionDataset(Dataset):
    """
    Loads processed images, returns (image, target_dict) for detection.
    The target_dict includes:
       boxes: FloatTensor [N, 4]  (xmin, ymin, xmax, ymax)
       labels: Int64Tensor [N]
       image_id: Int
       area: FloatTensor [N]
       iscrowd: Int64Tensor [N]  (usually all zero for this dataset)
    """
    def __init__(self, image_dir, annotation_dict, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        
        # Map base filenames -> full paths
        self.image_dict = {}
        for path in self.image_files:
            base = os.path.splitext(os.path.basename(path))[0]
            self.image_dict[base] = path
        
        # We store bounding boxes for each base name
        self.annotation_dict = annotation_dict
        
        # Only keep samples that have entries in annotation_dict
        self.samples = []
        for base in self.image_dict:
            if base in self.annotation_dict:
                self.samples.append(base)
            else:
                # If no annotation found, skip
                pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        base = self.samples[idx]
        img_path = self.image_dict[base]
        bboxes_raw = self.annotation_dict[base]  # list of (xmin, ymin, xmax, ymax, class_id)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Convert bounding boxes to tensors
        boxes = []
        labels = []
        for (xmin, ymin, xmax, ymax, cls_id) in bboxes_raw:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # For area, we do (xmax - xmin)*(ymax - ymin)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        # Construct the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),  # some unique ID
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

# -------------------------------
# Block 5: Create the Detection Model
# -------------------------------
def create_fasterrcnn_model(num_classes):
    """
    Create and return a Faster R-CNN model with a ResNet50 FPN backbone.
    num_classes includes the background class.
    """
    # Use a torchvision model with pretrained weights
    # model = models.detection.fasterrcnn_mobilenet_v3_ large_fpn(weights='FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT')

    model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our desired num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# -------------------------------
# Block 6: Training and Evaluation
# -------------------------------
def collate_fn(batch):
    """
    Custom collate function for DataLoader when loading variable # of objects per image.
    """
    return tuple(zip(*batch))

def train_model(model, train_dataset, device, num_epochs=10, lr=0.001, batch_size=4):
    """
    Training loop for the object detection model.
    Logs current batch loss every 10 batches.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += float(losses)
            
            # Log current loss every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {losses.item():.4f}")
        
        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def evaluate_model(model, dataset, device, max_samples=None, iou_thresh=0.5):
    """
    Evaluate the model using mean Average Precision (mAP) at an IoU threshold of 0.5.
    The evaluation aggregates detections across the dataset, sorts them by confidence,
    and computes the area under the precision–recall curve as an approximation of mAP.
    """
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_detections = []  # List of tuples: (score, is_true_positive)
    n_total_gts = 0      # Total number of ground-truth boxes across all images
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating", unit="img")):
            if max_samples is not None and idx >= max_samples:
                break
            
            images = [img.to(device) for img in images]
            outputs = model(images)  # outputs is a list with one element (batch size = 1)
            
            pred_boxes = outputs[0]["boxes"].cpu()  # shape: [N_pred, 4]
            scores = outputs[0]["scores"].cpu()       # shape: [N_pred]
            gt_boxes = targets[0]["boxes"].cpu()        # shape: [N_gt, 4]
            
            n_total_gts += len(gt_boxes)
            
            if len(pred_boxes) == 0:
                continue  # No detections, move to next image
            
            # Sort predictions by descending confidence score
            order = scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            scores = scores[order]
            
            # Keep track of whether each ground truth has been detected
            detected = [False] * len(gt_boxes)
            for i in range(len(pred_boxes)):
                box = pred_boxes[i]
                score = scores[i].item()
                if len(gt_boxes) > 0:
                    ious = box_iou(box.unsqueeze(0), gt_boxes)  # shape: [1, N_gt]
                    max_iou, max_iou_idx = ious.max(dim=1)
                    if max_iou.item() >= iou_thresh and not detected[max_iou_idx.item()]:
                        is_tp = 1  # True positive
                        detected[max_iou_idx.item()] = True
                    else:
                        is_tp = 0  # False positive
                else:
                    is_tp = 0
                all_detections.append((score, is_tp))
    
    if len(all_detections) == 0:
        print("No detections were made.")
        return
    
    # Sort all detections by score in descending order
    all_detections.sort(key=lambda x: x[0], reverse=True)
    tp_flags = np.array([d[1] for d in all_detections])
    fp_flags = 1 - tp_flags
    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (n_total_gts + 1e-6)
    
    # Compute Average Precision (AP) using the trapezoidal rule
    ap = np.trapz(precision, recall)
    
    print(f"\nEvaluation results on {max_samples if max_samples is not None else len(dataset)} images:")
    print(f"mAP @ IoU=0.5: {ap:.3f}\n")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    check_environment()
    process_images()
    split_dataset()
    annotation_dict = build_annotation_dict()  # base -> list of (xmin, ymin, xmax, ymax, class_id)
    
    # Directories for train, val, test
    train_dir = os.path.join(OUTPUT_SPLIT_DIR, "train")
    val_dir = os.path.join(OUTPUT_SPLIT_DIR, "val")
    test_dir = os.path.join(OUTPUT_SPLIT_DIR, "test")
    
    # Transforms (typical for detection):
    data_transform = transforms.Compose([
        transforms.ToTensor()  # convert PIL to Tensor in [0,1]
        # Optionally, add data augmentation (random flip, color jitter, etc.)
    ])
    
    # Create Datasets
    train_dataset = LegoDetectionDataset(train_dir, annotation_dict, transform=data_transform)
    val_dataset = LegoDetectionDataset(val_dir, annotation_dict, transform=data_transform)
    test_dataset = LegoDetectionDataset(test_dir, annotation_dict, transform=data_transform)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create/Load the model
    model = create_fasterrcnn_model(NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        print("Existing model checkpoint found. Loading from disk...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print("Model loaded.")
    else:
        print("No existing model found; training from scratch...")
        train_model(model, train_dataset, device, num_epochs=1, lr=0.0001, batch_size=16)
    
    # Interactive prompt
    while True:
        print("\nSelect an option:")
        print("1: Train further")
        print("2: Evaluate on validation set")
        print("3: Evaluate on test set")
        print("4: Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            try:
                additional_epochs = int(input("Enter number of additional epochs to train: "))
            except ValueError:
                print("Invalid input. Please enter an integer.")
                continue
            train_model(model, train_dataset, device, num_epochs=additional_epochs, lr=0.0001, batch_size=16)

        elif choice == '2':
            user_in = input("Enter number of validation samples to evaluate (0 for all): ")
            try:
                val_samples = int(user_in)
                if val_samples <= 0:
                    val_samples = None
            except ValueError:
                val_samples = None
            evaluate_model(model, val_dataset, device, max_samples=val_samples)

        elif choice == '3':
            user_in = input("Enter number of test samples to evaluate (0 for all): ")
            try:
                test_samples = int(user_in)
                if test_samples <= 0:
                    test_samples = None
            except ValueError:
                test_samples = None
            evaluate_model(model, test_dataset, device, max_samples=test_samples)

        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
