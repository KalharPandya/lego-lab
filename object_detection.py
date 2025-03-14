#!/usr/bin/env python3
"""
lego_detection.py

This script:
1. Checks the environment (OpenCV, PyTorch, CUDA).
2. Processes raw images via mean shift filtering and caches them.
3. Splits the dataset into train/val/test with proper subfolder structure.
4. Parses XML to extract bounding boxes for each LEGO piece.
5. Converts XML annotations to YOLO-format label files (using letterbox transformation).
6. Trains a YOLO model (using YOLO11 Train mode) to detect LEGO pieces.
7. Allows further training or evaluation on val/test sets.
"""

import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub

from ultralytics import YOLO
torch.cuda.empty_cache()

# -------------------------------
# Configuration & Directory Paths
# -------------------------------
RAW_IMAGES_DIR = r"lego_dataset\dataset_20210629145407_top_600\images"
PROCESSED_IMAGES_DIR = r"output"
ANNOTATIONS_DIR = r"lego_dataset\dataset_20210629145407_top_600\annotations"
# Updated split folder structure under a 'datasets' folder
OUTPUT_SPLIT_DIR = r"datasets/output_split"
MODEL_PATH = "./best.pt"  # save YOLO checkpoint here
TARGET_SIZE = (224, 224)

# Create necessary directories if they don’t exist
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

# Download dataset if empty
if not os.listdir(RAW_IMAGES_DIR) or not os.listdir(ANNOTATIONS_DIR):
    print("Raw images or annotation directory is empty. Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("dreamfactor/biggest-lego-dataset-600-parts")
    print("Path to dataset files:", path)
    source_images = os.path.join(path, "dataset_20210629145407_top_600", "images")
    source_annotations = os.path.join(path, "dataset_20210629145407_top_600", "annotations")
    if os.path.exists(source_images):
        for file in os.listdir(source_images):
            shutil.copy(os.path.join(source_images, file), RAW_IMAGES_DIR)
    if os.path.exists(source_annotations):
        for file in os.listdir(source_annotations):
            shutil.copy(os.path.join(source_annotations, file), ANNOTATIONS_DIR)
    print("Dataset copied successfully!")

# Mean shift parameters for image processing
SPARAM1, SRANGE1 = 10, 30
SPARAM2, SRANGE2 = 10, 40

# If you want to treat all pieces as a single class:
PART_NAME_TO_ID = {"lego_piece": 1}
NUM_CLASSES = len(PART_NAME_TO_ID) + 1  # +1 for background (not used in YOLO, kept for reference)

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
# Letterbox Resizing Function
# -------------------------------
def letterbox_image(img, target_size, color=(114,114,114)):
    """
    Resize image to target_size while preserving aspect ratio.
    Adds padding to reach the target size.
    Returns the resized (letterboxed) image, the scaling factor, and the padding (pad_x, pad_y).
    """
    h0, w0 = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w0, target_h / h0)
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    padded = cv2.copyMakeBorder(resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x, 
                                cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_x, pad_y

# -------------------------------
# Block 2: Image Processing & Caching
# -------------------------------
def process_and_save(image_path, output_dir, target_size=TARGET_SIZE):
    """
    Reads an image from image_path, applies letterbox resizing (preserving aspect ratio),
    processes it (using mean shift filtering), and saves the processed image to output_dir.
    Returns the path to the saved image.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load {image_path}")
            return None
        # Use letterbox resize to preserve aspect ratio
        letterboxed, scale, pad_x, pad_y = letterbox_image(image, target_size)
        
        # Convert the letterboxed image to HSV
        image_hsv = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2HSV)
        # Apply mean shift filtering twice
        segmented = cv2.pyrMeanShiftFiltering(image_hsv, sp=SPARAM1, sr=SRANGE1)
        segmented = cv2.pyrMeanShiftFiltering(segmented, sp=SPARAM2, sr=SRANGE2)
        # Convert back to BGR
        final_image = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
        
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, final_image)
        return output_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images(raw_dir=RAW_IMAGES_DIR, output_dir=PROCESSED_IMAGES_DIR, target_size=TARGET_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    processed_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    if processed_files:
        print(f"Processed images already exist in '{output_dir}' (found {len(processed_files)} files). Skipping processing.")
        return processed_files
    else:
        print("No processed images found. Processing raw images now...")
        raw_image_files = glob.glob(os.path.join(raw_dir, "*.jpg"))
        with ProcessPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(process_and_save, img_path, output_dir, target_size): img_path 
                       for img_path in raw_image_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", unit="file"):
                future.result()
        processed_files = glob.glob(os.path.join(output_dir, "*.jpg"))
        tqdm.write(f"Processed and saved {len(processed_files)} images to '{output_dir}'\n")
        return processed_files

# -------------------------------
# Block 3: Dataset Splitting with YOLO Structure
# -------------------------------
def split_dataset(processed_dir=PROCESSED_IMAGES_DIR, split_dir=OUTPUT_SPLIT_DIR):
    # For each split, create subfolders: images and labels
    splits = ['train', 'val', 'test']
    for s in splits:
        img_dir = os.path.join(split_dir, s, "images")
        lbl_dir = os.path.join(split_dir, s, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    existing_train_images = glob.glob(os.path.join(split_dir, "train", "images", "*.jpg"))
    if existing_train_images:
        print(f"Dataset already split (found {len(existing_train_images)} train images). Skipping splitting.\n")
        return
    else:
        print("Splitting dataset into train/val/test...")
        all_files = glob.glob(os.path.join(PROCESSED_IMAGES_DIR, "*.jpg"))
        if not all_files:
            raise ValueError("No images found. Please check your image processing step or input directory.")
        from sklearn.model_selection import train_test_split
        train_val_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")

        def copy_files(file_list, target_dir):
            for file_path in tqdm(file_list, desc=f"Copying -> {os.path.basename(target_dir)}", unit="file"):
                shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))
        
        copy_files(train_files, os.path.join(split_dir, "train", "images"))
        copy_files(val_files, os.path.join(split_dir, "val", "images"))
        copy_files(test_files, os.path.join(split_dir, "test", "images"))
        print("Dataset split and images copied successfully.\n")

# -------------------------------
# Block 4: XML Parsing with Original Dimensions
# -------------------------------
def parse_xml_bboxes(xml_file):
    """
    Parse the XML file to get bounding boxes and part labels.
    Also extracts original image width and height from the <size> element.
    Returns a tuple: (list of bboxes, orig_width, orig_height)
    Each bbox is (xmin, ymin, xmax, ymax, class_id).
    """
    bboxes = []
    orig_width = None
    orig_height = None
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        if size is not None:
            orig_width = int(size.find("width").text)
            orig_height = int(size.find("height").text)
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_tag is None or bndbox is None:
                continue
            class_id = PART_NAME_TO_ID.get("lego_piece", 1)
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bboxes.append((xmin, ymin, xmax, ymax, class_id))
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return bboxes, orig_width, orig_height

# -------------------------------
# Block 5: Build Annotation Dictionary
# -------------------------------
def build_annotation_dict(annotations_dir, cache_file="annotation_bboxes.npy"):
    """
    Build a dictionary: { base_filename: (list of bboxes, orig_width, orig_height) }
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
            bboxes, orig_w, orig_h = parse_xml_bboxes(xml_path)
            return base, (bboxes, orig_w, orig_h)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_single_xml, xf) for xf in xml_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="XML -> BBoxes"):
                base_name, annot = future.result()
                annotation_dict[base_name] = annot
        np.save(cache_file, {"annotation_dict": annotation_dict})
        print(f"Cached bounding boxes to '{cache_file}'.")
    total_bboxes = sum(len(v[0]) for v in annotation_dict.values())
    print(f"Total bounding boxes: {total_bboxes}")
    return annotation_dict

# -------------------------------
# Block 6: Convert XML Annotations to YOLO Labels with Letterbox Transformation
# -------------------------------
def generate_yolo_labels_for_split(split_folder, annotation_dict, target_size=TARGET_SIZE):
    """
    For each image in the images subfolder of split_folder, generate a YOLO label file.
    The transformation uses the original dimensions from XML and applies a letterbox resize transformation.
    YOLO label format: "class_id x_center y_center width height" (normalized to target_size).
    """
    images_dir = os.path.join(split_folder, "images")
    labels_dir = os.path.join(split_folder, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    target_w, target_h = target_size
    for img_path in tqdm(image_files, desc=f"Generating labels for {os.path.basename(split_folder)}", unit="file"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        if base not in annotation_dict:
            continue
        annotations, orig_w, orig_h = annotation_dict[base]
        if orig_w is None or orig_h is None:
            temp_img = cv2.imread(img_path)
            orig_h, orig_w = temp_img.shape[:2]
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        lines = []
        for (xmin, ymin, xmax, ymax, class_id) in annotations:
            x_center_orig = (xmin + xmax) / 2.0
            y_center_orig = (ymin + ymax) / 2.0
            box_w_orig = xmax - xmin
            box_h_orig = ymax - ymin
            x_center_new = x_center_orig * scale + pad_x
            y_center_new = y_center_orig * scale + pad_y
            box_w_new = box_w_orig * scale
            box_h_new = box_h_orig * scale
            x_center_norm = x_center_new / target_w
            y_center_norm = y_center_new / target_h
            box_w_norm = box_w_new / target_w
            box_h_norm = box_h_new / target_h
            lines.append(f"{class_id - 1} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}")
        if lines:
            label_file = os.path.join(labels_dir, base + ".txt")
            with open(label_file, "w") as f:
                f.write("\n".join(lines))

# -------------------------------
# Block 7: YOLO Model Creation using YOLO11 Train Mode
# -------------------------------
def create_yolo_model(num_classes, variant="x"):
    model_name = "./best.pt"
    model = YOLO(model_name)
    if hasattr(model.model, 'yaml'):
        model.model.yaml['nc'] = num_classes
    return model

# -------------------------------
# Block 8: YOLO Training and Evaluation (Using Ultralytics API)
# -------------------------------
def train_model_yolo(model, data_yaml, epochs=10, imgsz=224, device="cpu", resume=False):
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device, resume=resume)
    return results

def evaluate_model_yolo(model, data_yaml, device="cpu", split="val"):
    results = model.val(data=data_yaml, device=device, split=split, imgsz=224)
    return results

# -------------------------------
# Main Execution
# -------------------------------
def main():
    check_environment()
    # Uncomment the following if you need to process and split images and generate labels:
    # process_images()
    # split_dataset()
    # annotation_dict = build_annotation_dict(ANNOTATIONS_DIR)
    # for split in ['train', 'val', 'test']:
    #     split_folder = os.path.join(OUTPUT_SPLIT_DIR, split)
    #     generate_yolo_labels_for_split(split_folder, annotation_dict)
    
    # Update DATA_YAML path to point to your dataset YAML file (with absolute paths).
    DATA_YAML = "lego.yaml"  # Ensure this YAML file is correctly configured.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    model = create_yolo_model(NUM_CLASSES, variant="n")
    if os.path.exists(MODEL_PATH):
        print("Existing model checkpoint found. Loading from disk...")
        model = YOLO(MODEL_PATH)
        print("Model loaded.")
    else:
        print("No existing model found; training from scratch...")
        train_model_yolo(model, DATA_YAML, epochs=3, imgsz=224, device=device)
        model.save(MODEL_PATH)
    
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
            try:
                train_model_yolo(model, DATA_YAML, epochs=additional_epochs, imgsz=224, device=device, resume=True)
            except AssertionError as e:
                if "nothing to resume" in str(e):
                    print("Checkpoint indicates training is finished. Starting new training session without resume...")
                    train_model_yolo(model, DATA_YAML, epochs=additional_epochs, imgsz=224, device=device, resume=False)
                else:
                    raise
            model.save(MODEL_PATH)
        elif choice == '2':
            evaluate_model_yolo(model, DATA_YAML, device=device, split="val")
        elif choice == '3':
            evaluate_model_yolo(model, DATA_YAML, device=device, split="test")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
