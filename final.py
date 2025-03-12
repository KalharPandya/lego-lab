#!/usr/bin/env python3
"""
lego_training.py

This script performs the following tasks:
1. Environment Check – reports OpenCV, PyTorch, and CUDA status.
2. Image Processing – processes raw images (using mean shift filtering) and caches the results.
3. Dataset Splitting – splits the processed images into train, validation, and test sets.
4. XML Annotation Processing – parses XML files to get the number of LEGO pieces per image and caches results.
5. ResNet Training & Evaluation – uses a pretrained ResNet18 (with its final layer replaced) to train on the training data.
   If a trained model exists, it is loaded. The user can then choose to continue training or evaluate on validation or test sets.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

torch.cuda.empty_cache()

# -------------------------------
# Configuration & Directory Paths
# -------------------------------
RAW_IMAGES_DIR = r"lego_dataset\dataset_20210629145407_top_600\images"
PROCESSED_IMAGES_DIR = r"output"
ANNOTATIONS_DIR = r"lego_dataset\dataset_20210629145407_top_600\annotations"
OUTPUT_SPLIT_DIR = r"output_split"
XML_CACHE_FILE = "annotation_counts.npy"
MODEL_PATH = "./final_model_quantized.pth"
TARGET_SIZE = (224, 224)

import os
import kagglehub

# Configuration paths
RAW_IMAGES_DIR = r"lego_dataset\dataset_20210629145407_top_600\images"
PROCESSED_IMAGES_DIR = r"output"
ANNOTATIONS_DIR = r"lego_dataset\dataset_20210629145407_top_600\annotations"
OUTPUT_SPLIT_DIR = r"output_split"
XML_CACHE_FILE = "annotation_counts.npy"
MODEL_PATH = "./final_model_quantized.pth"
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
# Block 3: Dataset Splitting (Train/Val/Test)
# -------------------------------
def split_dataset(processed_dir=PROCESSED_IMAGES_DIR, split_dir=OUTPUT_SPLIT_DIR):
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")
    
    # Create directories if they do not exist
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Check if dataset is already split (e.g., train_dir has images)
    train_files = glob.glob(os.path.join(train_dir, "*.jpg"))
    if train_files:
        print(f"Dataset already split (found {len(train_files)} train images). Skipping splitting.\n")
        return
    else:
        print("Splitting dataset into train/val/test...")
        all_files = glob.glob(os.path.join(processed_dir, "*.jpg"))
        # First split into train+val and test (80/20)
        train_val_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        # Then split train_val into train and val (80/20 of train+val -> overall ~64/16/20)
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
        
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")
        
        def copy_files(file_list, target_dir):
            for file_path in tqdm(file_list, desc=f"Copying to {os.path.basename(target_dir)}", unit="file"):
                shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))
        
        copy_files(train_files, train_dir)
        copy_files(val_files, val_dir)
        copy_files(test_files, test_dir)
        print("Dataset split and files copied successfully.\n")

# -------------------------------
# Block 4: XML Annotation Processing and Caching
# -------------------------------
def process_xml_annotations(annotations_dir=ANNOTATIONS_DIR, cache_file=XML_CACHE_FILE):
    if os.path.exists(cache_file):
        print(f"Loading cached annotation counts from '{cache_file}'...")
        data = np.load(cache_file, allow_pickle=True).item()
        counts = data['counts']
    else:
        print("Processing XML annotation files...")
        xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
        counts = {}
        piece_counts = []
        
        def parse_xml_file(xml_file):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                object_count = len(root.findall("object"))
                filename = os.path.basename(xml_file)
                return filename, object_count
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(parse_xml_file, xml_file): xml_file for xml_file in xml_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing XML files"):
                result = future.result()
                if result is not None:
                    filename, count = result
                    counts[filename] = count
                    piece_counts.append(count)
        np.save(cache_file, {'counts': counts, 'piece_counts': piece_counts})
        print(f"Annotation counts cached to '{cache_file}'")
    total_objects = np.sum(list(counts.values()))
    print("\nTotal number of LEGO pieces in dataset:", total_objects, "\n")
    
    # Optional: Plot histogram of piece counts
    plt.figure(figsize=(8, 6))
    bins = np.arange(0, max(counts.values()) + 2) - 0.5
    plt.hist(list(counts.values()), bins=bins, edgecolor='black')
    plt.xlabel("Number of LEGO pieces per image")
    plt.ylabel("Frequency")
    plt.title("Histogram of LEGO Pieces per Image")
    plt.xticks(range(0, max(counts.values()) + 1))
    plt.show()
    
    return counts

# -------------------------------
# Custom Dataset for Training
# -------------------------------
class LegoDataset(Dataset):
    """
    Dataset for loading processed LEGO images and their corresponding annotation count.
    The annotation dictionary keys are converted to base filenames (without extension) to match the image files.
    """
    def __init__(self, image_dir, annotation_dict, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        
        # Build mapping from base name to file path for images
        self.image_dict = {}
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.image_dict[base] = img_path
        
        # Create a new annotation dict with base filename as key
        self.annotation_dict = {}
        for xml_name, count in annotation_dict.items():
            base = os.path.splitext(xml_name)[0]
            self.annotation_dict[base] = count
        
        # Collect samples that have a corresponding annotation
        self.samples = []
        for base, img_path in self.image_dict.items():
            if base in self.annotation_dict:
                self.samples.append((img_path, self.annotation_dict[base]))
            else:
                print(f"Warning: No annotation found for image {base}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Regression target as a float tensor (wrapped in a 1-element tensor)
        return image, torch.tensor([label], dtype=torch.float)

# -------------------------------
# Block 5: ResNet Training on Training Data
# -------------------------------
def train_resnet(train_dir, annotation_dict, num_epochs=10, batch_size=32, learning_rate=1e-3):
    """
    Train a ResNet18 model from scratch.
    """
    print("Starting training of ResNet model on training data...")
    
    # Define transforms (using ImageNet statistics for normalization)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and DataLoader for training
    train_dataset = LegoDataset(train_dir, annotation_dict, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load pretrained ResNet18 and modify final layer for regression (1 output)
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    
    criterion = nn.MSELoss()  # Using Mean Squared Error for regression
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
    
    resnet.train()
    print("Training in progress...")
    log_interval = 5  # Log loss every 5 batches

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {avg_loss:.4f}")
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
    print("Training complete. Saving the model checkpoint...")
    torch.save(resnet.state_dict(), MODEL_PATH)
    return resnet

def continue_training(model, train_dir, annotation_dict, num_epochs, batch_size, learning_rate, device):
    """
    Continue training an existing model.
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = LegoDataset(train_dir, annotation_dict, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    log_interval = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {avg_loss:.4f}")
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def evaluate_model(model, data_dir, annotation_dict, batch_size, device, max_samples=None):
    """
    Evaluate the model on a subset of the dataset (up to `max_samples`) and print:
     - MSE, MAE
     - % Error (rounded outputs)
     - Accuracy (rounded outputs)
     - F1 Score (macro)
     - Confusion Matrix
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load entire dataset
    full_dataset = LegoDataset(data_dir, annotation_dict, transform=data_transform)
    
    # If max_samples is provided, take only a subset of data
    if max_samples is not None and max_samples < len(full_dataset):
        indices = range(max_samples)  # first 'max_samples' samples
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_mae = 0.0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {data_dir}", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)  # shape: [batch_size, 1]
            
            outputs = model(inputs)     # shape: [batch_size, 1]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # MAE for this batch
            batch_mae = torch.mean(torch.abs(outputs - labels)).item()
            total_mae += batch_mae * inputs.size(0)
            
            # Round model outputs to nearest integer
            rounded_preds = torch.round(outputs).squeeze(1).long()  # shape: [batch_size]
            true_labels = labels.squeeze(1).long()                  # shape: [batch_size]
            
            # Collect for classification metrics
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    # Compute average MSE & MAE
    avg_loss = total_loss / len(dataset)
    avg_mae = total_mae / len(dataset)
    
    print(f"\nEvaluation on '{data_dir}' (Samples used: {len(dataset)}):")
    print(f"Average MSE Loss: {avg_loss:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    # ---- Percentage Error (like MAPE) ----
    # Avoid divide-by-zero; skip samples with true label == 0
    all_preds_np = np.array(all_preds, dtype=np.float32)
    all_labels_np = np.array(all_labels, dtype=np.float32)
    
    nonzero_mask = (all_labels_np != 0)
    if np.any(nonzero_mask):
        pct_errors = 100.0 * np.abs(all_preds_np[nonzero_mask] - all_labels_np[nonzero_mask]) / all_labels_np[nonzero_mask]
        avg_pct_error = np.mean(pct_errors)
    else:
        # If all labels are zero, measure absolute difference
        avg_pct_error = 100.0 * np.mean(np.abs(all_preds_np - all_labels_np))
    
    print(f"Average % Error (rounded): {avg_pct_error:.2f}%")
    
    # ---- Classification Metrics (Accuracy, F1, Confusion Matrix) ----
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"Accuracy (exact integer match): {acc:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Print Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

# -------------------------------
# Main Execution with Interactive Options
# -------------------------------
def main():
    check_environment()
    process_images()
    split_dataset()
    annotation_counts = process_xml_annotations()
    
    # Define directories for training, validation, and testing
    train_dir = os.path.join(OUTPUT_SPLIT_DIR, "train")
    val_dir = os.path.join(OUTPUT_SPLIT_DIR, "val")
    test_dir = os.path.join(OUTPUT_SPLIT_DIR, "test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or train model
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 1)
    resnet = resnet.to(device)
    
    if os.path.exists(MODEL_PATH):
        print("Trained model already exists. Loading model from checkpoint...")
        resnet.load_state_dict(torch.load(MODEL_PATH))
        resnet.eval()
    else:
        resnet = train_resnet(train_dir, annotation_counts, num_epochs=1, batch_size=256, learning_rate=1e-3)
    
    # Interactive loop for further training or evaluation

    while True:
        print("\nSelect an option:")
        print("1: Train further")
        print("2: Check validation performance")
        print("3: Check testing performance")
        print("4: Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            try:
                additional_epochs = int(input("Enter number of additional epochs to train: "))
            except ValueError:
                print("Invalid input. Please enter an integer.")
                continue
            resnet = continue_training(resnet, train_dir, annotation_counts, additional_epochs,
                                    batch_size=512, learning_rate=1e-4, device=device)

        elif choice == '2':
            # Ask how many validation samples to evaluate
            while True:
                user_in = input("Enter number of validation samples to evaluate (0 for all): ")
                try:
                    val_samples = int(user_in)
                    if val_samples < 0:
                        print("Number of samples must be >= 0. Please try again.")
                        continue
                    elif val_samples == 0:
                        val_samples = None  # Signal 'use entire set'
                except ValueError:
                    print("Invalid input. Please enter an integer.")
                    continue
                break
            
            evaluate_model(resnet, val_dir, annotation_counts,
                        batch_size=256, device=device, max_samples=val_samples)

        elif choice == '3':
            # Ask how many test samples to evaluate
            while True:
                user_in = input("Enter number of test samples to evaluate (0 for all): ")
                try:
                    test_samples = int(user_in)
                    if test_samples < 0:
                        print("Number of samples must be >= 0. Please try again.")
                        continue
                    elif test_samples == 0:
                        test_samples = None  # Signal 'use entire set'
                except ValueError:
                    print("Invalid input. Please enter an integer.")
                    continue
                break

            evaluate_model(resnet, test_dir, annotation_counts,
                        batch_size=256, device=device, max_samples=test_samples)

        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
