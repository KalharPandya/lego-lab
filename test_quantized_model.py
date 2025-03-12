#!/usr/bin/env python3
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

# -------------------------------
# Configuration & Paths
# -------------------------------
OUTPUT_SPLIT_DIR = r"output_split"
TEST_DIR = os.path.join(OUTPUT_SPLIT_DIR, "test")
XML_CACHE_FILE = "annotation_counts.npy"
MODEL_PATH = "./final_model_quantized.pth"
TARGET_SIZE = (224, 224)

# -------------------------------
# Dataset Definition
# -------------------------------
class LegoDataset(torch.utils.data.Dataset):
    """
    Dataset for loading processed LEGO images and their corresponding annotation count.
    Keys are based on the image filename (without extension).
    """
    def __init__(self, image_dir, annotation_dict, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        
        # Build mapping from base name to file path
        self.image_dict = {}
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.image_dict[base] = img_path
        
        # Create a new annotation dict with base filename as key
        self.annotation_dict = {}
        for xml_name, count in annotation_dict.items():
            base = os.path.splitext(xml_name)[0]
            self.annotation_dict[base] = count
        
        # Collect samples that have corresponding annotations
        self.samples = []
        for base, img_path in self.image_dict.items():
            if base in self.annotation_dict:
                self.samples.append((img_path, self.annotation_dict[base]))
            else:
                print(f"Warning: No annotation for image {base}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float)

# -------------------------------
# Helper Functions
# -------------------------------
def load_annotation_counts():
    if os.path.exists(XML_CACHE_FILE):
        data = np.load(XML_CACHE_FILE, allow_pickle=True).item()
        return data['counts']
    else:
        print("Annotation cache not found.")
        return {}

def load_quantized_model(device):
    """
    Rebuild the ResNet18 architecture (with modified final layer), apply dynamic quantization,
    and load the quantized state dictionary.
    """
    # Recreate the model architecture
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    
    # Apply dynamic quantization (quantize Linear layers)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()
    
    # Load the quantized weights. Using strict=False allows for quantization-specific keys.
    state_dict = torch.load(MODEL_PATH)
    quantized_model.load_state_dict(state_dict, strict=False)
    return quantized_model

def evaluate_model(model, dataset, batch_size, device):
    """
    Evaluate the model on the given dataset and compute:
     - Average MSE and MAE loss
     - Accuracy (after rounding predictions)
     - F1 Score (macro)
     - Classification Report and Confusion Matrix
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            mae = torch.mean(torch.abs(outputs - labels)).item()
            total_mae += mae * inputs.size(0)
            
            # Round predictions to nearest integer
            rounded_preds = torch.round(outputs).squeeze(1).long()
            true_labels = labels.squeeze(1).long()
            
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataset)
    avg_mae = total_mae / len(dataset)
    
    print(f"\nEvaluation on Test Dataset (N = {len(dataset)} samples):")
    print(f"Average MSE Loss: {avg_loss:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"Accuracy (exact match): {acc:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# -------------------------------
# Main Execution
# -------------------------------
def main():
    device = torch.device("cpu")
    annotation_counts = load_annotation_counts()
    
    # Define data transforms (same as training)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(TARGET_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Build the test dataset
    test_dataset = LegoDataset(TEST_DIR, annotation_counts, transform=data_transform)
    
    # Load the quantized model
    quantized_model = load_quantized_model(device)
    print("Quantized model loaded. Starting evaluation on test dataset...")
    
    # Evaluate the quantized model
    evaluate_model(quantized_model, test_dataset, batch_size=256, device=device)

if __name__ == "__main__":
    main()
