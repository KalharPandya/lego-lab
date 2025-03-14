import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

def select_file():
    """Open a file dialog and return the selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    # Assume header is provided in the CSV file
    df = pd.read_csv(file_path)
    return df

def plot_data(df):
    """Plot several metrics vs. epoch."""
    # Convert epoch to numeric if needed
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot training losses: box_loss, cls_loss, dfl_loss
    axes[0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
    axes[0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', marker='o')
    axes[0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='o')
    axes[0].set_title("Training Losses")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot evaluation metrics: precision, recall, mAP50, mAP50-95
    axes[1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
    axes[1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='o')
    axes[1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='o')
    axes[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='o')
    axes[1].set_title("Evaluation Metrics")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot validation losses and learning rates
    axes[2].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', marker='o')
    axes[2].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', marker='o')
    axes[2].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', marker='o')
    axes[2].plot(df['epoch'], df['lr/pg0'], label='LR', marker='o')
    axes[2].set_title("Validation Losses and Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Value")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        return
    print(f"Loading data from: {file_path}")
    df = load_data(file_path)
    print("Data loaded successfully. Here are the first few rows:")
    print(df.head())
    
    plot_data(df)

if __name__ == "__main__":
    main()
