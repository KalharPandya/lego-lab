import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# Global configuration
TARGET_SIZE = (224, 224)

# Define a function to build the ResNet18 architecture (with modified final layer)
def load_resnet_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Regression output (number of LEGO pieces)
    return model

# Load and prepare the quantized model (CPU only)
def load_quantized_model(model_path):
    # Build the base model architecture
    model = load_resnet_model()
    model = model.to("cpu")
    model.eval()
    # Apply dynamic quantization on Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()
    # Load the quantized state dictionary (using strict=False to allow quantization-specific keys)
    state_dict = torch.load(model_path, map_location="cpu")
    quantized_model.load_state_dict(state_dict, strict=False)
    return quantized_model

# Load the quantized model (ensure the model file exists)
MODEL_PATH = "./final_model_quantized.pth"
model = load_quantized_model(MODEL_PATH)

# Define image transforms (same as during training)
preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function that returns a nicely formatted text output
def predict_lego_count(images):
    """
    Accepts one or many images (PIL format) and returns a formatted string that shows the
    predicted LEGO piece count along with the raw output value and a confidence percentage.
    """
    if not isinstance(images, list):
        images = [images]
    
    results = []
    for img in images:
        input_tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            output = model(input_tensor)
        raw_value = output.item()
        rounded_value = int(round(raw_value))
        # Compute an arbitrary confidence value: higher if raw_value is very close to the rounded integer.
        confidence = max(0, 100 - abs(raw_value - rounded_value) * 100)
        results.append(
            f"Predicted LEGO pieces: {rounded_value} (raw: {raw_value:.4f}, confidence: {confidence:.1f}%)"
        )
    return "\n".join(results)

# Evaluation metrics markdown to showcase accuracy
evaluation_markdown = """
## Model Evaluation on Test Dataset

- **Number of Test Samples:** 33,560  
- **Average MSE Loss:** 0.0095  
- **Average MAE:** 0.0247  
- **Accuracy (Exact Match):** 99.32%  
- **F1 Score (Macro):** 0.6636  

**Classification Report:**  
```
              precision    recall  f1-score   support
           2       1.00      0.99      0.99     11979
           3       0.00      0.00      0.00         0
           4       1.00      1.00      1.00     21581
```

**Confusion Matrix:**  
```
[[11836   121    22]
 [    0     0     0]
 [   17    67 21497]]
```

*Note:* Class “3” is absent in this evaluation, leading to undefined recall for that class.
"""

# Gradio Interface using the new Interface class
iface = gr.Interface(
    fn=predict_lego_count,
    inputs=gr.Image(type="pil", label="Upload Images"),
    outputs=gr.Textbox(label="Predicted LEGO Piece Counts"),
    title="LEGO Piece Counter - Quantized Model Tester",
    description="Upload one or more images (JPG/PNG) to predict the number of LEGO pieces.",
    article=evaluation_markdown
)

iface.launch(server_port=80, server_name="0.0.0.0")
