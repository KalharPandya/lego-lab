import torch
import torch.nn as nn
import torchvision.models as models

# Define the model architecture (ResNet18 with modified final layer)
def load_resnet_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Regression output
    return model

# Load the model architecture
model = load_resnet_model()

# Load the trained weights
model.load_state_dict(torch.load("./resnet_regressor.pth"))

# Set to evaluation mode
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # Quantize only Linear layers
    dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "final_model_quantized.pth")

print("Quantization complete. Model saved as final_model_quantized.pth")
