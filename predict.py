import torch
from torchvision import transforms
from PIL import Image

from models.alexnet import AlexNetInspired

# -----------------------------------------
# Configuration
# -----------------------------------------

MODEL_PATH = "best_model.pth"

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# Image Transform
# -----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# Load Model
# -----------------------------------------

model = AlexNetInspired(num_classes=5)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)

model.eval()

# -----------------------------------------
# Prediction Function
# -----------------------------------------

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    flower = CLASS_NAMES[predicted.item()]

    return flower, confidence.item()

# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    image_path = input("Enter image path : ")

    flower, confidence = predict(image_path)

    print("\nPrediction")
    print("---------------------------")
    print("Flower :", flower)
    print(f"Confidence : {confidence*100:.2f}%")