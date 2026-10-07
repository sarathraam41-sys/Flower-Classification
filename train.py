import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns

from models.alexnet import AlexNetInspired

#####################################################
# CONFIGURATION
#####################################################

DATASET_PATH = "dataset/train"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device :", DEVICE)

#####################################################
# TRAIN TRANSFORM
#####################################################

train_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8,1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(0.2),

    transforms.RandomRotation(30),

    transforms.RandomAffine(

        degrees=20,

        translate=(0.15,0.15),

        scale=(0.85,1.15)

    ),

    transforms.ColorJitter(

        brightness=0.3,

        contrast=0.3,

        saturation=0.3,

        hue=0.1

    ),

    transforms.ToTensor(),

    transforms.RandomErasing(

        p=0.25

    ),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

#####################################################
# VALIDATION TRANSFORM
#####################################################

val_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

#####################################################
# DATASET
#####################################################

full_dataset = datasets.ImageFolder(DATASET_PATH)

class_names = full_dataset.classes

print("\nClasses")

print(class_names)

train_size = int(0.8 * len(full_dataset))

val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(

    full_dataset,

    [train_size,val_size],

    generator=torch.Generator().manual_seed(42)

)

#####################################################
# APPLY TRANSFORMS
#####################################################

train_dataset.dataset.transform = train_transform

val_copy = copy.deepcopy(full_dataset)

val_copy.transform = val_transform

val_dataset.dataset = val_copy

#####################################################
# DATALOADERS
#####################################################

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=False

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False

)

#####################################################
# MODEL
#####################################################

model = AlexNetInspired(

    num_classes=len(class_names)

)

model.to(DEVICE)

#####################################################
# LOSS
#####################################################

criterion = nn.CrossEntropyLoss(

    label_smoothing=0.1

)

#####################################################
# OPTIMIZER
#####################################################

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

#####################################################
# SCHEDULER
#####################################################

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

    optimizer,

    T_0=10,

    T_mult=2

)

#####################################################
# TRAINING VARIABLES
#####################################################

best_model = copy.deepcopy(

    model.state_dict()

)

best_accuracy = 0

best_loss = 999

early_counter = 0

train_loss_history = []

val_loss_history = []

train_acc_history = []

val_acc_history = []

start_time = time.time()

#####################################################
# TRAINING LOOP
#####################################################

for epoch in range(EPOCHS):

    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"{'='*60}")

    ###############################################
    # TRAIN
    ###############################################

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)

        running_correct += (predicted == labels).sum().item()

        total += labels.size(0)

    train_loss = running_loss / len(train_dataset)

    train_accuracy = running_correct / total

    train_loss_history.append(train_loss)

    train_acc_history.append(train_accuracy)

    ###############################################
    # VALIDATION
    ###############################################

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            val_running_loss += loss.item() * images.size(0)

            val_correct += (predicted == labels).sum().item()

            val_total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_dataset)

    val_accuracy = val_correct / val_total

    val_loss_history.append(val_loss)

    val_acc_history.append(val_accuracy)

    ###############################################
    # Scheduler
    ###############################################

    scheduler.step()

    ###############################################
    # Print Results
    ###############################################

    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Train Accuracy  : {train_accuracy*100:.2f}%")

    print()

    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Validation Acc  : {val_accuracy*100:.2f}%")

    ###############################################
    # Save Best Model
    ###############################################

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_loss = val_loss

        best_model = copy.deepcopy(model.state_dict())

        torch.save(best_model, "best_model.pth")

        early_counter = 0

        print("\n✅ Best Model Saved")

    else:

        early_counter += 1

        print(f"Early Stop Counter : {early_counter}/{PATIENCE}")

    ###############################################
    # Early Stopping
    ###############################################

    if early_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        break

#####################################################
# LOAD BEST MODEL
#####################################################

model.load_state_dict(best_model)

#####################################################
# FINAL EVALUATION
#####################################################

print("\n" + "=" * 60)
print("Evaluating Best Model...")
print("=" * 60)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#####################################################
# CLASSIFICATION REPORT
#####################################################

print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4
))

#####################################################
# CONFUSION MATRIX
#####################################################

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.close()

#####################################################
# ACCURACY GRAPH
#####################################################

plt.figure(figsize=(10,5))

plt.plot(
    train_acc_history,
    label="Train Accuracy",
    linewidth=2
)

plt.plot(
    val_acc_history,
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("accuracy.png")

plt.close()

#####################################################
# LOSS GRAPH
#####################################################

plt.figure(figsize=(10,5))

plt.plot(
    train_loss_history,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    val_loss_history,
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("loss.png")

plt.close()

#####################################################
# FINAL RESULTS
#####################################################

training_time = (time.time() - start_time) / 60

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best Validation Accuracy : {best_accuracy * 100:.2f}%")

print(f"Best Validation Loss     : {best_loss:.4f}")

print(f"Training Time            : {training_time:.2f} minutes")

print("\nSaved Files")

print("✓ best_model.pth")
print("✓ accuracy.png")
print("✓ loss.png")
print("✓ confusion_matrix.png")

print("\nModel training completed successfully!")