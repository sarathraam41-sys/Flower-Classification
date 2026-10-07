\# Flower Classification Using an AlexNet-Inspired CNN



A deep learning image classification project that identifies five flower categories using a custom \*\*AlexNet-inspired Convolutional Neural Network (CNN)\*\* built with PyTorch.



\## 🌸 Supported Classes



The model classifies images into five categories:



\* Daisy

\* Dandelion

\* Rose

\* Sunflower

\* Tulip



\## 🚀 Project Overview



This project implements an AlexNet-inspired convolutional architecture designed specifically for flower image classification.



The model performs:



1\. Image preprocessing and augmentation

2\. Training/validation dataset splitting

3\. CNN-based feature extraction

4\. Multi-class classification

5\. Model checkpointing

6\. Classification evaluation

7\. Confusion matrix generation

8\. Training history visualization

9\. Single-image prediction



\## 🧠 Model Architecture



The project uses a custom `AlexNetInspired` model.



\### Feature Extractor



The convolutional backbone contains:



\* Conv2D: `3 → 96`

\* Batch Normalization

\* ReLU

\* Max Pooling

\* Conv2D: `96 → 256`

\* Batch Normalization

\* ReLU

\* Max Pooling

\* Conv2D: `256 → 384`

\* Batch Normalization

\* ReLU

\* Conv2D: `384 → 384`

\* Batch Normalization

\* ReLU

\* Conv2D: `384 → 256`

\* Batch Normalization

\* ReLU

\* Max Pooling

\* Adaptive Average Pooling



\### Classifier



The classification head contains:



\* Dropout

\* Fully connected layer: `256 × 6 × 6 → 4096`

\* ReLU

\* Dropout

\* Fully connected layer: `4096 → 4096`

\* ReLU

\* Output layer: `4096 → 5`



The convolutional and linear layers use explicit weight initialization.



\## 📂 Project Structure



```text

Flower-Classification/

│

├── models/

│   ├── \_\_init\_\_.py

│   └── alexnet.py

│

├── train.py

├── predict.py

├── test\_model.py

│

├── requirements.txt

├── README.md

└── .gitignore

```



The dataset and trained model files are intentionally excluded from GitHub through `.gitignore`.



\## 📊 Dataset



The project expects the dataset to follow this structure:



```text

dataset/

└── train/

&#x20;   ├── daisy/

&#x20;   ├── dandelion/

&#x20;   ├── rose/

&#x20;   ├── sunflower/

&#x20;   └── tulip/

```



The training script uses `torchvision.datasets.ImageFolder`, allowing the directory names to define the class labels automatically.



The dataset is split into:



\* \*\*80% training\*\*

\* \*\*20% validation\*\*



A fixed random seed is used for reproducibility.



\## 🔧 Training Configuration



The current training configuration includes:



| Parameter        |                          Value |

| ---------------- | -----------------------------: |

| Image Size       |                      224 × 224 |

| Batch Size       |                             32 |

| Epochs           |                            100 |

| Learning Rate    |                       3 × 10⁻⁴ |

| Weight Decay     |                       1 × 10⁻⁴ |

| Optimizer        |                          AdamW |

| Loss             |                  Cross Entropy |

| Label Smoothing  |                            0.1 |

| Scheduler        | Cosine Annealing Warm Restarts |

| Validation Split |                            20% |

| Random Seed      |                             42 |



\## 🖼️ Data Augmentation



Training images are augmented using techniques including:



\* Random resized crop

\* Horizontal flipping

\* Vertical flipping

\* Rotation

\* Random affine transformations

\* Color jitter

\* Random erasing

\* ImageNet normalization



Validation images use resizing and normalization without the training augmentations.



\## 💻 Installation



Clone the repository:



```bash

git clone https://github.com/sarathraam41-sys/Flower-Classification.git

cd Flower-Classification

```



Create and activate a virtual environment:



\### Windows



```cmd

python -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```cmd

pip install -r requirements.txt

```



\## 🧪 Test the Model Architecture



Before training, the architecture can be tested using:



```cmd

python test\_model.py

```



This performs a forward pass using a random `224 × 224` RGB image and verifies that the model produces five class outputs.



Expected output shape:



```text

torch.Size(\[1, 5])

```



\## 🏋️ Training



Place the dataset in the required directory structure and run:



```cmd

python train.py

```



The training script automatically:



\* Loads the dataset

\* Creates the training/validation split

\* Applies data augmentation

\* Initializes the AlexNet-inspired model

\* Trains the network

\* Tracks training and validation performance

\* Saves the best-performing model

\* Generates evaluation results



The trained model is saved as:



```text

best\_model.pth

```



The trained model is intentionally excluded from Git because model checkpoints can be large.



\## 🔮 Prediction



After training, a single image can be classified using:



```cmd

python predict.py

```



The prediction pipeline:



1\. Loads `best\_model.pth`

2\. Preprocesses the input image

3\. Runs the image through the CNN

4\. Determines the predicted flower class

5\. Reports the prediction confidence



\## 📈 Evaluation



The training pipeline generates evaluation information including:



\* Training loss

\* Validation loss

\* Training accuracy

\* Validation accuracy

\* Classification report

\* Confusion matrix

\* Training history plots



These outputs can be used to analyze model convergence and class-wise performance.



\## ⚡ Hardware Acceleration



The implementation automatically detects CUDA availability.



If a compatible NVIDIA GPU is available, PyTorch uses CUDA for training and inference.



Otherwise, the project falls back to CPU execution.



\## 🛠️ Technologies



\* Python

\* PyTorch

\* Torchvision

\* NumPy

\* Scikit-learn

\* Matplotlib

\* Seaborn

\* Pillow



\## 🎯 Learning Objectives



This project demonstrates practical implementation of:



\* Convolutional Neural Networks

\* AlexNet-inspired architectures

\* Image classification

\* Transfer-learning-inspired architectural design

\* Data augmentation

\* Regularization

\* Label smoothing

\* AdamW optimization

\* Learning-rate scheduling

\* Model checkpointing

\* Classification evaluation

\* Confusion matrix analysis

\* GPU-accelerated deep learning



\## 👨‍💻 Author



\*\*Sarath B Raam\*\*



Artificial Intelligence and Machine Learning

SRM Institute of Science and Technology



GitHub:

https://github.com/sarathraam41-sys



