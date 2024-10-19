# Car Angle Classification API

This project provides an API for classifying the angle of a car in an image using a pre-trained deep learning model. The model was trained on a dataset of car images and is served using FastAPI. The project also includes steps to Dockerize the entire application for easy deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
  - [Setting up Kaggle API](#setting-up-kaggle-api)
  - [Downloading Dataset Using Kaggle API](#downloading-dataset-using-kaggle-api)
- [Model Training](#model-training)
- [Serving the Model with FastAPI](#serving-the-model-with-fastapi)
- [Running the Application with Docker](#running-the-application-with-docker)
- [Testing the API](#testing-the-api)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

The objective of this project is to classify the angle at which a car is photographed (e.g., front, rear-left, etc.) using a deep learning model. The model is trained using PyTorch and is deployed using FastAPI, allowing users to upload images and receive predictions.

**Key Features:**
- Model training and evaluation using PyTorch
- Serving the model using FastAPI
- Dockerizing the application for easy deployment
- Swagger documentation for testing the API

## Project Structure

```
car_angle_classification/
│
├── app/                         # API-related files
│   ├── app.py                   # FastAPI application
│   ├── Dockerfile               # Docker configuration for the FastAPI 
│   ├── requirements.txt         # Dependencies for FastAPI app
│   └── bestmodel.pt             # TorchScript model for inference
│
├── train/                       # Training-related files
│   ├── data.py                  # Data loading and transformation functions
│   ├── models.py                # Model architecture definitions and loading
│   ├── train.py                 # Model training and validation logic
│   └── charts/                  # Generated training/validation charts (optional)
│
├── dataset/                     # Dataset folder for images
│   └── images/                  # Placeholder for the car image dataset
│
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

## Dataset

The dataset used for this project consists of car images, each labeled with the angle at which the car is photographed (e.g., 0, 40, 90, etc.).

### Dataset Structure

To use torchvision.datasets.ImageFolder, the dataset should be structured as follows:

```
dataset/
├── 0/
│   └── image1.jpg
│   └── image2.jpg
│   └── ...
├── 40/
│   └── image1.jpg
│   └── image2.jpg
│   └── ...
├── 90/
│   └── ...
├── 130/
│   └── ...
├── 180/
│   └── ...
├── 230/
│   └── ...
├── 270/
│   └── ...
└── 320/
    └── ...
```

### Setting up Kaggle API

To download datasets from Kaggle, follow these steps:

1. Sign in to Kaggle: If you don't already have an account, sign up at Kaggle.

2. Create an API Token:
   - Go to your Kaggle account API page
   - Click "Create New API Token" to download `kaggle.json`

3. Place the kaggle.json file in the right location:
   ```bash
   mkdir ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json  # Restrict file permissions for security
   ```

4. Install the Kaggle API client:
   ```bash
   pip install kaggle
   ```

### Downloading Dataset Using Kaggle API

1. Find the Dataset on Kaggle
2. Download using the API:
   ```bash
   kaggle datasets download username/dataset-name
   ```
3. Unzip and organize:
   ```bash
   unzip dataset-name.zip -d dataset/
   ```

## Model Training

### Training Environment

The training process is implemented in PyTorch and located in the `train/` directory. Key files:
- `data.py`: Handles data loading and preprocessing
- `models.py`: Defines model architectures (ResNet, EfficientNet, etc.)
- `train.py`: Script for training and validating the model

### Training Steps

1. Install dependencies:
   ```bash
   pip install torch torchvision tqdm matplotlib scikit-learn
   ```

2. Prepare the dataset under the `dataset/` directory

3. Train the model:
   ```bash
   python train/train.py
   ```

## Serving the Model with FastAPI

### FastAPI Overview

The trained model is served using FastAPI, which provides a REST API for inference. The API accepts an image, processes it, and returns the predicted car angle along with a confidence score.

### Running FastAPI

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn torch torchvision pillow
   ```

2. Start the FastAPI app:
   ```bash
   uvicorn app:app --reload
   ```

3. Access the API at `http://127.0.0.1:8000` or documentation at `http://127.0.0.1:8000/docs`

## Running the Application with Docker

1. Build the Docker image:
   ```bash
   docker build -t car-angle-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 car-angle-api
   ```

3. Access the API at `http://localhost:8000/docs`

## Testing the API

### Example Usage with Curl

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/car_image.jpg'
```

Example Response:
```json
{
  "angle_class": "270",
  "confidence": 0.85
}
```

## API Endpoints

### `/predict/` (POST)
- Upload an image to get the predicted car angle and confidence score
- **Request:** multipart/form-data file (car image)
- **Response:** JSON object with predicted angle and confidence

Example Response:
```json
{
  "angle_class": "180",
  "confidence": 0.92
}
```

### `/docs` (GET)
- Access the auto-generated Swagger UI documentation
