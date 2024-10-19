
# Image Manipulation and Background Replacement

## Overview
This project manipulates car images by removing backgrounds, placing realistic shadows, and aligning the perspective with a new background. The key tasks accomplished include:

1. **Background Replacement**: Using background removal masks to cleanly extract car images and replace the background with a wall and floor image.
2. **Shadow Placement**: Utilizing shadow masks to place natural shadows underneath the car.
3. **Perspective Matching**: Ensuring the car and the floor align properly for a realistic composition.

## Prerequisites
Before running this project, ensure you have Python installed. You can check your Python version by running:

```bash
python --version
```

Ensure that your Python version is 3.6 or higher.

## Setup Instructions

### Step 1: Create and Activate Virtual Environment
To ensure that all dependencies are managed properly, it is recommended to use a virtual environment.

#### For Windows:
1. Open Command Prompt or PowerShell and navigate to your project folder.
2. Create a virtual environment:
   ```bash
   python -m venv env
   ```
3. Activate the virtual environment:
   ```bash
   env/Scripts/activate
   ```

#### For Mac/Linux:
1. Open the terminal and navigate to your project folder.
2. Create a virtual environment:
   ```bash
   python3 -m venv env
   ```
3. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```

### Step 2: Install Dependencies
After activating the virtual environment, install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, manually install the dependencies by running:

```bash
pip install numpy Pillow opencv-python matplotlib
```

### Step 3: Prepare Input Data
Ensure that you have the following directories and files in your project folder:

- `./data/wall.png`: The wall image.
- `./data/floor.png`: The floor image.
- `./data/car_masks/`: Folder containing masks to isolate the cars from their backgrounds.
- `./data/images/`: Folder containing the car images.
- `./data/shadow_masks/`: Folder containing masks for realistic shadow placement.

### Step 4: Run the Script
Once the environment is set up and the required images are in place, you can run the script using:

```bash
python project1.py
```

The processed car images with backgrounds and shadows will be saved in the `./output` folder.

## Output
The final images with the new background and shadows are saved as PNG files in the `./output` folder. Each file will be named in the format `final_output_<car_image_name>.png`.

## Notes
- You can modify the parameters for shadow placement and scaling factors within the script to adjust the final image composition as needed.
- Ensure that the shadow and car masks are appropriately named and match the car image file names.
