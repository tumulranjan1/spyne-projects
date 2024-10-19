import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define file paths
wall_img_path = './data/wall.png'
floor_img_path = './data/floor.png'
car_masks_folder = './data/car_masks'
car_images_folder = './data/images'
shadow_masks_folder = './data/shadow_masks'
output_folder = './output'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load wall and floor images
wall_img = Image.open(wall_img_path).convert('RGBA')
floor_img = Image.open(floor_img_path).convert('RGBA')

# Function to remove transparent pixels
def remove_transparent_pixels(image):
    img_array = np.array(image)
    non_transparent_mask = img_array[:, :, 3] != 0
    non_transparent_img = img_array[non_transparent_mask.any(axis=1)]
    return Image.fromarray(non_transparent_img)

# Clean the wall and floor images
wall_img_cleaned = remove_transparent_pixels(wall_img)
floor_img_cleaned = remove_transparent_pixels(floor_img)

# Crop the top 25% of the wall image
crop_height = int(wall_img_cleaned.height * 0.25)
cropped_wall_img = wall_img_cleaned.crop((0, crop_height, wall_img_cleaned.width, wall_img_cleaned.height))

# Combine the wall and floor
combined_height = cropped_wall_img.height + floor_img_cleaned.height
combined_img = Image.new('RGBA', (cropped_wall_img.width, combined_height))
combined_img.paste(cropped_wall_img, (0, 0))
combined_img.paste(floor_img_cleaned, (0, cropped_wall_img.height))

# Resize the combined image
new_width = combined_img.width // 2
new_height = combined_img.height // 2
resized_combined_img = combined_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
resized_combined_img.save("combined_background_half_size.png")

# Load the combined background image
background_img = Image.open("combined_background_half_size.png")
background_width, background_height = background_img.size
car_height_percentage = 0.8  # Car height as percentage of background
floor_start_y = int(background_height * 1.05)  # Floor start position

# Define a function to pad and shift shadow mask to match car mask
def pad_and_shift_shadow_mask(car_mask, shadow_mask, shift_down=0):
    car_mask_height, car_mask_width = car_mask.shape
    shadow_mask_height, shadow_mask_width = shadow_mask.shape

    # Calculate the necessary padding to match car mask's dimensions
    pad_top = (car_mask_height - shadow_mask_height) // 2
    pad_bottom = car_mask_height - shadow_mask_height - pad_top
    pad_left = (car_mask_width - shadow_mask_width) // 2
    pad_right = car_mask_width - shadow_mask_width - pad_left

    # Apply the padding
    padded_shadow_mask = cv2.copyMakeBorder(
        shadow_mask,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Padding with black pixels
    )

    # Shift the mask downward by applying an affine transformation
    if shift_down > 0:
        M = np.float32([[1, 0, 0], [0, 1, shift_down]])  # Shift vertically by shift_down pixels
        padded_shadow_mask = cv2.warpAffine(padded_shadow_mask, M, (car_mask_width, car_mask_height))

    return padded_shadow_mask


# Parameter to control manual top padding for the shadow mask
extra_top_padding = 50  # Shift the shadow mask down by this value


# Parameter for shadow offset during placement
shadow_offset_y = 30  # Shift the shadow further down by 30 pixels during placement


# Loop through all car images and masks
for car_image_filename in os.listdir(car_images_folder):
    if car_image_filename.endswith((".jpeg", ".jpg", ".png")):
        car_image_path = os.path.join(car_images_folder, car_image_filename)
        car_mask_path = os.path.join(car_masks_folder, car_image_filename.replace(".jpeg", ".png").replace(".jpg", ".png"))
        shadow_mask_path = os.path.join(shadow_masks_folder, car_image_filename.replace(".jpeg", ".png").replace(".jpg", ".png"))
        
        if os.path.exists(car_mask_path) and os.path.exists(shadow_mask_path):
            # Load car image, mask, and shadow mask
            car_img = Image.open(car_image_path).convert("RGBA")
            car_mask_img = Image.open(car_mask_path).convert("L")
            shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Step 1: Denoise the car mask
            denoised_car_mask_img = car_mask_img.filter(ImageFilter.MedianFilter(size=5))
            
            # Step 2: Extract car using mask
            car_img_np = np.array(car_img)
            car_mask_np = np.array(denoised_car_mask_img)
            car_img_np[car_mask_np == 0] = [0, 0, 0, 0]  # Transparent background
            extracted_car_img = Image.fromarray(car_img_np)
            
            # Step 3: Resize the car
            car_width, car_height = extracted_car_img.size
            new_car_height = int(background_height * car_height_percentage)
            scaling_factor = new_car_height / car_height
            new_car_width = int(car_width * scaling_factor)
            resized_car_img = extracted_car_img.resize((new_car_width, new_car_height), Image.Resampling.LANCZOS)
            
            # Step 4: Pad and shift shadow mask to match car size (with extra top padding and shift)
            car_mask = np.array(denoised_car_mask_img)
            padded_shadow_mask = pad_and_shift_shadow_mask(car_mask, shadow_mask, shift_down=extra_top_padding)
            
            # Step 5: Resize the shadow mask to match resized car
            resized_shadow_mask = cv2.resize(padded_shadow_mask, (new_car_width, new_car_height))
            
            # Step 6: Extract the shadow from the car image using the shadow mask
            car_img_np_resized = np.array(resized_car_img)
            shadow_img_np = np.zeros_like(car_img_np_resized)
            shadow_img_np[..., 3] = resized_shadow_mask
            
            # Create shadow image by extracting shadow pixels from car
            shadow_img_np[..., :3] = car_img_np_resized[..., :3] * (resized_shadow_mask[..., None] / 255)
            shadow_img = Image.fromarray(shadow_img_np, 'RGBA')
            
            # Step 7: Create final image by pasting the car and shadow
            final_img = background_img.copy()
            car_position_x = (background_width - new_car_width) // 2
            car_position_y = floor_start_y - new_car_height
            
            # Adjust the shadow to be beneath the car with an extra offset during placement
            shadow_position_y = car_position_y + int(new_car_height * 0.1) + shadow_offset_y
            
            # Paste shadow first
            final_img.paste(shadow_img, (car_position_x, shadow_position_y), shadow_img)
            # Paste car
            final_img.paste(resized_car_img, (car_position_x, car_position_y), resized_car_img)
            
            # Step 8: Save final image
            output_path_png = os.path.join(output_folder, f"final_output_{car_image_filename.replace('.jpeg', '.png')}")
            final_img.save(output_path_png, format="PNG")

            print(f"Processed and saved: {output_path_png}")