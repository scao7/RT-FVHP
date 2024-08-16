import os
import cv2
import numpy as np
from glob import glob

subject = "sample3"

# Paths
images_folder = f'/data/Shengting/resub_paperII/RTFVHP/instant-nvr/data_exp/{subject}/images/{subject}'
smpl_masks_folder = f'/data/Shengting/resub_paperII/RTFVHP/instant-nvr/data_exp/{subject}/smpl_masks/{subject}'
masks_folder = f'/data/Shengting/resub_paperII/RTFVHP/instant-nvr/data_exp/{subject}/schp/{subject}'
output_folder = f'/data/Shengting/resub_paperII/RTFVHP/instant-nvr/data_picked_2/{subject}'

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'images', subject), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'masks', subject), exist_ok=True)

# Get the image, SMPL mask, and mask paths
image_paths = sorted(glob(os.path.join(images_folder, '*.jpg')))
smpl_mask_paths = sorted(glob(os.path.join(smpl_masks_folder, '*.png')))
mask_paths = sorted(glob(os.path.join(masks_folder, '*.png')))

# Function to calculate overlap between two masks
def calculate_overlap(mask1, mask2):
    return np.sum(mask1 & mask2)

# Function to calculate mismatch between two masks
def calculate_mismatch(mask1, mask2):
    return np.sum(np.bitwise_xor(mask1, mask2))

# Function to keep the largest connected component in a mask
def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background label 0
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    return largest_mask

# Loop through the images and masks
scores = []
for img_path, smpl_mask_path, mask_path in zip(image_paths, smpl_mask_paths, mask_paths):
    smpl_mask = cv2.imread(smpl_mask_path, cv2.IMREAD_GRAYSCALE) // 255
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert all non-zero values in mask to 255
    mask[mask != 0] = 255
    mask = mask // 255
    
    # Keep only the largest connected component in both masks
    largest_smpl_mask = keep_largest_component(smpl_mask)
    largest_mask = keep_largest_component(mask)
    
    # Calculate overlap and mismatch
    overlap = calculate_overlap(largest_smpl_mask, largest_mask)
    mismatch = calculate_mismatch(largest_smpl_mask, largest_mask)
    
    # Combine scores (higher overlap and lower mismatch is better)
    score = overlap 
    scores.append((score, img_path, largest_smpl_mask, largest_mask))

# Sort by combined score (descending, as higher score is better)
scores.sort(reverse=True, key=lambda x: x[0])
top_100 = scores[:100]

# Process the top 100
for i, (_, img_path, largest_smpl_mask, largest_mask) in enumerate(top_100):
    img = cv2.imread(img_path)
    
    # Combine the largest components of the masks
    combined_mask = cv2.bitwise_or(largest_smpl_mask, largest_mask)

    # Save the image and combined mask
    cv2.imwrite(os.path.join(output_folder, 'images', subject, f'{i:06}.jpg'), img)
    cv2.imwrite(os.path.join(output_folder, 'masks', subject, f'{i:06}.png'), combined_mask)

print("Processing complete!")
