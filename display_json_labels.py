import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
from pycocotools import mask as mask_util

# Function to decode RLE masks
def decode_mask(mask_rle, shape=(256, 256)):
    return mask_util.decode(mask_rle).reshape(shape)

# Function to draw segmentation masks
def draw_segmentation(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay segmentation mask on the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

# Load image
image_path = r"C:\Users\Rasmu\Repos\InstanceDiffusion\dataset_stuff\GCI_Front_With_Duplications_100\Cam=F-FN=20230705.141711.192122_White-SVID=FullView-PID=-x=0000-y=0000-w=1088-h=1456.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Load JSON data
json_data = r"C:\Users\Rasmu\Repos\InstanceDiffusion-main\dataset-generation\train-data\label_Cam=F-FN=20230705.141711.192122_White-SVID=FullView-PID=-x=0000-y=0000-w=1088-h=1456.json"
with open(json_data, 'r') as file:
    data = json.load(file)

# Iterate over the annotations to draw bounding boxes and segmentation masks
for anno in data['annos']:
    bbox = anno['bbox']
    mask_rle = anno['mask']
    text_embedding_before = anno.get('text_embedding_before')
    blip_embeddings = anno.get('captions')
    # Decode mask
    mask = decode_mask(mask_rle, shape=(image.shape[0], image.shape[1]))
    
    # Draw segmentation mask
    image = draw_segmentation(image, mask)
    # Draw bounding box
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    
    # Print text embeddings
    print(f"Category name: '{anno.get('category_name', 'Unknown')}', Caption:'{anno.get('caption', 'Unknown')}")

# Display the image with masks and bounding boxes
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.show()