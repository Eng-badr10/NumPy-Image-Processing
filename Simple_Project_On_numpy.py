from PIL import Image
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


#1-Image Properties
image_path = Path(__file__).parent / "image.jpg"
img = Image.open(image_path).convert("RGB")
array = np.array(img)
print(f"Image_Shape is {array.shape}")
print(f"Image_Type is {array.dtype}")

#increase brightness
Brightness_array = array.astype(np.float32) + 10
Brightness_array = np.clip(Brightness_array, 0, 255).astype(np.uint8)
Image.fromarray(Brightness_array).save(OUTPUT_DIR / "brightness.png")


#Grayscale Conversion
gray_img = array.mean(axis=2).astype(np.uint8)
print(f"gray average shape is {gray_img.shape}")
Image.fromarray(gray_img).save(OUTPUT_DIR / "gray_mean.png")

#Grayscale Conversion with equations
Red = array[:, :, 0]
Green = array[:, :, 1]
Blue = array[:, :, 2]

gray_img_with_equation = 0.299 * Red + 0.587 * Green + 0.114 * Blue
print("gray image shape is", gray_img_with_equation.shape)
Image.fromarray(gray_img_with_equation.astype(np.uint8)).save(OUTPUT_DIR / "gray_equation.png")


#Flip the image horizontally and vertically
vertical_flip = array[::-1, :, :]
print(vertical_flip.shape)
Image.fromarray(vertical_flip).save(OUTPUT_DIR / "vertical_flip.png")

horizontal_flip = array[:, ::-1, :]
print(horizontal_flip.shape)
Image.fromarray(horizontal_flip).save(OUTPUT_DIR / "horizontal_flip.png")


#Image Cropping & View
crop = array[128:384, 127:381, :].copy()
print("Croping shape = ", crop.shape)
Image.fromarray(crop).save(OUTPUT_DIR / "crop.png")


#Reshape & Batch Dimension
## Convert HWC to CHW
image_reshape = array.transpose(2, 0, 1)
print("Image_Reshape(CHW) = ", image_reshape.shape)
## Add Batch Dimension
image_with_Batch = np.expand_dims(image_reshape, axis=0)
print("shape for image after adding batch ", image_with_Batch.shape)


#Masking
threshold = 80

gray = array.mean(axis=2)

mask = gray < threshold

masked_image = array.copy()
masked_image[mask] = [0, 0, 0]

Image.fromarray(masked_image).save(OUTPUT_DIR / "masked.png")
