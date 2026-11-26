from PIL import Image
import numpy as np
from pathlib import Path


#1-Image Properties
image_path = Path(__file__).parent / "image.jpg"
img = Image.open(image_path).convert("RGB")
array = np.array(img)
print(f"Image_Shape is {array.shape}")
print(f"Image_Type is {array.dtype}")

#increase brightness
Brightness_array = array.astype(np.float32) + 10
Brightness_array=np.clip(Brightness_array,0,255).astype(np.uint8)
# Image.fromarray(Brightness_array).show()


#Grayscale Conversion
gray_img= array.mean(axis=2)
print(f"gray average shape is {gray_img.shape}")
# Image.fromarray(gray_img).show()

#Grayscale Conversion with equations
Red = array[:,:,0]
Green = array[:,:,1]
Blue = array[:,:,2]

gray_img_with_equation =  0.299 * Red + 0.587 * Green +0.114 * Blue
print("gray image shape is",gray_img_with_equation.shape)
# Image.fromarray(gray_img).show()


#Flip the image horizontally and vertically
image_Horizontal = array[::-1,:,:]
print(image_Horizontal.shape)
# Image.fromarray(image_Horizontal).show()

image_vertical = array[:,::-1,:]
print(image_vertical.shape)
# Image.fromarray(image_vertical).show()


#Image Cropping & View
crop = array[128:384, 127:381, :].copy()
print("Croping shape = ",crop.shape)
# Image.fromarray(crop).show()



#Reshape & Batch Dimension
## Convert HWC to CHW
image_reshape = array.transpose(2,0,1)
print("Image_Reshape(CHW) = ",image_reshape.shape)
## Add Batch Dimension
image_with_Batch = np.expand_dims(image_reshape , axis =0)
print("shape for image after adding batch " , image_with_Batch.shape)


#Masking
threshold = 80

gray = array.mean(axis=2)

mask = gray < threshold

masked_image = array.copy()
masked_image[mask] = [0, 0, 0]

Image.fromarray(masked_image).show()