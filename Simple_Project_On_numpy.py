from PIL import Image
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "outputs"


#1-Image Properties
def image_properties(array):
    print(f"Image_Shape is {array.shape}")
    print(f"Image_Type is {array.dtype}")


#increase brightness
def increase_brightness(array):
    Brightness_array = array.astype(np.float32) + 10
    Brightness_array = np.clip(Brightness_array, 0, 255).astype(np.uint8)
    Image.fromarray(Brightness_array).save(OUTPUT_DIR / "brightness.png")


#Grayscale Conversion
def grayscale_conversion(array):
    gray_img = array.mean(axis=2).astype(np.uint8)
    print(f"gray average shape is {gray_img.shape}")
    Image.fromarray(gray_img).save(OUTPUT_DIR / "gray_mean.png")


#Grayscale Conversion with equations
def grayscale_conversion_with_equations(array):
    Red = array[:, :, 0]
    Green = array[:, :, 1]
    Blue = array[:, :, 2]

    gray_img_with_equation = 0.299 * Red + 0.587 * Green + 0.114 * Blue
    print("gray image shape is", gray_img_with_equation.shape)
    Image.fromarray(gray_img_with_equation.astype(np.uint8)).save(OUTPUT_DIR / "gray_equation.png")


#Flip the image horizontally and vertically
def flip_image(array):
    vertical_flip = array[::-1, :, :]
    print(vertical_flip.shape)
    Image.fromarray(vertical_flip).save(OUTPUT_DIR / "vertical_flip.png")

    horizontal_flip = array[:, ::-1, :]
    print(horizontal_flip.shape)
    Image.fromarray(horizontal_flip).save(OUTPUT_DIR / "horizontal_flip.png")


#Image Cropping & View
def crop_image(array):
    crop = array[128:384, 127:381, :].copy()
    print("Croping shape = ", crop.shape)
    Image.fromarray(crop).save(OUTPUT_DIR / "crop.png")


#Reshape & Batch Dimension
def reshape_and_batch(array):
    ## Convert HWC to CHW
    image_reshape = array.transpose(2, 0, 1)
    print("Image_Reshape(CHW) = ", image_reshape.shape)
    ## Add Batch Dimension
    image_with_Batch = np.expand_dims(image_reshape, axis=0)
    print("shape for image after adding batch ", image_with_Batch.shape)


#Masking
def masking(array):
    threshold = 80

    gray = array.mean(axis=2)

    mask = gray < threshold

    masked_image = array.copy()
    masked_image[mask] = [0, 0, 0]

    Image.fromarray(masked_image).save(OUTPUT_DIR / "masked.png")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    image_path = Path(__file__).parent / "image.jpg"
    img = Image.open(image_path).convert("RGB")
    array = np.array(img)

    image_properties(array)
    increase_brightness(array)
    grayscale_conversion(array)
    grayscale_conversion_with_equations(array)
    flip_image(array)
    crop_image(array)
    reshape_and_batch(array)
    masking(array)


if __name__ == "__main__":
    main()
