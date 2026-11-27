# NumPy Image Processing

## Project Overview
A practical project exploring image processing fundamentals using Python, NumPy, and Pillow without OpenCV.

## Topics Covered
- Image loading using Pillow
- NumPy arrays and image properties
- Brightness adjustment using astype() and clip()
- RGB to Grayscale conversion
- Horizontal and Vertical Flipping using slicing
- Image Cropping, Views, and Copies
- Reshaping and Batch Dimensions
- Boolean Masking

## Image Filters from Scratch (Notebook)
`Image_Filters.ipynb` builds convolution filters by hand: a 3x3 kernel slides over the image and each pixel becomes the sum of the kernel times its 3x3 neighborhood. Every result is shown next to the original.

| Filter | Kernel | What it does |
| --- | --- | --- |
| Blur (box filter) | `np.ones((3,3)) / 9` | Averages each 3x3 neighborhood and softens the image |
| Sharpen | `[[0,-1,0], [-1,5,-1], [0,-1,0]]` | Boosts the center pixel against its neighbors, so edges look crisper |
| Edge detection (Sobel) | `Kx` and `Ky` | Computes the gradient in x and y, then the magnitude `sqrt(Gx**2 + Gy**2)` |

The rule behind the kernels: a kernel whose values sum to 1 keeps the brightness of the image (blur, sharpen), and a kernel that sums to 0 cancels flat areas and keeps only the edges (Sobel, Prewitt, Laplacian).

Implementation notes:
- The loop covers interior pixels only (`range(1, h-1)`), so the 1-pixel border stays black because the output starts as `np.zeros`. This is expected.
- Sharpen output is clipped to 0-255 so it never saturates, and the Sobel magnitude is divided by its max to display in 0-1.
- Alternative edge kernels that work with the same loop: Prewitt `[[-1,0,1],[-1,0,1],[-1,0,1]]` and Laplacian `[[0,1,0],[1,-4,1],[0,1,0]]`.

## Technologies
- Python
- NumPy
- Pillow (PIL)
- Matplotlib (display)

## How to Run
Install dependencies:

```bash
pip install numpy pillow matplotlib
```

Run the script:

```bash
python Simple_Project_On_numpy.py
```

Run the filters notebook: put an image named `Photo.jpg` in the same folder as `Image_Filters.ipynb` (the repo's `image.jpg` works, just rename it or change the path in the first cell), then run the cells from top to bottom in Jupyter or VS Code.
