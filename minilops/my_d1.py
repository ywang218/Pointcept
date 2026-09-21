import numpy as np

img = np.array([
    [[255, 0, 0],
     [0, 255, 0]],
    [[0, 0, 255],
     [255, 255, 0]] 
])

print(img.shape)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("error")

    #权重取人眼敏感度
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights 


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    