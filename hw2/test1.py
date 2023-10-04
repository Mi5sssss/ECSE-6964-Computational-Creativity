import lpips
import numpy as np
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error, structural_similarity
from skimage.transform import rotate, resize
from skimage.util import random_noise
from PIL import Image
import os
import cv2

def compute_metrics(imgA, imgB):
    imgA, imgB = np.array(imgA), np.array(imgB)

    imgA = (imgA * 255).astype("uint8")
    imgB = (imgB * 255).astype("uint8")

    # print(imgA.shape, imgB.shape)
    
    imgA_resized = resize(imgA, (64, 64))
    imgB_resized = resize(imgB, (64, 64))

    lpips_value = loss_fn.forward(lpips.im2tensor(imgA_resized), lpips.im2tensor(imgB_resized))
    mse_value = mean_squared_error(imgA, imgB)

    histA = cv2.calcHist([imgA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imgB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    histA = cv2.normalize(histA, histA).astype('float32')
    histB = cv2.normalize(histB, histB).astype('float32')
    
    ssim_value = cv2.compareHist(histA, histB, method=cv2.HISTCMP_INTERSECT)

    return lpips_value, mse_value, ssim_value


image_folder = "/home/xie/OpenCamera/hw2"
image_files = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]  # or other extensions

images = [img_as_float(Image.open(f)) for f in image_files]

noisy = random_noise(images[0], mode='gaussian', mean=0, var=0.01)
rotated = rotate(images[0], angle=45)
cropped = images[0][50:250, 50:250]

loss_fn = lpips.LPIPS(net='alex')


output_file = 'results.txt'

with open(output_file, 'w') as f:
    f.write('Image Comparison Results\n')
    f.write('=======================\n\n')

for i in range(len(images)):
    for j in range(i+1, len(images)):
        lpips_val, mse_val, ssim_val = compute_metrics(images[i], images[j])
        
        result_str = f"""
Metrics between {image_files[i]} and {image_files[j]}:
LPIPS: {lpips_val}
MSE: {mse_val}
SSIM: {ssim_val}
---------------------------
"""
        print(result_str)

        with open(output_file, 'a') as f:
            f.write(result_str)

print(f"Results saved to {output_file}")
