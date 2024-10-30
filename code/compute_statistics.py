import os
import glob
import numpy as np
import rasterio
 
IMAGE_PATH = '../dataset/masks/patches/'
IMAGE_PATTERN = '*.tif'

def get_img_arr(path):
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
    img = np.float32(img)  
    return img

def compute_statistics(image_files):
    # Initialize for single channel
    sum_channel = 0
    sum_squares_channel = 0
    total_pixels = 0
    
    for idx, img_file in enumerate(image_files):
        img = get_img_arr(img_file)
        h, w, c = img.shape
        total_pixels += h * w
        
        # Reshape to flatten the image, but don't try to split into 10 channels
        flattened_img = img.reshape(-1)
        sum_channel += np.sum(flattened_img)
        sum_squares_channel += np.sum(flattened_img**2)
        
        if idx % 100 == 0:
            print('Procesadas {} imágenes'.format(idx))
    
    # Calculate statistics for single channel
    mean_channel = sum_channel / total_pixels
    std_channel = np.sqrt(sum_squares_channel / total_pixels - mean_channel**2)
    
    print('Canal 1: media = {:.2f}, desviación estándar = {:.2f}'.format(
        mean_channel, std_channel))


if __name__ == '__main__':
    image_files = glob.glob(os.path.join(IMAGE_PATH, IMAGE_PATTERN))
    num_images = len(image_files)
    print('Número de imágenes:', num_images)
    compute_statistics(image_files)