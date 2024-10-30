import os
import glob


IMAGE_PATH = '../dataset/masks/patches/'
#IMAGE_PATH = '../dataset/images/landsat_images/tiff_images/'
IMAGE_PATTERN = '*.tif'


def count_images(image_path, image_pattern):
    image_files = glob.glob(os.path.join(image_path, image_pattern))
    num_images = len(image_files)
    print('Número de imágenes:', num_images)
    return num_images


if __name__ == '__main__':
    num_images = count_images(IMAGE_PATH, IMAGE_PATTERN)