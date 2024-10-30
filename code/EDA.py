import os
import glob
import numpy as np
import rasterio

def get_img_arr(path):
    """
    Lee y verifica la imagen, proporcionando información detallada sobre sus dimensiones
    """
    with rasterio.open(path) as src:
        # Obtener información de la imagen
        print(f"\nInformación de la imagen: {os.path.basename(path)}")
        print(f"Número de bandas: {src.count}")
        print(f"Dimensiones (alto x ancho): {src.height} x {src.width}")
        
        # Leer la imagen
        img = src.read()
        print(f"Forma del array inicial: {img.shape}")
        
        # Transponer solo si es necesario
        if img.shape[0] != 10:
            print("⚠️ Advertencia: El número de bandas no es 10")
            print(f"Bandas detectadas: {img.shape[0]}")
            return None
        
        # Transponer para tener el formato (height, width, channels)
        img = img.transpose((1, 2, 0))
        print(f"Forma después de transponer: {img.shape}")
        
        # Verificar que los datos son válidos
        print(f"Tipo de datos: {img.dtype}")
        print(f"Rango de valores: [{np.min(img)}, {np.max(img)}]")
        print(f"¿Contiene NaN?: {np.any(np.isnan(img))}")
        print(f"¿Contiene infinitos?: {np.any(np.isinf(img))}")
        
    return np.float32(img)

def compute_statistics(image_files):
    """
    Calcula estadísticas para las imágenes, con mejor manejo de dimensiones
    """
    # Inicializar arrays para almacenar las sumas
    sum_channels = None
    sum_squares_channels = None
    total_pixels = 0
    processed_images = 0

    for idx, img_file in enumerate(image_files):
        try:
            print(f"\nProcesando imagen {idx + 1}/{len(image_files)}: {os.path.basename(img_file)}")
            
            # Leer y verificar la imagen
            img = get_img_arr(img_file)
            if img is None:
                print(f"Saltando imagen {os.path.basename(img_file)} debido a número incorrecto de bandas")
                continue
                
            # Obtener dimensiones
            h, w, c = img.shape
            print(f"Dimensiones de la imagen: {h} x {w} x {c}")
            
            # Verificar que el número de canales es correcto
            if c != 10:
                print(f"Error: Número incorrecto de canales ({c})")
                continue
            
            # Reshape con verificación
            try:
                img_reshaped = img.reshape(-1, c)
                print(f"Forma después de reshape: {img_reshaped.shape}")
            except ValueError as e:
                print(f"Error al hacer reshape: {e}")
                print(f"Tamaño del array: {img.size}")
                continue
            
            # Inicializar arrays en la primera imagen válida
            if sum_channels is None:
                sum_channels = np.zeros(c, dtype=np.float64)
                sum_squares_channels = np.zeros(c, dtype=np.float64)
            
            # Calcular estadísticas
            valid_mask = ~np.isnan(img_reshaped) & ~np.isinf(img_reshaped)
            valid_pixels = valid_mask.all(axis=1)
            
            if np.any(valid_pixels):
                sum_channels += np.sum(img_reshaped[valid_pixels], axis=0)
                sum_squares_channels += np.sum(np.square(img_reshaped[valid_pixels]), axis=0)
                n_valid = np.sum(valid_pixels)
                total_pixels += n_valid
                processed_images += 1
                print(f"Píxeles válidos en esta imagen: {n_valid}")
            else:
                print("⚠️ No se encontraron píxeles válidos en esta imagen")
            
        except Exception as e:
            print(f"Error procesando imagen: {str(e)}")
            continue

    # Calcular estadísticas finales
    print("\n=== Resumen Final ===")
    print(f"Imágenes procesadas exitosamente: {processed_images}/{len(image_files)}")
    print(f"Total de píxeles válidos: {total_pixels}")
    
    if total_pixels > 0:
        mean_channels = sum_channels / total_pixels
        std_channels = np.sqrt((sum_squares_channels / total_pixels) - np.square(mean_channels))
        
        print("\nEstadísticas por banda:")
        for i in range(len(mean_channels)):
            print(f'Banda {i+1}: media = {mean_channels[i]:.2f}, '
                  f'desviación estándar = {std_channels[i]:.2f}')
    else:
        print("No se pudieron calcular estadísticas: no hay píxeles válidos")




if __name__ == '__main__':
    IMAGE_PATH = '../images/landsat_images/tiff_images/'
    image_files = glob.glob(os.path.join(IMAGE_PATH, '*.TIF'))  # Nota: usando .TIF en mayúsculas
    
    print(f'Número de imágenes encontradas: {len(image_files)}')
    if len(image_files) == 0:
        print("No se encontraron imágenes. Verifica la ruta y la extensión.")
    else:
        compute_statistics(image_files)