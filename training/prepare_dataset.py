
import os
import wget
import zipfile
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm


def download_totalsegmentator_subset():
    """
    Загрузка подмножества данных TotalSegmentator
    """
    print("Загрузка примеров из TotalSegmentator dataset...")
    
    # Создание директорий
    data_dir = Path("data/training")
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Здесь должны быть ссылки на реальные данные
    # Для демонстрации создаем синтетические данные
    print("Создание демонстрационного датасета...")
    
    for i in range(20):
        # Создание синтетического CT изображения
        volume = create_synthetic_ct_volume()
        mask = create_synthetic_femur_mask()
        
        # Сохранение
        image_path = images_dir / f"case_{i:04d}.nii.gz"
        label_path = labels_dir / f"case_{i:04d}_mask.nii.gz"
        
        affine = np.eye(4)
        img_nii = nib.Nifti1Image(volume, affine)
        mask_nii = nib.Nifti1Image(mask, affine)
        
        nib.save(img_nii, image_path)
        nib.save(mask_nii, label_path)
    
    print(f"Создано 20 примеров в {data_dir}")


def create_synthetic_ct_volume():
    """
    Создание синтетического CT объема
    """
    # Базовый шум
    volume = np.random.randn(256, 256, 128) * 50
    
    # Добавление структур
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 256),
        np.linspace(-1, 1, 256),
        np.linspace(-1, 1, 128),
        indexing='ij'
    )
    
    # Тело (цилиндр)
    body = (x**2 + y**2) < 0.6
    volume[body] += 50
    
    # Кости таза (эллипсоид)
    pelvis = ((x/0.4)**2 + (y/0.5)**2 + ((z+0.3)/0.2)**2) < 1
    volume[pelvis] += 800
    
    # Бедренные кости (цилиндры)
    # Левая
    left_femur = ((x + 0.2)**2 + y**2) < 0.05**2
    left_femur = left_femur & (z > -0.3) & (z < 0.5)
    volume[left_femur] += 1000
    
    # Правая
    right_femur = ((x - 0.2)**2 + y**2) < 0.05**2
    right_femur = right_femur & (z > -0.3) & (z < 0.5)
    volume[right_femur] += 1000
    
    # Применение HU window
    volume = np.clip(volume, -1000, 2000)
    
    return volume.astype(np.float32)


def create_synthetic_femur_mask():
    """
    Создание синтетической маски бедренных костей
    """
    mask = np.zeros((256, 256, 128), dtype=np.uint8)
    
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 256),
        np.linspace(-1, 1, 256),
        np.linspace(-1, 1, 128),
        indexing='ij'
    )
    
    # Левая бедренная кость
    left_femur = ((x + 0.2)**2 + y**2) < 0.05**2
    left_femur = left_femur & (z > -0.3) & (z < 0.5)
    mask[left_femur] = 1
    
    # Правая бедренная кость  
    right_femur = ((x - 0.2)**2 + y**2) < 0.05**2
    right_femur = right_femur & (z > -0.3) & (z < 0.5)
    mask[right_femur] = 1
    
    # Добавление вариативности
    from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
    
    if np.random.rand() > 0.5:
        mask = binary_erosion(mask, iterations=1)
    else:
        mask = binary_dilation(mask, iterations=1)
    
    # Сглаживание
    mask = gaussian_filter(mask.astype(float), sigma=0.5)
    mask = (mask > 0.5).astype(np.uint8)
    
    return mask


if __name__ == "__main__":
    download_totalsegmentator_subset()
    print("\nДатасет готов!")
    print("Запустите обучение командой:")
    print("python train_monai_model.py --data-dir data/training --epochs 50")
