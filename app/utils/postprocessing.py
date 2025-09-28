import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List
import asyncio
import zipfile
from PIL import Image
import io


async def export_to_stl(mask_path: Path, output_path: Path):
    """
    Экспорт маски сегментации в STL формат
    
    Args:
        mask_path: путь к NIFTI маске
        output_path: путь для сохранения STL
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _export_to_stl_sync, mask_path, output_path)


def _export_to_stl_sync(mask_path: Path, output_path: Path):
    """
    Синхронная версия экспорта в STL
    """
    try:
        from skimage import measure
        from stl import mesh
    except ImportError:
        print("Установите numpy-stl и scikit-image для экспорта в STL")
        return
    
    # Загрузка маски
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    affine = mask_nii.affine
    
    # Получение spacing из affine матрицы
    spacing = np.abs(np.diag(affine)[:3])
    
    # Сглаживание маски
    mask = smooth_mask(mask)
    
    # Генерация mesh с помощью marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        mask,
        level=0.5,
        spacing=spacing,
        step_size=1
    )
    
    # Создание STL mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j], :]
    
    # Сохранение STL
    stl_mesh.save(str(output_path))
    print(f"STL сохранен: {output_path}")


async def export_to_png_slices(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    axis: int = 2
):
    """
    Экспорт срезов с наложением маски в PNG
    
    Args:
        input_path: путь к исходному NIFTI
        mask_path: путь к маске сегментации
        output_path: путь для сохранения ZIP архива
        axis: ось для срезов (0=sagittal, 1=coronal, 2=axial)
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, 
        _export_to_png_slices_sync, 
        input_path, 
        mask_path, 
        output_path, 
        axis
    )


def _export_to_png_slices_sync(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    axis: int = 2
):
    """
    Синхронная версия экспорта PNG срезов
    """
    # Загрузка данных
    input_nii = nib.load(input_path)
    input_volume = input_nii.get_fdata()
    
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    
    # Нормализация изображения для визуализации
    input_norm = normalize_for_display(input_volume)
    
    # Создание ZIP архива
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Получение срезов вдоль выбранной оси
        if axis == 0:  # Sagittal
            num_slices = input_volume.shape[0]
            get_slice = lambda i: (input_norm[i, :, :], mask[i, :, :])
            prefix = "sagittal"
        elif axis == 1:  # Coronal
            num_slices = input_volume.shape[1]
            get_slice = lambda i: (input_norm[:, i, :], mask[:, i, :])
            prefix = "coronal"
        else:  # Axial
            num_slices = input_volume.shape[2]
            get_slice = lambda i: (input_norm[:, :, i], mask[:, :, i])
            prefix = "axial"
        
        # Экспорт каждого среза
        for i in range(num_slices):
            img_slice, mask_slice = get_slice(i)
            
            # Создание RGB изображения с наложением маски
            rgb_image = create_overlay_image(img_slice, mask_slice)
            
            # Сохранение в памяти
            img_buffer = io.BytesIO()
            Image.fromarray(rgb_image).save(img_buffer, format='PNG')
            
            # Добавление в архив
            zipf.writestr(f"{prefix}_slice_{i:04d}.png", img_buffer.getvalue())
    
    print(f"PNG срезы сохранены в: {output_path}")


def smooth_mask(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Сглаживание бинарной маски
    
    Args:
        mask: бинарная маска
        sigma: параметр сглаживания
    
    Returns:
        сглаженная маска
    """
    from scipy.ndimage import gaussian_filter, binary_fill_holes
    
    # Заполнение дыр
    mask = binary_fill_holes(mask)
    
    # Гауссово сглаживание
    mask_smooth = gaussian_filter(mask.astype(float), sigma=sigma)
    
    # Пороговая обработка
    mask_smooth = (mask_smooth > 0.5).astype(float)
    
    return mask_smooth


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Выделение самой большой связной компоненты
    
    Args:
        mask: бинарная маска
    
    Returns:
        маска с самой большой компонентой
    """
    from scipy.ndimage import label
    
    # Поиск связных компонент
    labeled_mask, num_features = label(mask)
    
    if num_features <= 1:
        return mask
    
    # Подсчет размера каждой компоненты
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # Игнорируем фон
    
    # Находим самую большую компоненту
    largest_component_label = component_sizes.argmax()
    
    # Создаем маску только с самой большой компонентой
    return (labeled_mask == largest_component_label).astype(mask.dtype)


def remove_small_objects(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Удаление малых объектов из маски
    
    Args:
        mask: бинарная маска
        min_size: минимальный размер объекта в вокселях
    
    Returns:
        очищенная маска
    """
    from scipy.ndimage import label
    
    # Поиск связных компонент
    labeled_mask, num_features = label(mask)
    
    # Подсчет размера каждой компоненты
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Удаление малых компонент
    for i in range(1, num_features + 1):
        if component_sizes[i] < min_size:
            mask[labeled_mask == i] = 0
    
    return mask


def morphological_closing(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Морфологическое закрытие для заполнения дыр
    
    Args:
        mask: бинарная маска
        iterations: количество итераций
    
    Returns:
        обработанная маска
    """
    from scipy.ndimage import binary_closing
    
    return binary_closing(mask, iterations=iterations).astype(mask.dtype)


def normalize_for_display(volume: np.ndarray) -> np.ndarray:
    """
    Нормализация объема для отображения (0-255)
    
    Args:
        volume: исходный объем
    
    Returns:
        нормализованный объем для отображения
    """
    # Применение костного окна для CT
    window_center = 700
    window_width = 2000
    
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    volume_windowed = np.clip(volume, min_value, max_value)
    
    # Нормализация к 0-255
    volume_norm = (volume_windowed - min_value) / (max_value - min_value)
    volume_norm = (volume_norm * 255).astype(np.uint8)
    
    return volume_norm


def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    mask_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Создание изображения с наложением маски
    
    Args:
        image: исходное изображение (grayscale)
        mask: бинарная маска
        alpha: прозрачность маски
        mask_color: цвет маски (R, G, B)
    
    Returns:
        RGB изображение с наложенной маской
    """
    # Преобразование grayscale в RGB
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    else:
        rgb_image = image
    
    # Убедимся, что изображение в диапазоне 0-255
    if rgb_image.max() <= 1:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Создание цветной маски
    mask_colored = np.zeros_like(rgb_image)
    mask_bool = mask > 0.5
    
    for i, color_value in enumerate(mask_color):
        mask_colored[:, :, i][mask_bool] = color_value
    
    # Наложение маски
    overlay = rgb_image.copy()
    overlay[mask_bool] = (
        (1 - alpha) * rgb_image[mask_bool] +
        alpha * mask_colored[mask_bool]
    ).astype(np.uint8)
    
    return overlay


def calculate_volume(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """
    Расчет объема сегментированной области
    
    Args:
        mask: бинарная маска
        spacing: разрешение вокселей (x, y, z) в мм
    
    Returns:
        объем в мл (кубических сантиметрах)
    """
    # Количество вокселей
    num_voxels = np.sum(mask > 0)
    
    # Объем одного вокселя в мм³
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    # Общий объем в мм³
    total_volume_mm3 = num_voxels * voxel_volume
    
    # Преобразование в мл (см³)
    total_volume_ml = total_volume_mm3 / 1000
    
    return total_volume_ml


def get_bounding_box(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    """
    Получение ограничивающего прямоугольника для маски
    
    Args:
        mask: бинарная маска
    
    Returns:
        кортеж из трех slice объектов для x, y, z
    """
    # Находим ненулевые элементы
    nonzero_idx = np.where(mask > 0)
    
    if len(nonzero_idx[0]) == 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    
    # Определяем границы
    x_slice = slice(nonzero_idx[0].min(), nonzero_idx[0].max() + 1)
    y_slice = slice(nonzero_idx[1].min(), nonzero_idx[1].max() + 1)
    z_slice = slice(nonzero_idx[2].min(), nonzero_idx[2].max() + 1)
    
    return x_slice, y_slice, z_slice