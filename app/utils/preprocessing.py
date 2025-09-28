import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import asyncio


async def preprocess_nifti(file_path: Path) -> Dict[str, Any]:
    """
    Загрузка и предобработка NIFTI файла
    
    Args:
        file_path: путь к NIFTI файлу
    
    Returns:
        словарь с обработанными данными и метаданными
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _preprocess_nifti_sync, file_path)


def _preprocess_nifti_sync(file_path: Path) -> Dict[str, Any]:
    """
    Синхронная версия предобработки
    """
    # Загрузка NIFTI
    nifti_img = nib.load(file_path)
    volume = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header
    
    # Получение метаданных
    metadata = {
        "affine": affine,
        "shape": volume.shape,
        "spacing": header.get_zooms()[:3] if hasattr(header, 'get_zooms') else (1, 1, 1),
        "dtype": volume.dtype,
        "min_value": float(np.min(volume)),
        "max_value": float(np.max(volume)),
        "mean_value": float(np.mean(volume)),
        "std_value": float(np.std(volume))
    }
    
    # Базовая предобработка
    processed_volume = preprocess_ct_volume(volume)
    
    return {
        "path": str(file_path),
        "volume": processed_volume,
        "original_volume": volume,
        "metadata": metadata,
        "task_id": file_path.stem
    }


def preprocess_ct_volume(
    volume: np.ndarray,
    window_center: float = 400,
    window_width: float = 1800,
    normalize: bool = True
) -> np.ndarray:
    """
    Предобработка CT изображения с применением оконной функции
    
    Args:
        volume: исходный 3D массив
        window_center: центр окна HU
        window_width: ширина окна HU
        normalize: нормализовать к диапазону [0, 1]
    
    Returns:
        обработанный объем
    """
    # Применение оконной функции (windowing)
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    volume = np.clip(volume, min_value, max_value)
    
    if normalize:
        # Нормализация к диапазону [0, 1]
        volume = (volume - min_value) / (max_value - min_value)
    
    return volume


def resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Ресемплирование объема к целевому разрешению
    
    Args:
        volume: исходный объем
        current_spacing: текущее разрешение (x, y, z)
        target_spacing: целевое разрешение
    
    Returns:
        ресемплированный объем и новое разрешение
    """
    from scipy.ndimage import zoom
    
    # Вычисление факторов масштабирования
    scaling_factors = [
        current_spacing[i] / target_spacing[i]
        for i in range(3)
    ]
    
    # Ресемплирование
    resampled_volume = zoom(volume, scaling_factors, order=1)
    
    return resampled_volume, target_spacing


def crop_to_body(volume: np.ndarray, margin: int = 10) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Обрезка объема до области тела (удаление пустого пространства)
    
    Args:
        volume: исходный объем
        margin: отступ в вокселях
    
    Returns:
        обрезанный объем и координаты обрезки
    """
    # Находим ненулевые вокселы
    nonzero_idx = np.where(volume > 0.01)
    
    if len(nonzero_idx[0]) == 0:
        return volume, {"x_min": 0, "x_max": volume.shape[0],
                        "y_min": 0, "y_max": volume.shape[1],
                        "z_min": 0, "z_max": volume.shape[2]}
    
    # Определяем границы
    x_min, x_max = max(0, nonzero_idx[0].min() - margin), min(volume.shape[0], nonzero_idx[0].max() + margin)
    y_min, y_max = max(0, nonzero_idx[1].min() - margin), min(volume.shape[1], nonzero_idx[1].max() + margin)
    z_min, z_max = max(0, nonzero_idx[2].min() - margin), min(volume.shape[2], nonzero_idx[2].max() + margin)
    
    # Обрезка
    cropped_volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
    
    crop_info = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max
    }
    
    return cropped_volume, crop_info


def apply_bone_window(volume: np.ndarray) -> np.ndarray:
    """
    Применение костного окна для CT
    
    Args:
        volume: CT объем в единицах HU
    
    Returns:
        объем с примененным костным окном
    """
    # Костное окно: центр ~700 HU, ширина ~2000 HU
    return preprocess_ct_volume(volume, window_center=700, window_width=2000)


def denoise_volume(volume: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Удаление шума из объема
    
    Args:
        volume: исходный объем
        method: метод удаления шума ('median', 'gaussian', 'bilateral')
    
    Returns:
        объем без шума
    """
    from scipy.ndimage import median_filter, gaussian_filter
    
    if method == "median":
        return median_filter(volume, size=3)
    elif method == "gaussian":
        return gaussian_filter(volume, sigma=0.5)
    elif method == "bilateral":
        # Билатеральный фильтр более сложный, используем упрощенную версию
        return gaussian_filter(volume, sigma=0.5)
    else:
        return volume


def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Нормализация интенсивности
    
    Args:
        volume: исходный объем
        method: метод нормализации ('zscore', 'minmax', 'percentile')
    
    Returns:
        нормализованный объем
    """
    if method == "zscore":
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            return (volume - mean) / std
        return volume - mean
    
    elif method == "minmax":
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val > min_val:
            return (volume - min_val) / (max_val - min_val)
        return volume - min_val
    
    elif method == "percentile":
        # Нормализация по процентилям (убираем выбросы)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        return (volume - p1) / (p99 - p1)
    
    return volume