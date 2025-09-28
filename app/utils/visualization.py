import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Использование backend без GUI
import matplotlib.pyplot as plt
from matplotlib import colors
import asyncio
import io
from PIL import Image


async def create_visualization(
    input_path: Path,
    mask_path: Path,
    slice_idx: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Path:
    """
    Создание визуализации сегментации
    
    Args:
        input_path: путь к исходному изображению
        mask_path: путь к маске
        slice_idx: индекс среза (если None, берется средний)
        save_path: путь для сохранения
    
    Returns:
        путь к сохраненному изображению
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _create_visualization_sync,
        input_path,
        mask_path,
        slice_idx,
        save_path
    )


def _create_visualization_sync(
    input_path: Path,
    mask_path: Path,
    slice_idx: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Path:
    """
    Синхронная версия создания визуализации
    """
    # Загрузка данных
    input_nii = nib.load(input_path)
    input_volume = input_nii.get_fdata()
    
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    
    # Выбор среза
    if slice_idx is None:
        # Находим срез с максимальной площадью маски
        slice_idx = find_best_slice(mask)
    
    # Создание визуализации
    fig = create_multi_view_plot(input_volume, mask, slice_idx)
    
    # Сохранение
    if save_path is None:
        save_path = Path("data/temp") / f"viz_{input_path.stem}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def find_best_slice(mask: np.ndarray, axis: int = 2) -> int:
    """
    Поиск среза с максимальной площадью маски
    
    Args:
        mask: бинарная маска
        axis: ось (0=sagittal, 1=coronal, 2=axial)
    
    Returns:
        индекс лучшего среза
    """
    # Сумма по срезам
    if axis == 0:
        slice_sums = np.sum(mask, axis=(1, 2))
    elif axis == 1:
        slice_sums = np.sum(mask, axis=(0, 2))
    else:
        slice_sums = np.sum(mask, axis=(0, 1))
    
    # Возвращаем индекс максимума
    return np.argmax(slice_sums)


def create_multi_view_plot(
    volume: np.ndarray,
    mask: np.ndarray,
    central_slice: Optional[int] = None
) -> plt.Figure:
    """
    Создание мультивидовой визуализации (3 проекции + 3D)
    
    Args:
        volume: исходный объем
        mask: маска сегментации
        central_slice: центральный срез для каждой проекции
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Результат сегментации бедренной кости', fontsize=16)
    
    # Нормализация для отображения
    volume_norm = apply_windowing(volume)
    
    # Если центральный срез не задан, используем центр
    if central_slice is None:
        central_slice = volume.shape[2] // 2
    
    # Axial view (поперечный)
    ax = axes[0, 0]
    slice_axial = volume_norm[:, :, central_slice]
    mask_axial = mask[:, :, central_slice]
    show_slice_with_mask(ax, slice_axial, mask_axial, "Axial View")
    
    # Coronal view (фронтальный)
    ax = axes[0, 1]
    slice_coronal = volume_norm[:, volume.shape[1]//2, :]
    mask_coronal = mask[:, volume.shape[1]//2, :]
    show_slice_with_mask(ax, slice_coronal, mask_coronal, "Coronal View")
    
    # Sagittal view (сагиттальный)
    ax = axes[0, 2]
    slice_sagittal = volume_norm[volume.shape[0]//2, :, :]
    mask_sagittal = mask[volume.shape[0]//2, :, :]
    show_slice_with_mask(ax, slice_sagittal, mask_sagittal, "Sagittal View")
    
    # 3D проекция (MIP - Maximum Intensity Projection)
    ax = axes[1, 0]
    mip_axial = np.max(volume_norm, axis=2)
    mip_mask = np.max(mask, axis=2)
    show_slice_with_mask(ax, mip_axial, mip_mask, "MIP Axial")
    
    # Статистика
    ax = axes[1, 1]
    show_statistics(ax, mask, volume)
    
    # Гистограмма
    ax = axes[1, 2]
    show_histogram(ax, volume, mask)
    
    plt.tight_layout()
    return fig


def show_slice_with_mask(
    ax: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    title: str
):
    """
    Отображение среза с наложенной маской
    
    Args:
        ax: matplotlib axes
        image: изображение среза
        mask: маска среза
        title: заголовок
    """
    # Отображение изображения
    ax.imshow(image.T, cmap='gray', origin='lower')
    
    # Наложение маски
    masked = np.ma.masked_where(mask.T < 0.5, mask.T)
    ax.imshow(masked, cmap='Reds', alpha=0.3, origin='lower')
    
    ax.set_title(title)
    ax.axis('off')


def apply_windowing(
    volume: np.ndarray,
    window_center: float = 700,
    window_width: float = 2000
) -> np.ndarray:
    """
    Применение оконной функции для CT
    
    Args:
        volume: CT объем
        window_center: центр окна
        window_width: ширина окна
    
    Returns:
        объем с примененным окном
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    volume_windowed = np.clip(volume, min_value, max_value)
    volume_norm = (volume_windowed - min_value) / (max_value - min_value)
    
    return volume_norm


def show_statistics(ax: plt.Axes, mask: np.ndarray, volume: np.ndarray):
    """
    Отображение статистики сегментации
    
    Args:
        ax: matplotlib axes
        mask: маска сегментации
        volume: исходный объем
    """
    ax.axis('off')
    
    # Расчет статистики
    num_voxels = np.sum(mask > 0)
    total_voxels = mask.size
    percentage = (num_voxels / total_voxels) * 100
    
    # Оценка объема (предполагаем spacing 1x1x1 мм)
    volume_ml = num_voxels / 1000  # приблизительно
    
    # Средняя интенсивность в области маски
    mean_intensity = np.mean(volume[mask > 0]) if num_voxels > 0 else 0
    
    stats_text = f"""
    Статистика сегментации:
    
    Количество вокселей: {num_voxels:,}
    Процент от объема: {percentage:.2f}%
    Приблизительный объем: {volume_ml:.1f} мл
    
    Средняя интенсивность HU: {mean_intensity:.0f}
    
    Размеры изображения:
    {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    ax.set_title("Статистика")


def show_histogram(ax: plt.Axes, volume: np.ndarray, mask: np.ndarray):
    """
    Отображение гистограммы интенсивностей
    
    Args:
        ax: matplotlib axes
        volume: исходный объем
        mask: маска сегментации
    """
    # Интенсивности в области маски
    masked_values = volume[mask > 0]
    
    if len(masked_values) > 0:
        # Гистограмма
        ax.hist(masked_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(masked_values), color='red', linestyle='--', label=f'Среднее: {np.mean(masked_values):.0f}')
        ax.axvline(np.median(masked_values), color='green', linestyle='--', label=f'Медиана: {np.median(masked_values):.0f}')
        
        ax.set_xlabel('Интенсивность HU')
        ax.set_ylabel('Количество вокселей')
        ax.set_title('Распределение интенсивностей в сегментированной области')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Нет данных для отображения', 
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('Гистограмма')


def create_3d_rendering(mask: np.ndarray) -> np.ndarray:
    """
    Создание 3D рендеринга маски
    
    Args:
        mask: бинарная маска
    
    Returns:
        изображение 3D рендеринга
    """
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure
        
        # Создание фигуры
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Генерация поверхности
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        
        # Создание коллекции полигонов
        mesh = Poly3DCollection(verts[faces], alpha=0.7, facecolor='cyan', edgecolor='none')
        ax.add_collection3d(mesh)
        
        # Настройка осей
        ax.set_xlim(0, mask.shape[0])
        ax.set_ylim(0, mask.shape[1])
        ax.set_zlim(0, mask.shape[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D визуализация бедренной кости')
        
        # Сохранение в массив
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return image
        
    except Exception as e:
        print(f"Ошибка 3D рендеринга: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def create_animation_frames(
    volume: np.ndarray,
    mask: np.ndarray,
    axis: int = 2,
    step: int = 5
) -> list:
    """
    Создание кадров для анимации прохода по срезам
    
    Args:
        volume: исходный объем
        mask: маска сегментации
        axis: ось прохода
        step: шаг между кадрами
    
    Returns:
        список изображений-кадров
    """
    frames = []
    volume_norm = apply_windowing(volume)
    
    # Определение диапазона срезов
    if axis == 0:
        num_slices = volume.shape[0]
        get_slice = lambda i: (volume_norm[i, :, :], mask[i, :, :])
    elif axis == 1:
        num_slices = volume.shape[1]
        get_slice = lambda i: (volume_norm[:, i, :], mask[:, i, :])
    else:
        num_slices = volume.shape[2]
        get_slice = lambda i: (volume_norm[:, :, i], mask[:, :, i])
    
    # Создание кадров
    for i in range(0, num_slices, step):
        img_slice, mask_slice = get_slice(i)
        
        # Создание изображения с наложением
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_slice.T, cmap='gray', origin='lower')
        
        masked = np.ma.masked_where(mask_slice.T < 0.5, mask_slice.T)
        ax.imshow(masked, cmap='Reds', alpha=0.3, origin='lower')
        
        ax.set_title(f'Срез {i}/{num_slices}')
        ax.axis('off')
        
        # Преобразование в массив
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
    
    return frames