import os
import asyncio
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEG_AVAILABLE = True
except ImportError:
    TOTALSEG_AVAILABLE = False
    print("TotalSegmentator не установлен. Используем MONAI.")

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
    AsDiscrete,
    SaveImaged
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference


class FemurSegmentationModel:
    """
    Класс для сегментации бедренной кости из CT изображений
    """
    
    def __init__(self, model_type: str = "totalsegmentator", device: str = None):
        """
        Инициализация модели
        
        Args:
            model_type: тип модели ('totalsegmentator' или 'monai')
            device: устройство для вычислений ('cuda' или 'cpu')
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transforms = None
        
        # Индексы классов для бедренной кости в TotalSegmentator
        self.femur_classes = {
            'femur_left': 85,
            'femur_right': 86
        }
        
        print(f"Используется устройство: {self.device}")
        
    async def load_model(self):
        """
        Загрузка предобученной модели
        """
        if self.model_type == "totalsegmentator" and TOTALSEG_AVAILABLE:
            await self._load_totalsegmentator()
        else:
            await self._load_monai_model()
    
    async def _load_totalsegmentator(self):
        """
        Загрузка TotalSegmentator
        """
        print("Загрузка TotalSegmentator...")
        # TotalSegmentator автоматически загружает веса при первом использовании
        self.model = "totalsegmentator"
        print("TotalSegmentator готов к использованию")
    
    async def _load_monai_model(self):
        """
        Загрузка MONAI U-Net модели
        """
        print("Загрузка MONAI модели...")
        
        # Определение архитектуры U-Net
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=5,  # фон + бедренная кость
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch"
        ).to(self.device)
        
        # Попытка загрузки весов
        weights_path = Path("data/models/femur_unet.pth")
        if weights_path.exists():
            print(f"Загрузка весов из {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Веса успешно загружены")
        else:
            print("Предобученные веса не найдены. Используется случайная инициализация.")
            print("Для лучших результатов обучите модель или скачайте веса.")
        
        self.model.eval()
        
        # Определение трансформаций для предобработки
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            # CropForegroundd(keys=["image"], source_key="image"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=1500,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            EnsureTyped(keys=["image"])
        ])
        
        print("MONAI модель загружена")
    
    async def segment(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Выполнение сегментации
        
        Args:
            data: словарь с данными изображения
        
        Returns:
            np.ndarray: маска сегментации
        """
        if self.model_type == "totalsegmentator" and TOTALSEG_AVAILABLE:
            return await self._segment_totalsegmentator(data)
        else:
            return await self._segment_monai(data)
    
    async def _segment_totalsegmentator(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Сегментация с помощью TotalSegmentator
        """
        print("Выполнение сегментации с TotalSegmentator...")
        
        input_path = data["path"]

        task_id = data.get("task_id", None)
        if task_id is None:
            raise ValueError("task_id отсутствует в данных для сегментации")

        output_dir = Path("data/temp") / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Запуск TotalSegmentator
        # task='total' - все органы, task='bones_extremities' - только кости
        try:
            # Асинхронный запуск в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                totalsegmentator,
                input_path,
                str(output_dir),
                False,  # ml
                1,  # nr_thr_resamp
                1,  # nr_thr_saving
                'body',  # task
                False,  # quiet
                False,  # verbose
                True,  # test
                1.5,  # crop_addon
                None,  # roi_subset (можно указать ['femur_left', 'femur_right'])
                'gpu' if self.device == 'cuda' else 'cpu',
                True,  # force
                False  # check_segmentations
            )
            
            # Загрузка и объединение масок бедренных костей
            mask_left = output_dir / "femur_left.nii.gz"
            mask_right = output_dir / "femur_right.nii.gz"
            
            combined_mask = None
            
            if mask_left.exists():
                left = nib.load(mask_left).get_fdata()
                combined_mask = left
            
            if mask_right.exists():
                right = nib.load(mask_right).get_fdata()
                if combined_mask is None:
                    combined_mask = right
                else:
                    combined_mask = np.maximum(combined_mask, right)
            
            # Очистка временных файлов
            import shutil
            shutil.rmtree(output_dir)
            
            print("Сегментация завершена")
            return combined_mask if combined_mask is not None else np.zeros_like(data["volume"])
            
        except Exception as e:
            import traceback

            print(f"Ошибка TotalSegmentator: {e}")
            traceback.print_exc() 
            print("Переключение на MONAI...")
            self.model_type = "monai"
            await self._load_monai_model()
            return await self._segment_monai(data)
    
    async def _segment_monai(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Сегментация с помощью MONAI модели
        """
        print("Выполнение сегментации с MONAI...")
        
        # Подготовка данных
        test_data = [{"image": data["path"]}]
        
        if self.transforms:
            test_data = self.transforms(test_data[0])
            image = test_data["image"].unsqueeze(0).to(self.device)
        else:
            # Базовая предобработка если трансформации не определены
            img_nib = nib.load(data["path"])
            image = img_nib.get_fdata()
            
            # Нормализация
            image = np.clip(image, -200, 1500)
            image = (image + 200) / 1700
            
            # Преобразование в тензор
            image = torch.from_numpy(image).float()
            image = image.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            # Использование sliding window для больших изображений
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            
            outputs = sliding_window_inference(
                image,
                roi_size,
                sw_batch_size,
                self.model,
                overlap=0.5
            )
            
            # Применение softmax и выбор класса
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            
            # Преобразование в numpy
            mask = outputs[0].cpu().numpy()
        
        # Постобработка
        mask = self._postprocess_mask(mask)
        
        print("Сегментация завершена")
        return mask
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Постобработка маски сегментации
        
        Args:
            mask: исходная маска
        
        Returns:
            обработанная маска
        """
        from scipy.ndimage import binary_closing, binary_opening, label
        
        # Морфологические операции для улучшения маски
        mask = binary_closing(mask, iterations=2)
        mask = binary_opening(mask, iterations=1)
        
        # Оставляем только самые большие компоненты (левая и правая бедренные кости)
        labeled_mask, num_features = label(mask)
        
        if num_features > 0:
            # Находим размеры компонент
            component_sizes = []
            for i in range(1, num_features + 1):
                component_sizes.append((i, np.sum(labeled_mask == i)))
            
            # Сортируем по размеру
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Оставляем две самые большие компоненты (левая и правая кость)
            final_mask = np.zeros_like(mask)
            for i in range(min(2, len(component_sizes))):
                component_id = component_sizes[i][0]
                final_mask[labeled_mask == component_id] = 1
            
            return final_mask
        
        return mask
    
    async def save_result(
        self,
        mask: np.ndarray,
        output_path: Path,
        metadata: Dict[str, Any]
    ):
        """
        Сохранение результата сегментации
        
        Args:
            mask: маска сегментации
            output_path: путь для сохранения
            metadata: метаданные изображения
        """
        # Создание NIFTI изображения
        if "affine" in metadata:
            mask_nii = nib.Nifti1Image(mask.astype(np.uint8), metadata["affine"])
        else:
            mask_nii = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        
        # Сохранение
        nib.save(mask_nii, output_path)
        print(f"Результат сохранен: {output_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        """
        info = {
            "model_type": self.model_type,
            "device": self.device,
            "status": "loaded" if self.model else "not_loaded"
        }
        
        if self.model_type == "monai" and self.model:
            info["parameters"] = sum(p.numel() for p in self.model.parameters())
            info["architecture"] = "UNet 3D"
        elif self.model_type == "totalsegmentator":
            info["architecture"] = "nnU-Net"
            info["classes"] = list(self.femur_classes.keys())
        
        return info