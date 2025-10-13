import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from monai.config import print_config
from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    decollate_batch,
    list_data_collate
)
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet, SegResNet, AttentionUnet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    AsDiscreted,
    EnsureTyped,
    SaveImaged,
    Activationsd,
    Invertd,
    KeepLargestConnectedComponentd
)
from monai.utils import set_determinism, MetricReduction
from monai.inferers import sliding_window_inference
from monai.optimizers import Novograd


class FemurSegmentationTrainer:
    """
    Класс для обучения модели сегментации бедренной кости
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model_type: str = "unet",
        learning_rate: float = 1e-4,
        batch_size: int = 2,
        num_epochs: int = 100,
        val_interval: int = 2,
        device: str = None
    ):
        """
        Инициализация тренера
        
        Args:
            data_dir: директория с данными
            output_dir: директория для сохранения результатов
            model_type: тип модели (unet, segresnet, attention_unet)
            learning_rate: скорость обучения
            batch_size: размер батча
            num_epochs: количество эпох
            val_interval: интервал валидации
            device: устройство для обучения
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_interval = val_interval
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
        # Инициализация модели и оптимизатора
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.dice_metric = None
        
    def prepare_data_list(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Подготовка списка файлов для обучения
        
        Returns:
            train_files: список файлов для обучения
            val_files: список файлов для валидации
        """
        # Поиск всех NIFTI файлов
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"
        
        # Создание списка пар изображение-маска
        data_list = []
        
        for image_file in sorted(images_dir.glob("*.nii*")):
            # Ищем соответствующую маску
            mask_name = image_file.name.replace(".nii", "_mask.nii")
            mask_file = labels_dir / mask_name
            
            # Альтернативные имена масок
            if not mask_file.exists():
                mask_file = labels_dir / image_file.name
            
            if mask_file.exists():
                data_list.append({
                    "image": str(image_file),
                    "label": str(mask_file)
                })
        
        if len(data_list) == 0:
            print("Создание синтетических данных для демонстрации...")
            data_list = self.create_synthetic_data()
        
        # Разделение на train/val (80/20)
        split_idx = int(len(data_list) * 0.8)
        train_files = data_list[:split_idx]
        val_files = data_list[split_idx:]
        
        print(f"Найдено {len(train_files)} файлов для обучения и {len(val_files)} для валидации")
        
        return train_files, val_files
    
    def create_synthetic_data(self) -> List[Dict]:
        """
        Создание синтетических данных для обучения (для демонстрации)
        """
        import nibabel as nib
        from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
        
        synthetic_dir = self.data_dir / "synthetic"
        images_dir = synthetic_dir / "images"
        labels_dir = synthetic_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        data_list = []
        
        # Создаем 10 синтетических примеров
        for i in range(10):
            # Создание синтетического CT изображения
            volume = np.random.randn(128, 128, 64) * 100
            
            # Добавляем структуры похожие на кости (высокая интенсивность)
            x, y, z = np.meshgrid(
                np.linspace(-1, 1, 128),
                np.linspace(-1, 1, 128),
                np.linspace(-1, 1, 64)
            )
            
            # Создаем цилиндрические структуры для бедренных костей
            # Левая бедренная кость
            left_femur = ((x + 0.3)**2 + y**2) < 0.1
            left_femur = left_femur & (z > -0.5) & (z < 0.5)
            
            # Правая бедренная кость
            right_femur = ((x - 0.3)**2 + y**2) < 0.1
            right_femur = right_femur & (z > -0.5) & (z < 0.5)
            
            # Объединение масок
            mask = (left_femur | right_femur).astype(np.float32)
            
            # Добавление вариативности
            if i % 2 == 0:
                mask = binary_erosion(mask, iterations=1).astype(np.float32)
            else:
                mask = binary_dilation(mask, iterations=1).astype(np.float32)
            
            # Сглаживание
            mask = gaussian_filter(mask, sigma=0.5)
            mask = (mask > 0.5).astype(np.float32)
            
            # Добавление костной структуры в изображение
            volume[mask > 0] = 1000 + np.random.randn(np.sum(mask > 0)) * 100
            
            # Добавление мягких тканей
            soft_tissue = gaussian_filter(np.random.randn(128, 128, 64), sigma=5) * 50
            volume += soft_tissue
            
            # Применение костного окна HU
            volume = np.clip(volume, -200, 1500)
            
            # Сохранение
            image_path = images_dir / f"synthetic_{i:03d}.nii.gz"
            label_path = labels_dir / f"synthetic_{i:03d}_mask.nii.gz"
            
            # Создание NIFTI
            affine = np.eye(4)
            img_nii = nib.Nifti1Image(volume.astype(np.float32), affine)
            mask_nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
            
            nib.save(img_nii, image_path)
            nib.save(mask_nii, label_path)
            
            data_list.append({
                "image": str(image_path),
                "label": str(label_path)
            })
            
        print(f"Создано {len(data_list)} синтетических примеров")
        return data_list
    
    def get_transforms(self) -> Tuple[Compose, Compose]:
        """
        Получение трансформаций для обучения и валидации
        """
        # Трансформации для обучения
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=1500,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            
            # Аугментации
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0
            # ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi/15),
                shear_range=(0.1, 0.1, 0.1),
                translate_range=(10, 10, 10),
                scale_range=(0.1, 0.1, 0.1)
            ),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.3),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            
            EnsureTyped(keys=["image", "label"])
        ])
        
        # Трансформации для валидации
        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=1500,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"])
        ])
        
        return train_transforms, val_transforms
    
    def create_model(self) -> nn.Module:
        """
        Создание модели
        """
        if self.model_type == "unet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH
            )
        elif self.model_type == "segresnet":
            model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                dropout_prob=0.2
            )
        elif self.model_type == "attention_unet":
            model = AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self):
        """
        Основной цикл обучения
        """
        # Подготовка данных
        train_files, val_files = self.prepare_data_list()
        train_transforms, val_transforms = self.get_transforms()
        
        # Создание датасетов
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4
        )
        
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=4
        )
        
        # Создание загрузчиков данных
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        # Создание модели
        self.model = self.create_model()
        print(f"Модель {self.model_type} создана")
        
        # Loss function
        self.loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            batch=True
        )
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Планировщик learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs
        )
        
        # Метрики
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean"
        )
        
        # Лучшая метрика
        best_metric = -1
        best_metric_epoch = -1
        
        # Цикл обучения
        print("Начало обучения...")
        for epoch in range(self.num_epochs):
            print(f"\n{'-' * 50}")
            print(f"Эпоха {epoch + 1}/{self.num_epochs}")
            
            # Обучение
            self.model.train()
            epoch_loss = 0
            step = 0
            
            train_bar = tqdm(train_loader, desc="Обучение")
            for batch_data in train_bar:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device)
                )
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                item_loss = f"{loss.item():.4f}"
                train_bar.set_postfix({"loss": item_loss})
            
            epoch_loss /= step
            self.train_losses.append(epoch_loss)
            print(f"Средний loss обучения: {epoch_loss:.4f}")
            
            # Валидация
            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    metric_values = []
                    
                    val_bar = tqdm(val_loader, desc="Валидация")
                    for val_data in val_bar:
                        val_inputs, val_labels = (
                            val_data["image"].to(self.device),
                            val_data["label"].to(self.device)
                        )
                        
                        # Sliding window inference для больших изображений
                        roi_size = (96, 96, 96)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(
                            val_inputs,
                            roi_size,
                            sw_batch_size,
                            self.model,
                            overlap=0.5
                        )
                        
                        # Loss
                        loss = self.loss_function(val_outputs, val_labels)
                        val_loss += loss.item()
                        
                        # Dice metric
                        val_outputs = [post_pred({"pred": i}) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label({"label": i}) for i in decollate_batch(val_labels)]

                        # Извлечение тензоров по ключам для метрики
                        y_pred = [x["pred"] for x in val_outputs]
                        y_true = [x["label"] for x in val_labels]

                        self.dice_metric(y_pred=y_pred, y=y_true)
                    
                    val_loss /= len(val_loader)
                    metric = self.dice_metric.aggregate().item()
                    self.dice_metric.reset()
                    
                    self.val_losses.append(val_loss)
                    self.val_dice_scores.append(metric)
                    
                    print(f"Validation loss: {val_loss:.4f}")
                    print(f"Validation Dice: {metric:.4f}")
                    
                    # Сохранение лучшей модели
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_metric': best_metric,
                            'loss': val_loss,
                        }, self.output_dir / "femur_unet_best.pth")
                        print(f"Сохранена лучшая модель с Dice: {best_metric:.4f}")
            
            scheduler.step()
        
        print(f"\nОбучение завершено!")
        print(f"Лучший Dice score: {best_metric:.4f} на эпохе {best_metric_epoch}")
        
        # Сохранение финальной модели
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
        }, self.output_dir / "femur_unet_final.pth")
        
        # Сохранение графиков
        self.plot_training_history()
    
    def plot_training_history(self):
        """
        Построение графиков обучения
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            val_epochs = list(range(self.val_interval, len(self.train_losses) + 1, self.val_interval))
            axes[0].plot(val_epochs[:len(self.val_losses)], self.val_losses, label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Dice Score
        if self.val_dice_scores:
            val_epochs = list(range(self.val_interval, len(self.train_losses) + 1, self.val_interval))
            axes[1].plot(val_epochs[:len(self.val_dice_scores)], self.val_dice_scores, marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Dice Score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()


# Постобработка для метрик
post_pred = Compose([
    EnsureTyped(keys="pred"),
    Activationsd(keys="pred", softmax=True),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2)
])

post_label = Compose([
    EnsureTyped(keys="label"),
    AsDiscreted(keys="label", to_onehot=2)
])


def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(description='Обучение модели сегментации бедренной кости')
    parser.add_argument('--data-dir', type=str, default='data/training',
                        help='Путь к директории с данными')
    parser.add_argument('--output-dir', type=str, default='data/models',
                        help='Путь для сохранения модели')
    parser.add_argument('--model-type', type=str, default='unet',
                        choices=['unet', 'segresnet', 'attention_unet'],
                        help='Тип модели')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Количество эпох')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Скорость обучения')
    parser.add_argument('--val-interval', type=int, default=2,
                        help='Интервал валидации')
    parser.add_argument('--device', type=str, default=None,
                        help='Устройство (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Установка seed для воспроизводимости
    set_determinism(seed=args.seed)
    
    # Создание тренера
    trainer = FemurSegmentationTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device
    )
    
    # Запуск обучения
    trainer.train()
    
    # Копирование лучшей модели в нужную директорию
    import shutil
    best_model_path = Path(args.output_dir) / "femur_unet_best.pth"
    target_path = Path("data/models/femur_unet.pth")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_model_path.exists():
        shutil.copy(best_model_path, target_path)
        print(f"\nМодель скопирована в {target_path}")
        print("Теперь вы можете использовать эту модель в сервисе сегментации!")


if __name__ == "__main__":
    main()