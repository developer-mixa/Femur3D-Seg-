import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models.segmentation import FemurSegmentationModel
from app.utils.preprocessing import preprocess_nifti
from app.utils.postprocessing import export_to_stl, export_to_png_slices
from app.utils.visualization import create_visualization

# Инициализация FastAPI
app = FastAPI(
    title="Femur 3D Segmentation API",
    description="API для сегментации бедренной кости из CT снимков",
    version="1.0.0"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
MODEL = None
TASKS = {}  # Хранилище задач
DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Создание директорий
for dir_path in [INPUT_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Инициализация модели при запуске"""
    global MODEL
    print("Загрузка модели сегментации...")
    MODEL = FemurSegmentationModel('monai')
    await MODEL.load_model()
    print("Модель загружена успешно!")


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "Femur 3D Segmentation",
        "status": "active",
        "endpoints": {
            "upload": "/upload",
            "segment": "/segment/{task_id}",
            "status": "/status/{task_id}",
            "result": "/result/{task_id}",
            "visualize": "/visualize/{task_id}",
            "export": "/export/{task_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Загрузка NIFTI файла для обработки
    """
    # Проверка формата файла
    if not file.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате NIFTI")
    
    # Генерация уникального ID задачи
    task_id = str(uuid.uuid4())
    
    # Сохранение файла
    file_path = INPUT_DIR / f"{task_id}.nii.gz"
    try:
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {str(e)}")
    
    # Создание задачи
    TASKS[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "input_path": str(file_path),
        "output_path": None,
        "created_at": datetime.now().isoformat(),
        "error": None
    }
    
    return {
        "task_id": task_id,
        "status": "uploaded",
        "message": f"Файл {file.filename} успешно загружен"
    }


@app.post("/segment/{task_id}")
async def segment(task_id: str, background_tasks: BackgroundTasks):
    """
    Запуск процесса сегментации
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    
    if task["status"] != "uploaded":
        raise HTTPException(
            status_code=400, 
            detail=f"Неверный статус задачи: {task['status']}"
        )
    
    # Запуск сегментации в фоне
    background_tasks.add_task(run_segmentation, task_id)
    
    task["status"] = "processing"
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Сегментация запущена"
    }


async def run_segmentation(task_id: str):
    """
    Фоновая задача для выполнения сегментации
    """
    task = TASKS[task_id]
    
    try:
        # Загрузка и предобработка данных
        input_path = Path(task["input_path"])
        preprocessed_data = await preprocess_nifti(input_path)
        
        # Сегментация
        segmentation_mask = await MODEL.segment(preprocessed_data)
        
        # Сохранение результата
        output_path = OUTPUT_DIR / f"{task_id}_mask.nii.gz"
        await MODEL.save_result(segmentation_mask, output_path, preprocessed_data["metadata"])
        
        task["output_path"] = str(output_path)
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        print(f"Ошибка сегментации для задачи {task_id}: {e}")


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Получение статуса задачи
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "filename": task["filename"],
        "created_at": task["created_at"],
        "error": task["error"]
    }


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Получение результата сегментации (NIFTI маска)
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Задача не завершена. Статус: {task['status']}"
        )
    
    output_path = Path(task["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Файл результата не найден")
    
    return FileResponse(
        path=output_path,
        media_type="application/gzip",
        filename=f"{task_id}_segmentation.nii.gz"
    )


@app.get("/visualize/{task_id}")
async def visualize(task_id: str, slice_idx: Optional[int] = None):
    """
    Визуализация результата сегментации
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Задача не завершена. Статус: {task['status']}"
        )
    
    try:
        # Создание визуализации
        input_path = Path(task["input_path"])
        output_path = Path(task["output_path"])
        
        viz_path = await create_visualization(
            input_path, 
            output_path,
            slice_idx=slice_idx
        )
        
        return FileResponse(
            path=viz_path,
            media_type="image/png"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка визуализации: {str(e)}")


@app.get("/export/{task_id}")
async def export(
    task_id: str, 
    format: str = "stl",
    background_tasks: BackgroundTasks = None
):
    """
    Экспорт результата в различных форматах
    
    Форматы:
    - stl: 3D mesh
    - png: набор PNG срезов (zip архив)
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Задача не завершена. Статус: {task['status']}"
        )
    
    output_path = Path(task["output_path"])
    
    try:
        if format == "stl":
            # Экспорт в STL
            stl_path = OUTPUT_DIR / f"{task_id}_femur.stl"
            if not stl_path.exists():
                await export_to_stl(output_path, stl_path)
            
            return FileResponse(
                path=stl_path,
                media_type="model/stl",
                filename=f"femur_{task_id}.stl"
            )
        
        elif format == "png":
            # Экспорт в PNG срезы
            zip_path = OUTPUT_DIR / f"{task_id}_slices.zip"
            if not zip_path.exists():
                input_path = Path(task["input_path"])
                await export_to_png_slices(input_path, output_path, zip_path)
            
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename=f"slices_{task_id}.zip"
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат: {format}. Используйте 'stl' или 'png'"
            )
    
    except Exception as e:
        raise e
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    Удаление задачи и связанных файлов
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = TASKS[task_id]
    
    # Удаление файлов
    for path_key in ["input_path", "output_path"]:
        if task[path_key]:
            path = Path(task[path_key])
            if path.exists():
                os.remove(path)
    
    # Удаление задачи из памяти
    del TASKS[task_id]
    
    return {"message": f"Задача {task_id} успешно удалена"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)