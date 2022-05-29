## Сервис по подсчету моржей на изображении от команды YOCO322
### Разрезание данных
Разрезание производится с помощью скрипта `utils/crop.py`. Внутри файла 
будут следующие параметры:
- `yolo_size` - размер окна для разрезания/выходного изображения;
- `tr` - threshold в долях, который используется для настройки нахлеста изображения на предыдущее;
- `INPUT_IMG` - Путь до папки с изображениями для нарезки;
- `INPUT_JSON` - Путь до разметки в формате JSON;
- `OUTPUT_PATH_IMAGES` - Путь до выходной папки с изображениями;
- `OUTPUT_PATH_LABELS` - Путь до выходной папки с разметкой в формате txt.


### Обучение и конвертация модели
Скачиваем проект [YOLOv5](https://github.com/ultralytics/yolov5) и 
устанавливаем требуемые зависимости из `requirements.txt`. Настраиваем обучение 
на своих данных [тут](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

Далее запускаем обучение:
```python
python train.py --img 1280 --batch 4 --epochs 30 --weights yolov5x6 --data custom.yaml --save-period 5
```

После обучения конвертируем модель в ONNX формат:
```python
python export.py --weights PATH2BEST_PT_MODEL --include onnx --imgsz 5184 --dynamic
```

### Запуск сервиса
Переносим обученную и сконвертированную модель в `PATH2PROJECT/walrus_yolo/models/`. 
Предобученную версию можно скачать [тут](https://disk.yandex.ru/d/mQF62pvS_6QqCQ).

Устанавливаем необходимый зависимости:
```
pip install -r requirements.txt
```
Запускаем сервис:
```bash
streamlit run main.py
```