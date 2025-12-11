Классический проект по классификации изображений «Кошки vs Собаки», выполненный на PyTorch. На основе датасета https://www.microsoft.com/en-us/download/details.aspx?id=54765 
Репозиторий включает полный ML-pipeline.

подготовка датасета

очистка битых изображений

автоматический сплит на train/val/test

динамическая сверточная нейросеть (DynamicConvNet)

обучение с чекпоинтами

сохранение лучшей модели

визуализация результатов (графики, confusion matrix)

анализ и сравнение нескольких моделей


Проект полностью готов для запуска и расширения.

cats-vs-dogs-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                     # появляется после подготовки датасета
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── main.py               # запуск одного эксперимента
│   ├── experiments.py        # массовый запуск нескольких моделей
│   ├── select_model.py       # статистика по чекпоинтам
│
│   ├── model.py              # DynamicConvNet и активации
│   ├── dataset.py            # transforms + загрузка датасета
│   ├── trainer.py            # класс Trainer: train, val, test
│   └── utils.py              # вспомогательные функции
│
├── tools/
│   ├── cleaner.py            # удаление битых изображений
│   └── split.py              # разбиение датасета на train/val/test
│
└── results/                  # генерируется автоматически
    └── <experiment_name>/
        ├── config.json
        ├── history.json
        ├── models/
        └── figures/
