# Генератор аккомпанемента к MIDI

Система машинного обучения которая генерирует барабаны и фортепиано по входной гитарной MIDI дорожке.

**Архитектура:** Bidirectional GRU  
**Датасет:** LakhMIDI (116k MIDI файлов)  
**Интерфейс:** Веб-приложение на Flask

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/vanyzz/TIProject.git
cd TIProject
```

### 2. Создать виртуальное окружение
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Запустить интерфейс
```bash
python app.py
```

Открыть в браузере: **http://localhost:5000**

## Обучение моделей

### Скачать датасет LakhMIDI
```bash
python -c "
import urllib.request, tarfile
urllib.request.urlretrieve(
    'http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz',
    'data/lmd_matched.tar.gz'
)
with tarfile.open('data/lmd_matched.tar.gz', 'r:gz') as t:
    t.extractall('data/lmd')
print('Готово!')
"
```

### Обучить модель барабанов
```bash
python src/train_guitar_drums_v2.py
```

### Обучить модель фортепиано
```bash
python src/train_guitar_piano_v2.py
```

## Технические детали

| Параметр | Значение |
|---|---|
| Представление данных | Piano Roll (128×T, 16 шагов/сек) |
| Архитектура | Bidirectional GRU, hidden_size=128 |
| Параметров модели | 264 064 |
| Функция потерь (барабаны) | DrumAwareLoss (бочка w=20, малый w=15) |
| Функция потерь (пианино) | FocalMSE (pos_weight=10) |
| Оптимизатор | Adam, lr=0.001 |
| Val loss (барабаны) | 0.1588 |
| Val loss (фортепиано) | 0.0645 |
| Квантизация | 1/16 нота |
