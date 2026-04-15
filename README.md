# Smartphone Addiction Classification (ML + Docker)

## Опис
Цей проєкт реалізує модель машинного навчання для визначення залежності користувача від смартфона на основі його поведінкових характеристик.

Модель приймає числові дані (час використання, кількість відкриттів додатків, сон тощо) і повертає результат:
- Addicted
- Not addicted

## Використані технології
- Python
- pandas
- scikit-learn
- joblib
- Docker

## Дані
Використано датасет:
<a href="https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction">Smartphone Usage and Addiction Prediction</a>

## Запуск без Docker
pip install -r requirements.txt
python src/train.py
python src/predict.py

## Запуск через Docker
docker build -t ml-console .
docker run -it ml-console

## ⌨️ Приклад вводу
20 7.5 3.2 1.0 5.0 6.5 120 80 9.0

## 📈 Результат
Точність моделі ~0.91, найкраще при 9 компонентах PCA.

## 📂 Структура проєкту
project/
├── artifacts/
├── data/
├── src/
├── requirements.txt
└── Dockerfile

