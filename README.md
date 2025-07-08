# Анализ тональности текста (Sentiment Analysis)

Проект для классификации эмоций в тексте на английском языке с использованием машинного обучения. Определяет, является ли текст позитивным (1) или негативным (0).

## Особенности
- Обучение модели Logistic Regression на датасете Twitter Sentiment Analysis (1.6 млн твитов)
- Веб-интерфейс на Flask для тестирования модели
- Точность модели: 77% (F1-score: 0.77-0.78)

## Технологии
- Python 3.8+
- Библиотеки: scikit-learn, NLTK, Flask, pandas
- Модель: Logistic Regression + TF-IDF векторизация

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```
Файл requirements.txt должен содержать:

```
pandas>=1.3.0
scikit-learn>=1.0.0
flask>=2.0.0
nltk>=3.6.0
joblib>=1.0.0
```

###2. Загрузка данных NLTK

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

###3. Обучение модели

```
python train.py
```

###4. Запуск веб-интерфейса

```
python app.py
```

##Структура проекта

```
sentiment-analysis/
├── train.py            # Скрипт обучения модели
├── app.py              # Flask-приложение
├── templates/
│   └── index.html      # HTML-шаблон
├── sentiment_model.pkl      # Обученная модель
└── tfidf_vectorizer.pkl    # Векторизатор текста
```
