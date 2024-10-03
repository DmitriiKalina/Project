import spacy
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    print("Модель не найдена. Устанавливаем модель...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "ru_core_news_sm"])
    nlp = spacy.load("ru_core_news_sm")

# Начальный словарь синонимических групп
synonym_groups = {
    'Заработная плата': ['зарплата', 'бабосики', 'зп', 'оплата', 'деньги', 'премии', 'бонусы'],
    'Команда': ['команда', 'коллеги', 'коллектив', 'поддержка'],
    'Атмосфера': ['атмосфера', 'климат', 'настроение', 'удовольствие от работы'],
    'Руководство': ['шеф', 'начальник', 'руководитель', 'начальство'],
    'Интересные задачи': ['амбициозные задачи', 'сложные задачи', 'развитие', 'самореализация', 'инновационные задачи'],
    'Карьерный рост': ['недостаток роста', 'перспективы', 'отсутствие карьерного роста', 'отсутствие признания']
}

data = []
labels = []

for category, words in synonym_groups.items():
    data.extend(words)
    labels.extend([category] * len(words))


model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data, labels)

# Функция для классификации нового ответа
def classify_answer(answer, model):
    return model.predict([answer])[0]

file_path = "./questions_answers.csv"
df = pd.read_csv(file_path)

# Функция для анализа ответов на конкретный вопрос
def analyze_answers(question, answers, model):
    categorized_answers = defaultdict(int)
    
    for answer in answers:
        answer = answer.lower()
        category = classify_answer(answer, model)
        categorized_answers[category] += 1

    df_result = pd.DataFrame(categorized_answers.items(), columns=['Category', 'Frequency'])
    df_result['Question'] = question 
    return df_result

results_db = pd.DataFrame(columns=['Question', 'Category', 'Frequency'])

for question in df['Question'].unique():
    answers = df[df['Question'] == question]['Answer'].tolist()
    result_df = analyze_answers(question, answers, model)
    results_db = pd.concat([results_db, result_df], ignore_index=True)

results_db_grouped = results_db.groupby(['Question', 'Category'], as_index=False).sum()

print(results_db_grouped.to_string(index=False))

output_file_path = "./answers_analysis.csv"
results_db_grouped.to_csv(output_file_path, index=False)

def visualize_question_stats():
    # Список уникальных вопросов
    questions = df['Question'].unique()
    
    # Выводим список вопросов для выбора
    print("\nДоступные вопросы:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    # Запрашиваем у пользователя выбор вопроса
    choice = int(input("\nВведите номер вопроса для отображения статистики: ")) - 1
    
    if 0 <= choice < len(questions):
        selected_question = questions[choice]
        df_plot = results_db_grouped[results_db_grouped['Question'] == selected_question]
        
        # Данные для круговой диаграммы
        labels = df_plot['Category']
        sizes = df_plot['Frequency']
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.axis('equal')  # Для круговой формы диаграммы
        plt.title(f'"{selected_question}"')
        plt.show()
    else:
        print("Неверный выбор. Пожалуйста, попробуйте снова.")

# Вызов функции визуализации
visualize_question_stats()