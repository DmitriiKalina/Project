import pandas as pd
import random

questions_answers = {
    'Что мотивирует вас работать больше?': ['команда', 'коллеги', 'зарплата', 'бабосики', 'шеф', 'атмосфера', 'амбициозные задачи', 'премии', 'бонусы', 'поддержка'],
    'Чем интересна ваша работа?': ['коллектив', 'инновационные задачи', 'развитие', 'атмосфера', 'амбициозные задачи', 'коллеги', 'команда', 'климат', 'удовольствие от работы'],
    'Какие факторы влияют на ваше решение об уходе?': ['зарплата', 'руководство', 'недостаток роста', 'коллеги', 'график', 'перспективы', 'отсутствие признания', 'начальство', 'отсутствие карьерного роста']
}

data = []
for _ in range(1000):
    question = random.choice(list(questions_answers.keys()))
    answer = random.choice(questions_answers[question])
    data.append([question, answer])

df = pd.DataFrame(data, columns=['Question', 'Answer'])
file_path = "./questions_answers.csv"
df.to_csv(file_path, index=False)

