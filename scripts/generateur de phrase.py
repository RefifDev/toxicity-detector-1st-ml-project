import random
from faker import Faker

fake = Faker()
phrases = []

for _ in range(1000):
    phrase = fake.sentence().lower()
    phrases.append(phrase)

# Enregistrement du fichier
with open('dataset.txt', 'w') as file:
    for phrase in phrases:
        file.write(phrase + '\n')