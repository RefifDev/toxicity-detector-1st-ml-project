import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("toxic_messages.csv")

print("success !")

x = data['text']
y = data['label']

print(f'taille de x :{len(x)} et y :{len(y)}') 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'taille de x_train :{len(x_train)} et y_train :{len(y_train)}')
print(f'taille de x_test :{len(x_test)} et y_test :{len(y_test)}')

def fct_vectorizer(x_train_vectorized, x_test_vectorized):
    vectorizer = CountVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train_vectorized)
    x_test_vectorized = vectorizer.transform(x_test_vectorized)

    print(f'shape du train de x : {x_train_vectorized.shape}')
    print(f'shape du test de x : {x_test_vectorized.shape}')

print(fct_vectorizer(x_train, x_test))