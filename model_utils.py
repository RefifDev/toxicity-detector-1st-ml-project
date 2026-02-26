import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

def save_model(model, vectorizer, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
    """Sauvegarde le modèle et le vectorizer"""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print('Modèle sauvegardé avec succès !')

def load_model(model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
    """Charge le modèle et le vectorizer"""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print('Modèle chargé avec succès !')
    return model, vectorizer

def predict_message(text, model, vectorizer):
    """Prédit si un message est toxique"""
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    prob_predict = model.predict_proba(text_vec)
    return prediction[0], prob_predict[0]

def prepare_data(filepath):
    """Charge et prépare les données"""
    data = pd.read_csv(filepath)
    print(f"Dataset chargé : {len(data)} messages")
    
    x = data['text']
    y = data['label']
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Vocabulaire: {len(vectorizer.vocabulary_)} mots")
    
    return x_train_vec, x_test_vec, y_train, y_test, vectorizer

def train_model(x_train_vec, y_train):
    """Entraîne le modèle"""
    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    print("Modèle entraîné !")
    return model