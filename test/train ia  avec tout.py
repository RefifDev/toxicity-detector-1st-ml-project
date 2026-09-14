import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def prepare_data(filepath):
    '''
        Charge et prépare les données pour l'entraînement d'un modèle de classification de toxicité.
        
        Args:
            filepath (str): Chemin vers le fichier CSV contenant les données
            
        Returns:
            tuple: (x_train_vec, x_test_vec, y_train, y_test, vectorizer)
                - x_train_vec: Messages d'entraînement vectorisés
                - x_test_vec: Messages de test vectorisés  
                - y_train: Labels d'entraînement
                - y_test: Labels de test
                - vectorizer: CountVectorizer entraîné
    '''

    data = pd.read_csv(filepath)
    print("success !")

    x = data['text']
    y = data['label']

    print(f'taille de x :{len(x)} et y :{len(y)}') 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    ## print(f'taille de x_train :{len(x_train)} et y_train :{len(y_train)}')
    ## print(f'taille de x_test :{len(x_test)} et y_test :{len(y_test)}')

    
    vectorizer = TfidfVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    ## print(f'shape du train de x : {x_train_vectorized.shape}')
    ## print(f'shape du test de x : {x_test_vectorized.shape}')

    
    return x_train_vectorized, x_test_vectorized, y_train, y_test, vectorizer, x_test, y_test


x_train_vec, x_test_vec, y_train, y_test, vectorizer, x_test, y_test = prepare_data("dataset2.csv")

print(f"\n=== Résumé ===")
print(f"Train shape: {x_train_vec.shape}")
print(f"Test shape: {x_test_vec.shape}")
print(f"Vocabulaire: {len(vectorizer.vocabulary_)} mots uniques")
# Dans prepare_data(), ajoute avant le return :
print(f"\n=== Stats du dataset ===")
print(f"Total train: {len(y_train)}")
print(f"Toxiques: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
print(f"Non-toxiques: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")


def train_model(x_train, y_train):
    """
        Entraîne un modèle de régression logistique
    """
    data2 = pd.read_csv("../data/dataset2.csv")
    model = MultinomialNB() ## max_iter=1000 : limite d'iteration de securité
    model.fit(x_train, y_train)
    # Après ton entraînement sur les données générées
    test_df = pd.DataFrame(data2, columns=['text', 'label'])
    x_manuel = test_df['text']
    y_manuel = test_df['label']

    x_manuel_vec = vectorizer.transform(x_manuel)
    acc_manuel = model.score(x_manuel_vec, y_manuel)

    print(f"\n🎯 VRAI TEST - Accuracy sur données manuelles : {acc_manuel * 100:.2f}%")

    # Affiche les erreurs
    predictions = model.predict(x_manuel_vec)
    for i, (text, pred, vrai) in enumerate(zip(x_manuel, predictions, y_manuel)):
        if pred != vrai:
            print(f"❌ Erreur {i+1}: '{text}' → Prédit {pred}, Vrai {vrai}")
        ## print(f"Nombre d'itérations utilisées : {model.n_iter_}")
        print('Success !')
    print(f"\n🎯 VRAI TEST - Accuracy sur données manuelles : {acc_manuel * 100:.2f}%")
    return model
        
m = train_model(x_train_vec, y_train)

def evaluate_model(model, x_test_vec, x_test_text, y_test):
    # Prédictions sur la version VECTORISÉE
    predictions = model.predict(x_test_vec)  # <-- Nombres
    acc = model.score(x_test_vec, y_test)     # <-- Nombres
    
    print(f"Prédictions : {predictions}")
    print(f"Vrais labels : {y_test.values}")
    print(f"Accuracy : {acc * 100:.2f}%")
    
    '''print("\n=== Comparaison ===")
    for i in range(len(predictions)):
        pred = predictions[i]
        vrai = y_test.iloc[i]
        message = x_test_text.iloc[i]  # <-- Texte pour l'affichage
        resultat = "✓" if pred == vrai else "X"
        
        print(f"Message {i+1}: Prédit={pred}, Vrai={vrai} {resultat}")
        print(f"  --> '{message}'")'''
    
    return acc

evaluate_model(m, x_test_vec, x_test, y_test)

def save_model(model, vectorizer, model_path='../models/model.pkl', vectorizer_path='../models/vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print('Save effectuée avec succes !')

def load_model(model_path='../models/model.pkl', vectorizer_path='../models/vectorizer.pkl'):
    m = joblib.load(model_path)
    v = joblib.load(vectorizer_path)
    print('Load effectué avec succes !')
    return m, v

def predict_message(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    prob_predict = model.predict_proba(text_vec)
    return prediction, prob_predict
        

save_model(m, vectorizer)
print(load_model())