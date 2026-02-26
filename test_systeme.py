from model_utils import load_model, predict_message

print("="*50)
print("TEST DE PRÉDICTIONS")
print("="*50)

# Charger le modèle sauvegardé
model, vectorizer = load_model()

# Messages à tester
messages_test = [
    "Merci pour le stream, c'était cool",
    "T'es vraiment nul va désinstaller",
    "GG bien joué",
    "Arrête de streamer personne te regarde",
    "Super gameplay aujourd'hui"
]

for msg in messages_test:
    pred, proba = predict_message(msg, model, vectorizer)
    label = "TOXIQUE" if pred == 1 else "OK"
    confiance = proba[pred] * 100
    print(f"\n'{msg}'")
    print(f"  → {label} (confiance: {confiance:.1f}%)")

print("\nTests terminés !")