from model_utils import prepare_data, train_model, save_model

print("="*50)
print("ENTRAÎNEMENT DU MODÈLE")
print("="*50)

# Préparer les données
x_train_vec, x_test_vec, y_train, y_test, vectorizer = prepare_data("data/dataset2.csv")

# Entraîner
model = train_model(x_train_vec, y_train)

# Évaluer
accuracy = model.score(x_test_vec, y_test)
print(f"Accuracy sur test: {accuracy*100:.2f}%")

# Sauvegarder
save_model(model, vectorizer)

print("\n✅ Modèle sauvegardé et prêt à l'emploi !")