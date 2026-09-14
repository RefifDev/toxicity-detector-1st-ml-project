from datasets import load_dataset
import pandas as pd

print("Téléchargement du dataset français...")

# Charger le dataset
dataset = load_dataset("textdetox/multilingual_toxicity_dataset")

# Accéder au français
french_data = dataset['fr']

print(f" Messages français téléchargés : {len(french_data)}")

# Convertir en DataFrame pandas
df_french = pd.DataFrame(french_data)

# Renommer 'toxic' en 'label' pour être compatible avec ton code
df_french = df_french.rename(columns={'toxic': 'label'})

print(f"Toxiques : {sum(df_french['label'] == 1)}")
print(f"Non-toxiques : {sum(df_french['label'] == 0)}")

# Afficher quelques exemples
print("\n Exemples de messages toxiques :")
print(df_french[df_french['label'] == 1].head(3)['text'].values)

print("\n Exemples de messages non-toxiques :")
print(df_french[df_french['label'] == 0].head(3)['text'].values)

# Garder seulement text et label
df_french = df_french[['text', 'label']]

import random
import pandas as pd

random.seed(42)

# TEMPLATES NON TOXIQUES
non_toxic_templates = [
    "Salut le chat",
    "Salut tout le monde",
    "Yo",
    "Bonsoir à tous",
    "Super stream aujourd'hui",
    "GG bien joué",
    "GG",
    "Merci pour le contenu",
    "Merci pour le stream",
    "J'adore ce jeu",
    "Ce jeu est incroyable",
    "Bonne chance pour la suite",
    "Très bonne ambiance ici",
    "Incroyable ce clutch",
    "Quel est ton setup ?",
    "Tu joues depuis combien de temps ?",
    "C'était une belle game",
    "Bien joué le move",
    "Continue comme ça",
    "Respect pour le niveau",
    "Ça fait plaisir à regarder",
    "Trop clean",
    "Masterclass",
    "T'es chaud aujourd'hui",
    "C'est trop bien ce que tu fais",
    "Franchement respect",
    "Force à toi",
    "Tu gères",
    "Ça régale",
    "Bien vu",
    "Propre",
    "Nice play",
    "Let's goooo",
    "Quel clutch",
    "Ça clutch fort",
    "Trop fort",
    "J'aime bien ton style de jeu",
    "La strat est bonne",
    "Ça joue bien",
    "Continue le grind",
    "Hâte de voir la suite",
    "Bon courage",
    "Bonne soirée le chat",
]

# TEMPLATES TOXIQUES
toxic_templates = [
    "T'es nul",
    "T'es vraiment nul",
    "T'es éclaté",
    "Ferme ta gueule",
    "Ferme-la",
    "Quel stream de merde",
    "C'est nul à chier",
    "T'es un gros con",
    "Retourne apprendre à jouer",
    "Apprends à jouer",
    "Personne t'aime ici",
    "Dégage de Twitch",
    "T'es pathétique",
    "Arrête de stream",
    "Arrête le stream",
    "T'es ridicule",
    "On s'ennuie à mort",
    "Va te faire foutre",
    "T'as aucun talent",
    "C'est catastrophique",
    "T'es gênant",
    "Quel malaise",
    "Ça fait pitié",
    "T'es une honte",
    "T'es mauvais",
    "Niveau zéro",
    "C'est insupportable",
    "Tu fais n'importe quoi",
    "Mais joue correctement",
    "T'as un niveau éclaté",
    "Supprime ta chaîne",
    "Pourquoi tu streams",
    "C'est de la merde",
    "Quelle purge",
    "J'en peux plus de te voir",
    "Retourne t'entraîner",
    "Change de jeu t'es nul",
    "Même ma grand-mère joue mieux",
    "T'es à la ramasse",
]

prefixes = [
    "",
    "mdr ",
    "ptdr ",
    "franchement ",
    "wesh ",
    "eh ",
    "non mais ",
    "lol ",
]

suffixes = [
    "",
    " lol",
    " mdr",
    " ptdr",
    " sérieux",
    " 😭",
    " 💀",
    " 🤡",
]

# GÉNÉRATION DU DATASET

N = 2500           # nombre de messages
ratio_toxic = 0.5  # niveau de toxicité

data = []

for i in range(N):
    if random.random() < ratio_toxic:
        msg = random.choice(toxic_templates)
        label = 1
    else:
        msg = random.choice(non_toxic_templates)
        label = 0

    msg = random.choice(prefixes) + msg + random.choice(suffixes)
    data.append((msg.strip(), label))

# DATAFRAME + EXPORT
df_generated = pd.DataFrame(data, columns=["text", "label"])

# Mélange pour éviter les patterns
df = df_generated.sample(frac=1).reset_index(drop=True)

# mix des deux generation de dataset
df_hybrid = pd.concat([df_french, df_generated], ignore_index=True)

df_hybrid = df_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n Dataset hybride créé : {len(df_hybrid)} messages")
print(f"  Toxiques : {sum(df_hybrid['label'] == 1)} ({sum(df_hybrid['label'] == 1)/len(df_hybrid)*100:.1f}%)")
print(f"  Non-toxiques : {sum(df_hybrid['label'] == 0)} ({sum(df_hybrid['label'] == 0)/len(df_hybrid)*100:.1f}%)")

# Sauvegarder en CSV
df_hybrid.to_csv('data/dataset3.csv', index=False, encoding='utf-8')
print("\n Dataset français sauvegardé : dataset3.csv")