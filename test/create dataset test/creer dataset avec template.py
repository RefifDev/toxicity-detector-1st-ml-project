import random
import pandas as pd

random.seed(42)

# =========================
# TEMPLATES NON TOXIQUES
# =========================

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

# =========================
# TEMPLATES TOXIQUES
# =========================

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

# =========================
# VARIATIONS STYLE TWITCH
# =========================

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

# =========================
# GÉNÉRATION DU DATASET
# =========================

N = 2500        # nombre de messages
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

# =========================
# DATAFRAME + EXPORT
# =========================

df = pd.DataFrame(data, columns=["text", "label"])

# Mélange pour éviter les patterns
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("dataset_twitch_1000.csv", index=False, encoding="utf-8")

print("✅ Dataset généré : dataset_twitch_1000.csv")
print(df["label"].value_counts())
print("Taux toxique :", df["label"].mean() * 100, "%")
