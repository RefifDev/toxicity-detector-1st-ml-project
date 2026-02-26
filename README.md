# toxicity-detector-1st-ml-project
Machine learning web app to detect toxic messages in online chats. Built with Python, Flask and Naive Bayes - work in progress.
# Toxicity Detector

> **Work in progress** - functional but not finalized

Automatic detection tool for toxic messages in online chats, developed as a personal project to learn Machine Learning and Data Science.

---

## Objective

Build a system capable of analyzing text messages and detecting toxic content (insults, harassment, hate speech) to help moderate online platforms, with a focus on gaming and Twitch chats.

---

## Current state

| Feature | Status |
|---|---|
| Trained Naive Bayes model | Done |
| TF-IDF vectorization | Done |
| Flask web interface | Done |
| Message analysis page | Done |
| About page | Done |
| Database history (SQLite) | In progress |
| Twitch API integration | Planned |
| Custom dataset upload | Planned |
| Visual statistics | Planned |

---

## ML Model

- **Dataset**: 7,500 hybrid messages (5,000 real via Hugging Face TextDetox + 2,500 generated)
- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (vocabulary of 11,695 words)
- **Accuracy**: 92.4% on test data / 81.25% on real manual test data

---

## Tech stack

**Backend & ML**
- Python 3.12
- Flask
- scikit-learn (Naive Bayes, TF-IDF)
- pandas, joblib
- Hugging Face Datasets

**Frontend**
- HTML & CSS
- Jinja2 (Flask templates)
- Font Awesome

---

## Installation

```bash
# Clone the repository
https://github.com/RefifDev/toxicity-detector

# Install dependencies
pip install flask scikit-learn pandas joblib

# Train the model (if needed)
python train_ia.py

# Run the app
python app.py
```

The app will be available at `http://localhost:5000`

---

## Project structure

```
toxicity-detector/
├── app.py                  # Main Flask application
├── model_utils.py          # ML functions (load, predict, train)
├── train_ia.py             # Model training script
├── test_systeme.py         # Prediction tests
├── models/
│   ├── model.pkl           # Trained model
│   └── vectorizer.pkl      # TF-IDF vectorizer
├── data/
│   └── dataset2.csv        # Training dataset
├── html/
│   ├── index.html
│   ├── test.html
│   └── about.html
└── css/
    └── style.css
```

---

## Known limitations

- The model analyzes each message in isolation, with no conversational context
- Struggles with sarcasm and irony
- Optimized for French gaming/Twitch content - performance may vary on other platforms
- Limited dataset (7,500 messages) - may miss rare slang or new expressions

---

## Roadmap

- [ ] Twitch API integration for real-time analysis
- [ ] Analysis history with SQLite
- [ ] CSV file upload for batch analysis
- [ ] Statistics dashboard
- [ ] Multilingual support

---

## Author

**refif ilan** - Personal ML/Data Science learning project (2025-2026)

Developed with Claude (Anthropic) as a learning tool - every concept was studied and coded independently.
