# About the Project

## Project Overview

The **Toxicity Detector** is an automated system for detecting toxic messages in online chats.

This personal project was developed as part of my learning journey in Machine Learning and Data Science, to introduce myself to the world of AI and data analysis for the studies I plan to pursue.

The goal is to create a tool capable of automatically analyzing messages and detecting toxic content (insults, harassment, hate speech) in order to facilitate moderation on online platforms.

---

## Technical Implementation

### Hybrid Dataset

The model was trained on a hybrid dataset of 7,500 messages:

- 5,000 real messages from the French Hugging Face dataset (TextDetox)
- 2,500 messages generated using a custom template system including linguistic variations (prefixes/suffixes) typical of gaming and Twitch language

### TF-IDF Vectorization

Text messages are transformed into numerical vectors using TF-IDF (Term Frequency – Inverse Document Frequency), a technique that weights words based on their importance and rarity. The learned vocabulary contains 11,695 unique words.

### Classification Model

The algorithm used is Multinomial Naive Bayes, a probabilistic classifier particularly effective for text classification. The model calculates the probability that a message is toxic or non-toxic by analyzing word patterns.

### Performance

- 92.4% accuracy on generated test data
- 81.25% accuracy on real manual data (generalization test)
- Convergence in only 7–8 iterations during training

---

## Technologies Used

### Backend & Machine Learning

- Python 3.12 – Main programming language
- Flask – Web framework for the API and interface
- scikit-learn – ML library (Naive Bayes, TF-IDF, train/test split)
- pandas – Dataset manipulation
- joblib – Saving/loading the trained model
- Hugging Face Datasets – Importing the French dataset

### Frontend

- HTML & CSS – Modern and responsive user interface
- Jinja2 – Flask template engine
- Font Awesome – Icons

### Development Tools

- VS Code – Code editor
- GitHub – Version control
- DB Browser for SQLite – Database management (planned)

---

## Identified Limitations

### Lack of Conversational Context

The model analyzes each message in isolation without considering:

- The tone of the conversation (joke between friends vs. attack)
- Message history
- The relationship between speakers

*Example: "You're so bad lol" between friends will be classified as toxic even though it's a joke.*

### Difficulty with Sarcasm and Irony

Sarcastic or ironic expressions are difficult to detect because they require deep semantic understanding that Bag-of-Words models do not possess.

### Platform Specificity

The model was trained mainly on French gaming/Twitch content. Performance may vary on:

- Twitter (different vocabulary)
- Forums (more formal writing style)
- Other languages

### Limited Dataset

With 7,500 training messages, the model may miss rare formulations or new slang expressions. A dataset of 50k–100k messages would significantly improve performance.

### False Positives and False Negatives

On manual test data, the model shows:

- 11 false positives: normal messages classified as toxic (e.g., "How do I complete this quest?")
- 4 false negatives: toxic messages not detected (e.g., "So trash", "Dirty noob")

---

## Improvement Roadmap

### Short Term

- [ ] Twitch API integration – Real-time chat analysis
- [ ] Database history – Saving analyses with SQLite
- [ ] Custom dataset upload – Analyze multiple messages at once
- [ ] Visual statistics – Charts showing toxic/non-toxic distribution

### Medium Term

- [ ] Contextual models – Using BERT or CamemBERT (transformers)
- [ ] Multi-label detection – Categorize by type (insult, harassment, spam, etc.)
- [ ] Extended dataset – Increase to 50k–100k messages via ethical scraping
- [ ] Platform fine-tuning – Twitch vs Twitter vs Discord specific models

### Long Term

- [ ] Production deployment – Hosting on Render.com with a public REST API
- [ ] Feedback system – Continuous learning through user feedback
- [ ] Multilingual extension – Support for English, Spanish, German
- [ ] Discord/Twitch bot integration – Real-time automatic moderation

---

## Skills Developed

This project allowed me to develop comprehensive skills in:

- Machine Learning – Full pipeline (data prep, vectorization, training, evaluation)
- NLP – Natural language processing, TF-IDF, text classification
- Full-stack web development – Flask backend + HTML/CSS frontend
- Software architecture – Modular, reusable, and maintainable code
- Critical analysis – Understanding ML limitations and overfitting
- Project management – Planning, iteration, documentation

---

## Learning Methodology

### Using AI as a Learning Tool

The development of this project was supported by Claude (Anthropic), a conversational AI used as:

- Personal tutor – Explaining Machine Learning, NLP, and web development concepts
- Technical guide – Helping debug and understand errors
- Optimization tool – Suggesting architecture and best practices
- Learning facilitator – Adapting the pace to my knowledge level

### Educational Approach

The use of Claude followed an active learning methodology:

- No copy-paste – Every concept was explained and then coded by myself
- Continuous questioning – Quizzes at the beginning of each session to reinforce knowledge
- Independent problem-solving – Personal research before asking for help
- Deep understanding – Focus on the "why" rather than just the "how"

### Added Value

This approach allowed me to:

- Learn Machine Learning from scratch in 2–3 months
- Understand core concepts (overfitting, generalization, vectorization)
- Develop a critical perspective on ML limitations
- Gain technical autonomy to continue learning independently

### Transparency and Ethics

I consider AI as a tool that amplifies human capabilities, not a substitute for learning. All the code in this project was written and understood by me, with Claude acting as a virtual mentor guiding my learning in a field that was initially unknown to me.
