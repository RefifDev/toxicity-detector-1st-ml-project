from flask import Flask, render_template, request
from model_utils import load_model, predict_message

app = Flask(__name__, 
            template_folder='html',
            static_folder='css')

model, vectorizer = load_model()
print("✅ Modèle chargé au démarrage !")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction, proba = predict_message(message, model, vectorizer)
    
    label = "TOXIQUE" if prediction == 1 else "OK"
    confiance = proba[prediction] * 100
    
    resultat = f"Message: '{message}' → {label} (Confiance: {confiance:.1f}%)"
    
    return render_template('test.html', resultat=resultat)
  
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 