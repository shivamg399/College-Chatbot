
from flask import Flask, render_template, request, jsonify
import random, json, pickle, numpy as np, os, openai, fitz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow import load_model
from textblob import TextBlob
from datetime import datetime

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# 🔑 Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Load training data
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")
pdf_text = ""

# 📄 Read PDF content
def extract_pdf_content(filepath):
    doc = fitz.open(filepath)
    return "".join([page.get_text() for page in doc])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global pdf_text
    file = request.files["file"]
    if file:
        path = os.path.join("pdf_folder", file.filename)
        file.save(path)
        pdf_text = extract_pdf_content(path)
        return "✅ PDF uploaded."
    return "❌ No file uploaded."

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    corrected_msg = str(TextBlob(msg).correct())
    result = predict_class(corrected_msg)
    response = get_response(result, intents, corrected_msg)
    log_chat(msg, response)
    return jsonify({"response": response})

# NLP Preprocessing
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(res)]
    results = sorted(results, key=lambda x: float(x['probability']), reverse=True)
    return results if results and float(results[0]['probability']) >= 0.7 else []

# Response generator
def get_response(intents_list, intents_json, user_input):
    try:
        if intents_list:
            tag = intents_list[0]['intent']
            for intent in intents_json['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])

        prompt = (
            "You are a helpful assistant for a college chatbot. Respond only in English.\n"
            + (f"PDF content:\n{pdf_text[:1500]}\n" if pdf_text else "")
            + f"User said: {user_input}\nReply clearly."
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("GPT Error:", e)
        return "I'm sorry, I didn't understand that."

# Save to chat log
def log_chat(user_msg, bot_msg):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}]\nUser: {user_msg}\nBot: {bot_msg}\n")

if __name__ == "__main__":
    print("✅ Running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
