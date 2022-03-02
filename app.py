from email import message
from flask import Flask, render_template, url_for, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "

    alpha_sent = alpha_sent.strip()
    return alpha_sent

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


def returnboll(num):
    if num == 0:
        return "False"
    return "True"

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message = cleanHtml(message)
        message = cleanPunc(message)
        message = keepAlpha(message)
        message = removeStopWords(message)
        message = stemming(message)
        data = [message]
        data = vectorizer.transform(data)
        print(data.shape)
        my_prediction = model.predict(data)
        arr = my_prediction.toarray()
        res = ''.join(map(str, arr))
        print("prediction done...")
        print(res)
    return render_template('result.html', toxic=returnboll(arr[0][0]), severe_toxic=returnboll(arr[0][1]), obscene=returnboll(arr[0][2]), threat=returnboll(arr[0][3]), insult=returnboll(arr[0][4]), identity_hate=returnboll(arr[0][5]))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
