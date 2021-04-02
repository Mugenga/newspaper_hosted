import pickle
import string
import pandas as pd

from flask import Flask, render_template, request
from nltk import pos_tag, WordNetLemmatizer

from nltk.corpus import wordnet, stopwords

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/category", methods=['GET', 'POST'])
def category():
    article = request.form.get('article')
    article = pd.DataFrame({"article": [article]})

    # Clean Data
    content = article['article'].apply(lambda x: clean_text(x))

    # Import Model
    pickle_in = open("newspaper_model.sav", "rb")
    model = pickle.load(pickle_in)

    # Preparing the tokenizer
    tokenizer = Tokenizer(num_words=100)
    encoded_text = tokenizer.texts_to_sequences([content[0]])
    max_length = 2
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')

    # Predict
    y_pred = model.predict(padded_text)
    if y_pred[0] == 0:
        category_ = "Arts, Culture, & Celebrities"
        catInternal = "arts_culture"
    elif y_pred[0] == 1:
        category_ = "Business"
        catInternal = "business"
    elif y_pred[0] == 2:
        category_ = "Politics"
        catInternal = "politics"
    else:
        category_ = "Sports"
        catInternal = "sports"
    print(y_pred)

    df = pd.read_csv("articles.csv")
    df = df.loc[df['category'] == 'business']
    df = df.values.tolist()

    return render_template("category.html", category_=category_, df=df)


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text


if __name__ == '__main__':
    app.run(host='0.0.0.0')
