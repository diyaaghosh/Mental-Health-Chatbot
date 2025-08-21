import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    tokenized_words = [stemming(w) for w in tokenized_sentence]
    stemmed_words = [stemming(w) for w in words]
    for idx, w in enumerate(stemmed_words):
        if w in tokenized_words:
            bag[idx] = 1
    return bag


