# nltk_utils.py
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources safely
nltk.download('punkt', quiet=True)  # only 'punkt' is needed

# Initialize stemmer
stemmer = PorterStemmer()

# Tokenize sentences (breaking sentence into words)
def tokenize(sentence):
    return word_tokenize(sentence)

# Stemming (converting word to its root form)
def stemming(word):
    return stemmer.stem(word.lower())

# Bag of words
# Returns an array where each position is 0 or 1
# 1 if corresponding word in 'words' exists in 'tokenized_sentence'
def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)

    # Stem each word in the sentence
    tokenized_words = [stemming(w) for w in tokenized_sentence]

    # Stem the reference words
    stemmed_words = [stemming(w) for w in words]

    # Set bag[i] = 1 if word exists in sentence
    for idx, w in enumerate(stemmed_words):
        if w in tokenized_words:
            bag[idx] = 1

    return bag


