---

#  Mental Health Chatbot 

This is a simple yet powerful *Mental Health Chatbot* developed using *PyTorch* and *Natural Language Processing (NLP)*. It is designed to recognize user intent from text input and provide supportive, empathetic responses to promote mental well-being.

---

##  Features

- Tokenization and stemming of user input.
- Bag-of-Words vectorization of text.
- Feedforward Neural Network (FFNN) for intent classification.
- Predefined supportive responses for each intent.
- Easily extendable intents.json file to add new categories.

---

##  Architecture Overview

### 1. *NLP Pipeline*
The preprocessing steps convert raw input text into a numerical format usable by the neural network.

Input Sentence: "Is anyone there?"

1. Tokenize:         ["Is", "anyone", "there", "?"]


2. Lowercase + Stem: ["is", "anyon", "there", "?"]


3. Remove Punctuation: ["is", "anyon", "there"]


4. Bag-of-Words:     [0, 0, 1, 0, 1, 0, 1]  



### 2. *Feedforward Neural Network (FFNN)*

- *Input*: Bag-of-Words vector.
- *Hidden Layers*: Fully connected layers with activation functions (e.g., ReLU).
- *Output*: Softmax layer for intent prediction.

Example: Input: "Hello?" → BoW: [0, 0, 0, 1, 0] → FFNN Output: [0.01, 0.02, 0.96, 0.01] → Predicted Intent: "greeting"

---

##  Project Structure

├── intents.json       
├── chatbot.py         
├── train.py          
├── model.py           
├── nltk_utils.py       
├── data.pth            
├── README.md          

---

##  Getting Started

###  Requirements

Install dependencies:

```bash
pip install torch nltk

Download NLTK tokenizer models:

import nltk
nltk.download('punkt')
```

---

 ### Training the Model

- Run the following to train the chatbot:

- python train.py

- This trains the model and saves it as data.pth.


---

### Chatting with the Bot

- After training, start chatting with the bot:

- python chat.py

- You'll be prompted to type a message, and the bot will respond accordingly.


---

### Dataset Format (intents.json)
```bash
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey?", "What's up?"],
      "responses": [
        "Hello there! How are you feeling today?",
         "Hi! I am your wellness buddy. What's your mood right now?"
      ]
    },
    {
      "tag": "anxiety",
      "patterns": ["I'm feeling anxious", "I can't calm down", "panic attack"],
      "responses": [
       "When anxiety takes over, it helps to remind yourself that you have come through tough moments before. You are stronger than this feeling. Would you like a short motivational speech, a calm soundscape, or a gentle mental reset exercise?",
        "It sounds like your thoughts are racing. That can feel scary, but it is manageable. Try to focus on your breathing for just a minute. I can play something soothing or guide you through a calm-down method if you would prefer."
  ]
      ]
    }

  ]
}

```
---

### Future Improvements

- Add Named Entity Recognition (NER) or Sentiment Analysis for better contextual understanding.

- Deploy the chatbot with a web interface using Streamlit.

- Enable logging of conversations (with consent) for analysis.



---

### Known Issues

- The chatbot may not handle out-of-scope questions effectively.

- Responses are limited to those defined in intents.json.



---

### License

- This project is licensed under the MIT License.


---




