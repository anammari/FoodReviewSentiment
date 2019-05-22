import pandas as pd
from IPython.display import display
import os
from ipywidgets import widgets
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

text = widgets.Text()
text2 = widgets.Text()
text3 = widgets.Text()
data = None
test = None
model = None

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our custom tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Basic function to clean the text
def clean_text(t):
    # Removing spaces and converting text into lowercase
    return t.strip().lower()

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(t) for t in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def load_data(sender):
	global data
	data = pd.read_csv(os.getcwd()+'/../data/train/'+text.value, sep='\t', engine='python', header=None, encoding='utf-8')
	data.columns = ['review', 'label']
	display(data.head(10))

def input_filename():
	global text
	display(text)
	text.on_submit(load_data)

def load_test(sender):
	global test
	test = pd.read_csv(os.getcwd()+'/../data/prediction/'+text2.value, sep='\t', engine='python', header=None, encoding='utf-8')
	test.columns = ['review']
	display(test.head(10))

def input_test_filename():
	global text2
	display(text2)
	text2.on_submit(load_test)

def load_model(sender):
	global model
	model = joblib.load(os.getcwd()+'/../model/'+text3.value)

def input_model_filename():
	global text3
	display(text3)
	text3.on_submit(load_model)