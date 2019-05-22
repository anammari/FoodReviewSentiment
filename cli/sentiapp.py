import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import click

model = None
err = ''

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
    	# Fitting model
        return self

    def get_params(self, deep=True):
    	# Not used
        return {}

class model_serving():
	def __init__(self, modelname, review):
		# Initialize variables
		self.modelname = modelname
		self.review = review
		self.sentiment = None
		self.score = None
		
	def load_model(self):
		# Loading model
		global model
		global err
		try:
			model = joblib.load(os.path.dirname(os.path.abspath(__file__))+'/../model/'+self.modelname+'.sav')
		except FileNotFoundError as fnf_error:
			model = None
			err += str(fnf_error)+'\n'

	def get_sentiment(self):
		# Predict and Score review
		global model
		global err
		if model is not None:
			try:
				self.sentiment = model.predict([self.review])[0]
				self.score = model.predict_proba([self.review])[0][self.sentiment]
			except Exception as e:
				err += str(e)


	def predict(self):
		# Return prediction and score in text message
		global err
		self.load_model()
		self.get_sentiment()
		if len(err) == 0:
			print('Sentiment: {0} - Score: {1}'.format(self.sentiment, self.score))
		else:
			print(err)

@click.command()
@click.argument('review', type=click.STRING)
@click.option('--modelname', '-m', default='model_lr_tfidf', type=click.STRING, 
	help='Name of the model to use for sentiment prediction. One of [model_lr_tfidf (default), model_lr_bow]')
def main(modelname, review):
	"""
	This CLI application predicts the sentiment (0: negative, 1: positive) of food reviews.
	ARGUMENTS: 
	REVIEW: Required. The food review string. Must be in double quotes.
	"""
	ms = model_serving(modelname, review)
	ms.predict()

if __name__ == "__main__": 
	main()
