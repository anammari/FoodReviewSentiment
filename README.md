# FoodReviewSentiment

A simple CLI application that uses NLP and Machine Learning to predict the sentiment of food reviews.

## Requirements

- Python 3.7.3
- SpaCy 2.1.3
- Scikit-Learn 0.20.3

## Running the application
- The easiest way to run the application is to use the `jupyter/datascience-notebook` Docker image:

```console
cd FoodReviewSentiment
docker run -v ${PWD}:/home/jovyan/work -it --rm -p 8888:8888 --name datascience-notebook jupyter/datascience-notebook
```

- Once you are in the `datascience-notebook` container, install SpaCy and download the English-language model:

```console
docker exec -it datascience-notebook /bin/bash
pip install spacy
python -m spacy download en
```

- To read how to use the app:

```console
python work/cli/sentiapp.py --help
Usage: sentiapp.py [OPTIONS] REVIEW

  This CLI application predicts the sentiment (0: negative, 1: positive) of
  food reviews. ARGUMENTS:  REVIEW: Required. The food review string. Must
  be in double quotes.

Options:
  -m, --modelname TEXT  Name of the model to use for sentiment prediction. One
                        of [model_lr_tfidf (default), model_lr_bow]
  --help                Show this message and exit.
```

- To run the app, supply a review string (required) and a model name (optional):

```console
python work/cli/sentiapp.py review="this is a great restaurant with amazing food"
Sentiment: 1 - Score: 0.838837431121674

python work/cli/sentiapp.py review="this is a horrible restaurant! the food sucks" -m model_lr_bow
Sentiment: 0 - Score: 0.7953132749299411
```