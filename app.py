#install required packages using pip or conda then execute 

# packages imports
import pandas as pd
import re
# natural language toolkit packages
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# saving model
import pickle


# to remove the morphological affixes from words eg: used->use (stemming)
def stem_text(text):
    stemmer = PorterStemmer()
    return [" ".join([stemmer.stem(w) for w in text.lower().split()])]

# to remove punctuation marks and to reduce the words to their root words (lemmatizing)
def clean_and_remove_punctuations(text):
    lemmatizer = WordNetLemmatizer() 
    row = re.sub(r"[^a-zA-Z0-9]+", ' ', text).lower().split()
    lst = " ".join([lemmatizer.lemmatize(x) for x in row])
    return lst

 
# model_lr_v1 is the name of the logistic regression model
filename = r"C:\Users\Aishwarya\PROJECTS\NLP\main\V1\models\model_lr_v1" # put your file location here
model_lr_v1 = pickle.load(open(filename, 'rb'))

# tfidf_vectorizer.pk is the name of the vectorizer
vect = pd.read_pickle(r"C:\Users\Aishwarya\PROJECTS\NLP\main\V1\models\tfidf_vectorizer.pk")

# place the word you want to find the sentiment in text
text = "awesome"
clean = clean_and_remove_punctuations(text)
print(clean)

# result returns 1 if the word is positive else it returns 0
result = model_lr_v1.predict(vect.transform(stem_text(clean)))
print(result)
