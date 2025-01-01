import matplotlib.pyplot as plt
import pandas 
import numpy as np

reviews = pandas.read_csv("Reviews.csv")
# Drop irrelevant columns, data cleaning 
reviews = reviews.drop(columns=["ProductId","UserId","ProfileName"])
#print(reviews.head(len(reviews)))  - this is optional, it just prints out the contents of the csv file but its a bit long so
# unless I want it seem like a nuke hit the program and made the program crash after taking so long, this shouldnt be executed 

length = len(reviews) #gets the length of the csv file 
score_counts = {} #this is a hashmap (aka dictionary), which will basically tell us how many times each score (1-5) is repeated
for i in range(length):
    score = int(reviews["Score"][i]) #making the score value an int, because csv values are not naturally ints they're like np.64s or something
    score_counts[score] = 1 + score_counts.get(score, 0) #add 1 to a specific score value if it pops up 

score_counts = dict(sorted(score_counts.items()))  #sorts the values in numerical order because or else it'll be: {5: 363122, 1: 52268, 4: 80655, 2: 29769, 3: 42640}

print("Count of how many times each score value comes up in the csv file: ", score_counts)


score_values = list(score_counts.keys())  
counts = list(score_counts.values())    

#Plot the data
plt.bar(score_values, counts)
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Distribution of Scores in Reviews")
plt.show()

#natural language processing - the concept of training a model on human language to be able to make 
# the machine comprehend and interpret language
#Roberta model 
import seaborn as sns #built on top of matplotlib, offers more aesthetically pleasing default styles and color palettes
plt.style.use("ggplot")

import nltk 
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

print(reviews.shape)
reviews = reviews.head(500)
print(reviews.shape)

sentence = input("Enter text: ") #google translate to handle different languages besides english 
from googletrans import Translator, constants
from pprint import pprint
translator = Translator()
translation = translator.translate(sentence)
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
translation = translation.text
# reviews["Score"].value_counts().sort_index().plot(kind = "bar", title = "Distribution of Scores in Reviews")
#basic NLTK stuff
example = translation #reviews["Text"][50]
print("50:", example)
tokens = nltk.word_tokenize(example)
print(tokens) #bit smarter than normal splitting, can also split the "don't" word for example 
# tokens[:10]

tagged = nltk.pos_tag(tokens)
print(tagged) #tells you what part of speech each word is from, like adjective, etc
#nltk.chunk.ne_chunk(tagged) -> groups these words into categories based on those parts of speech 


"""
Token Normalization

Stemming: A process of removing and replacing suffixes to get to the root form of the word, which is called stem.
Lemmatization: returns the base or dictionary form of a word.
"""

# *******************On to sentiment analysis********************
# VADER (Valence Aware Dictionary and sentiment Reasoner)
# VADER is a rule-based sentiment analysis tool designed specifically for analyzing the sentiment of social media text.
# It has simple rules for determining the intensity of positive, negative, and neutral sentiment.

# RoBERTa (Robustly Optimized BERT) RoBERTa is a deep learning model based on the BERT architecture 
# (Bidirectional Encoder Representations from Transformers) but optimized for better performance through changes
# like training with more data and removing the next sentence prediction task. 

#Vader = faster, but less accurate, RoBERTa = slower, but more accurate (BETTER OPTION) - better for context (sarcasm for example) 

from transformers import AutoTokenizer #pretrained tokenizer for handling text, automatically selects the appropriate tokenizer for a given pre-trained model
from transformers import AutoModelForSequenceClassification #used to load a sentiment analysis model pretrained on a multilingual dataset 
from scipy.special import softmax
import csv
import urllib.request

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL) #splits into pieces 
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print(example)
#Run for roberta model 
# The return_tensors="pt" parameter is used in Hugging Face Transformers when tokenizing text.
# It specifies the format in which the tokenized data should be returned, 
# and in this case, it ensures that the data is returned as PyTorch tensors.
# When you tokenize text using a tokenizer, the raw text is converted into token IDs
# (numbers that represent tokens in the model's vocabulary). These token IDs can be returned in
# different formats, depending on the library or framework you are using.

# "pt": Return the tokenized output as PyTorch tensors.
# Other options:
# "tf": TensorFlow tensors.
# "np": NumPy arrays.
# None: A standard Python dictionary of lists.
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

encoded_text = tokenizer(example, return_tensors = "pt") #changes it to 1s and 0s, in pytorch format 
output = model(**encoded_text) #run model on encoded text 
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
print(translation)
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
