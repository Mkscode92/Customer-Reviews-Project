import matplotlib.pyplot as plt
import pandas 
import numpy as np

reviews = pandas.read_csv("Reviews.csv")
# Drop irrelevant columns
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

# Plot the data
plt.bar(score_values, counts)
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Distribution of Scores in Reviews")
plt.show() #show the bar graph


