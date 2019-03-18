# -*- coding: utf-8 -*-
"""
Python 3.7
"""

# -*- coding: utf-8 -*-
"""

"""

from pattern.en import sentiment


entry = "This is so bad"

'''Introduction to the function'''
print(sentiment(entry))


'''Repurposing the function for our needs'''
#print(sentiment(entry)[0])


'''Let's try a few'''

#caption1 = "OMG this is awesome"
#caption2 = "NOOOO, that's so ugly"
#caption3 = "It's alright"
#
#print("'",caption1,"'" , " Has a sentiment of " , sentiment(caption1)[0], " which is Positive")
#print("'",caption2,"'" , " Has a sentiment of " , sentiment(caption2)[0], " which is Negative")
#print("'",caption3,"'" , " Has a sentiment of " , sentiment(caption3)[0], " which is Neutral")


'''Making use of the function to test multiple captions'''
#data = ['This is ok','Comment F to pay respects', 'Could be better', 'Fucking hell', 'OMG', 'Been a cracking two years with these gents', 'Glen Eyre till I die', 'This man is a genius', 'Some studies say that by the end of this century, 80% of insects could be extinct. Their end will also cause our end!!', 'so true and so upsetting' ]
#
#positive = 0
#neutral = 0
#negative = 0
#total = 0
#
#for item in data: 
#    if sentiment(item)[0] > 0:
#        positive += 1
#    elif sentiment(item)[0] < 0:
#        negative += 1
#    else:
#        neutral += 1
#        
#total = positive + neutral + negative
#    
#print("Positve  captions ¦ Instances: ", positive, "¦ Percentage: ", float(positive/total), "%")
#print("Neutral  captions ¦ Instances: ", neutral, "¦ Percentage: ", float(neutral/total), "%")
#print("Negative captions ¦ Instances: ", negative, "¦ Percentage: ", float(negative/total), "%")
