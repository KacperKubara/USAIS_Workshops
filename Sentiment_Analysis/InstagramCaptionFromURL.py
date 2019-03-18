# -*- coding: utf-8 -*-
"""
Instragram Captions from the URL using, BeautifulSoup library
"""

import requests
from bs4 import BeautifulSoup

import re

'''Whatever URL you want (but has to be a public, not private, account)'''

url = ('https://www.instagram.com/p/BsnyFfxHBj7/')
    
response = requests.get(url)


'''Basically copies the html code of the entire page into our little soup bowl'''

soup = BeautifulSoup(response.text, 'html.parser')


'''Scan through the whole text and find anything with the <script> <\script> brackets'''

scripts = soup.find_all("script")


'''For each instance of <script> we check for the below keywords'''

for i in range(len(scripts)):
    if "edge_media_to_caption" in scripts[i].text:
        keyscript = scripts[i].text
        
        
    
    '''START: "edge_media_to_caption":{"edges":[{"node":{"text":"
       END:  "}}]},"caption_is_edited"                                 '''
    
'''Within the right script we then extract the caption from inbtween the START & END points'''

try:
    found = re.search('{"node":{"text":"(.+?)"}}]},"caption_is_edited"',keyscript).group(1)
except AttributeError:
    found = ''    
    
    
'''1'''
#print(found)    

'''2'''
#print(found.replace('\\u', ' \\ '))


'''Can you improve this and remove the emojis properly?'''


'''3'''

#from pattern.en import sentiment
#
#print("'",found.replace('\\u', ' \\ '),"'" , " ------ Has a sentiment of " , sentiment(found)[0])