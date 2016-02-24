__author__ = 'penghao'
import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open

#sys.setdefaultencoding('utf8')
change_of_sentence_flag = 0 #a marker for the end of sentence
boi_full_list = [] #store all the boi tags that occur in the training set
boi_end_list = [] #store boi tags that are at the end of the sentence
wordStartList = [] #store words that are begining of the sentence
wordStartList.append('I')
BOI_list = ['B-NP', 'I-NP', 'O']
labeled_features = []

training_file = open("oct27.train.np", "rb")

previous_BOI = "start"
input_file = training_file
for line in input_file:
	s = re.match(r'^\s*$', line)  #find empty line
	if s:
		change_of_sentence_flag = 1
		previous_BOI = "start"
	else:
		sentenceList = line.split()
		word = sentenceList[0]
		print word
		tag = sentenceList[1]
		boi = sentenceList[1]

		#store words that are begining of the sentence
		if change_of_sentence_flag == 1:

			wordStartList.append(word)
			boi_end_list.append(boi_full_list[-1])
			change_of_sentence_flag = 0

		boi_full_list.append(boi)
		item = word, tag, boi, previous_BOI
        labeled_features.append(item)
	previous_BOI = boi

print boi_full_list
print wordStartList
print boi_end_list
print labeled_features

input_file.close()