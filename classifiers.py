__author__ = 'penghao'
__author__ = 'penghao'
import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open

change_of_sentence_flag = 0 #a marker for the end of sentence
boi_full_list = [] #store all the boi tags that occur in the training set
boi_end_list = [] #store boi tags that are at the end of the sentence
wordStartList = [] #store words that are begining of the sentence
wordStartList.append('I')
BOI_list = ['!', '#', '$', '&', ',', 'A', '@', 'E', 'D', 'G', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'V', 'Y', 'X', 'Z', '^', '~']
labeled_features = []
output_file = open("twitter.txt", "wb")
output_file2 = open("trainA30Test.txt", "wb")

testing_file = open("oct27.test.np", "rb")

training_file = open("oct27.train.np", "rb")

#f = open('my_classifier.pickle', 'rb')
#maxent_classifier = pickle.load(f)
#f.close()


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
		#print word
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


print len(boi_end_list)
print len(boi_full_list)

dicE = {} #temporarry dic
countTag = 0
countEnd = 0
#calculate the prior (End|state) = C(state, End)/C(state)
for i in BOI_list:
	for j  in range(len(boi_end_list)):
		for f in boi_full_list:
			if j == 0:
				if i == f:
					countTag = countTag + 1
		if i == boi_end_list[j]:
			countEnd = countEnd + 1

	ProbE = format(countEnd/(countTag*1.0), '.5f')
	#print ProbE

	dicE.update({i: {"END":ProbE}})

	countEnd = 0
	countTag = 0
def MEMM_features(word, tag, previous_BOI):
	stemmer = PorterStemmer()
	features = {}
	features['current_word'] = word
	features['current_tag'] = tag
	#puc = '-'.decode("utf-8")
	 #some char is outof ASCII
	#print (word)
	features['capitalization'] = word[0].isupper()
	features['start_of_sentence'] = word in wordStartList
	features['cap_start'] = word not in wordStartList and word[0].isupper()
	features['previous_NC'] = previous_BOI

	return features
labeled_featuresets = [(MEMM_features(word, tag, previous_BOI), boi )for (word, tag, boi, previous_BOI) in labeled_features]

train_set = labeled_featuresets

f = open("my_classifier300.pickle", "wb")

maxent_classifier = MaxentClassifier.train(train_set, max_iter=300)
pickle.dump(maxent_classifier , f)