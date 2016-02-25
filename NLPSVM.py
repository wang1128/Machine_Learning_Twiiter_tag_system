__author__ = 'penghao'
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
from sklearn.cross_validation import train_test_split
from pystruct.learners import NSlackSSVM
from pystruct.models import MultiClassClf
from sklearn import svm


#a marker for the end of sentence
wordList = []
boi_full_list = [] #store all the boi tags that occur in the training set
boi_end_list = [] #store boi tags that are at the end of the sentence
wordStartList = [] #store words that are begining of the sentence
wordStartList.append('I')
BOI_list = ['!', '#', '$', '&', ',', 'A', '@', 'E', 'D', 'G', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'V', 'Y', 'X', 'Z', '^', '~']

l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


testing_file = open("oct27.test.np", "rb")
training_file = open("oct27.train.np", "rb")

def featureGenerate(file,n):
	labeled_features = []
	change_of_sentence_flag = 0
	previous_BOI = "start"
	input_file = file
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
			if n == 1:
				if boi == '!':
					l1.append(word)
				if boi == '#':
					l2.append(word)
				if boi == '$':
					l3.append(word)
				if boi == '&':
					l4.append(word)
				if boi == ',':
					l5.append(word)
				if boi == 'A':
					l6.append(word)
				if boi == '@':
					l7.append(word)
				if boi == 'E':
					l8.append(word)
				if boi == 'D':
					l9.append(word)
				if boi == 'G':
					l10.append(word)
				if boi == 'M':
					l11.append(word)
				if boi == 'L':
					l12.append(word)
				if boi == 'O':
					l13.append(word)
				if boi == 'N':
					l14.append(word)
				if boi == 'P':
					l15.append(word)
				if boi == 'S':
					l16.append(word)
				if boi == 'R':
					l17.append(word)
				if boi == 'U':
					l18.append(word)
				if boi == 'T':
					l19.append(word)
				if boi == 'V':
					l20.append(word)
				if boi == 'Y':
					l21.append(word)
				if boi == 'X':
					l22.append(word)
				if boi == 'Z':
					l23.append(word)
				if boi == '^':
					l24.append(word)
				if boi == '~':
					l25.append(word)
			#store words that are begining of the sentence
			if change_of_sentence_flag == 1:

				wordStartList.append(word)
				boi_end_list.append(boi_full_list[-1])
				change_of_sentence_flag = 0

			boi_full_list.append(boi)
			wordList.append(word)
			item = word, boi, previous_BOI
			labeled_features.append(item)
		previous_BOI = boi
	return labeled_features

train= featureGenerate(training_file,1)
test = featureGenerate(testing_file,0)
print len(train)
print len(test)

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
def MEMM_features(word, boi,previous_BOI):

	features = []
	#features.append(word)
	#features['current_tag'] = tag
	#puc = '-'.decode("utf-8")
	 #some char is outof ASCII
	#print (word)
	if word in l1:
		features.append(1)
	else: features.append(0)
	if word in l2:
		features.append(1)
	else: features.append(0)
	if word in l3:
		features.append(1)
	else: features.append(0)
	if word in l4:
		features.append(1)
	else: features.append(0)
	if word in l5:
		features.append(1)
	else: features.append(0)
	if word in l6:
		features.append(1)
	else: features.append(0)
	if word in l7:
		features.append(1)
	else: features.append(0)
	if word in l8:
		features.append(1)
	else: features.append(0)
	if word in l9:
		features.append(1)
	else: features.append(0)
	if word in l10:
		features.append(1)
	else: features.append(0)
	if word in l11:
		features.append(1)
	else: features.append(0)
	if word in l12:
		features.append(1)
	else: features.append(0)
	if word in l13:
		features.append(1)
	else: features.append(0)
	if word in l14:
		features.append(1)
	else: features.append(0)
	if word in l15:
		features.append(1)
	else: features.append(0)
	if word in l16:
		features.append(1)
	else: features.append(0)
	if word in l17:
		features.append(1)
	else: features.append(0)
	if word in l18:
		features.append(1)
	else: features.append(0)
	if word in l19:
		features.append(1)
	else: features.append(0)
	if word in l20:
		features.append(1)
	else: features.append(0)
	if word in l21:
		features.append(1)
	else: features.append(0)
	if word in l22:
		features.append(1)
	else: features.append(0)
	if word in l23:
		features.append(1)
	else: features.append(0)
	if word in l24:
		features.append(1)
	else: features.append(0)
	if word in l25:
		features.append(1)
	else: features.append(0)
	features.append(word[0].isupper())
	features.append(word in wordStartList)
	features.append(word not in wordStartList and word[0].isupper())
	for element in ['!', '#', '$', '&', ',', 'A', '@', 'E', 'D', 'G', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'V', 'Y', 'X', 'Z', '^', '~']:
		if previous_BOI == element:
			features.append(1)
		else:
			features.append(0)
	if previous_BOI in boi_end_list:
		features.append(1)
	else: features.append(0)


	#features.append(previous_BOI)

	return features
print len(train)

Xdata = []
for (word,boi,previous_BOI) in train:
	f = MEMM_features(word, boi, previous_BOI)
	Xdata.append(f)

Xtarget = [ boi for (word, boi, previous_BOI) in train]

TestData =[]
for (word,boi,previous_BOI) in test:
	f = MEMM_features(word, boi, previous_BOI)
	TestData.append(f)

TestTarget = [ boi for (word, boi, previous_BOI) in test]
print len(Xdata)
print len(TestData)
print Xdata[1]
print TestData[1]
countList = []
X_train, X_test, y_train, y_test = train_test_split(Xdata, Xtarget, test_size=0.4, random_state=0)

for i in (1,3 ,10,15,20,25,30,50,100,150,200,300,500,700,1000,1500,2000,1500):
	c_f=6.5
	gamma_f=0.5
	clf = svm.SVC(max_iter=i)
	clf.fit(Xdata, Xtarget)
	a = clf.predict(TestData)
	count = 0
	for i in range(len(a)):
		if a[i] == TestTarget[i]:
			count += 1
	countList.append(count)
	#print count, len(a)
	clf = None
print countList
print len(a)