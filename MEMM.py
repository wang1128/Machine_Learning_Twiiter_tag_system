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

testing_file = open("oct27.train.np", "rb")

training_file = open("oct27.test.np", "rb")

f = open('my_classifier300.pickle', 'rb')
maxent_classifier = pickle.load(f)
f.close()


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
		tag = 1
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

	ProbE = format(countEnd/(countTag*1.0+1), '.5f')
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
	#features['cap_start'] = word not in wordStartList and word[0].isupper()
	features['previous_NC'] = previous_BOI

	return features
labeled_featuresets = [(MEMM_features(word, tag, previous_BOI), boi )for (word, tag, boi, previous_BOI) in labeled_features]

train_set = labeled_featuresets

#f = open("my_classifier.pickle", "wb")

#maxent_classifier = MaxentClassifier.train(train_set, max_iter=30)
#pickle.dump(maxent_classifier , f)



def MEMM(wordList,tagList):
	BOI_list = ['!', '#', '$', '&', ',', 'A', '@', 'E', 'D', 'G', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'V', 'Y', 'X', 'Z', '^', '~']
	w1 = wordList[0] #the first word of the sentence
	t1 = tagList[0]
	tRange = len(BOI_list)
	wRange = len(wordList)

	viterbi = [[0 for x in range(100)] for x in range(100)]
	backpointer = [['' for x in range(100)] for x in range(100)]
	#intialization
	for t in range(tRange):#t = 0,1,2
		probability = maxent_classifier.prob_classify(MEMM_features(w1,t1, "start" ))
		posterior = float(probability.prob(BOI_list[t]))
		#print ("boi: " + BOI_list[t] + ' posterior (start)' + str(posterior))
		#score transition 0(start) -> q given w1
		viterbi[t][1] = posterior
		backpointer[t][1] = 0 #stand for q0 (start point)

	#for word w from 2 to T
	maxViterbi = 0
	maxPreviousState = 0
	maxPreTerminalProb = 0
	for w in range (1, wRange):
		for t in range (tRange):
			#find max verterbi = max (previous * posterior)
			word = wordList[w]
			tag = tagList[w]
			probability = maxent_classifier.prob_classify(MEMM_features(word,tag,BOI_list[0] ))
			posterior = float(probability.prob(BOI_list[t]))
			maxViterbi = float(viterbi[0][w]) * posterior
			maxPreviousState = 0
			for i in range (1, tRange):
				word = wordList[w]
				tag = tagList[w]
				probability = maxent_classifier.prob_classify(MEMM_features(word,tag,BOI_list[i] ))
				posterior = float(probability.prob(BOI_list[t]))
				if float(viterbi[i][w]) * posterior > maxViterbi:
					 maxViterbi = float(viterbi[i][w]) * posterior
					 maxPreviousState = i #content BOI_List[i]
			viterbi[t][w+1] = maxViterbi
			backpointer[t][w+1] = BOI_list[maxPreviousState] #points to the matrix x axis (max previous)

			maxViterbi = 0
			maxPreviousState = 0
			maxPreTerminalProb = 0
	#termination step
	#viterbi[qF, T] = max (viterbi[s,T] *as,qF)
	maxPreTerminalProb = float(viterbi[0][wRange] )* float(dicE[BOI_list[0]]["END"])

	maxPreviousState = 0
	for i in range (1, tRange):

		if float(viterbi[i][wRange]) * float(dicE[BOI_list[i]]["END"]) > maxPreTerminalProb:
			maxPreTerminalProb = float(viterbi[i][wRange]) * float(dicE[BOI_list[i]]["END"])

			maxPreviousState = i

			#print ("maxPreTerminalProb: " + str(maxPreTerminalProb))
	viterbi[tRange][wRange+1] = maxPreTerminalProb
	backpointer[tRange][wRange+1] = BOI_list[maxPreviousState]
	#return POS tag path
	pathReverse = [BOI_list[maxPreviousState]]
	maxPreviousTag = BOI_list[maxPreviousState]

	i = 0
	while i < (wRange -1):
		pathReverse.append(backpointer[BOI_list.index(maxPreviousTag)][wRange - i])
		maxPreviousTag = backpointer[BOI_list.index(maxPreviousTag)][wRange - i]
		i = i + 1

	#reverse the path to make it correct
	index = len(pathReverse)
	path = []
	while index >= 1 :
		path.append(pathReverse[index - 1])
		index = index -1
	return path

##test

wordList = [] #store words in a sentence
tagList = [] #store part-of-speech tag in a sentence
boiList = [] #store boi tags in a sentence
#prob_table = {} #stpre the posterior
previous_BOI = "start"
BOI_list = ['!', '#', '$', '&', ',', 'A', '@', 'E', 'D', 'G', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'V', 'Y', 'X', 'Z', '^', '~']
countAll = 0
countRight = 0

input_file = testing_file
for line in input_file:

	if line.strip() != '': #if not empty do following
		sentenceList = line.split()
		word = sentenceList[0]
		#print (word)
		tag = 1
		boi = sentenceList[1]
		wordList.append(word)
		tagList.append(tag)
		boiList.append(boi)

		if change_of_sentence_flag == 1:
			wordStartList.append(word)
			change_of_sentence_flag = 0
	s = re.match(r'^\s*$', line)  #find empty line
	if s:
		print (wordList)

		change_of_sentence_flag = 1
		previous_BOI = "start"
		path = MEMM(wordList, tagList) #list of BOI_tags returned by HMM function call
		print path

		for i in range(len(wordList)): #part_of_speech_tag(tagList) and token_list(wordList) has the same length
			output_file2.write(wordList[i]+"	"+  " " + path[i] + "\n")
			if boiList[i] == path[i]:
				countRight += 1

		countAll = countAll + len(wordList)
		output_file.write("\n")
		wordList = [] # refresh word list
		tagList = []
		boiList = []
		print "Processing"
		#prob_table = {}#refresh prob_table

print countAll
print countRight

input_file.close()
output_file.close()