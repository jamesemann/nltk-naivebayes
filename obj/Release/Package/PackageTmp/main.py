import nltk
import pickle
import os.path

from os import environ
from flask import Flask

with open('intent-turnofflight.txt', 'r') as f:
    intentTurnoffLightSamples = f.readlines()

with open('intent-turnonlight.txt', 'r') as f:
    intentTurnonLightSamples = f.readlines()

def getVocabulary():
    turnoffLightWordList = [word for line in intentTurnoffLightSamples for word in line.split()]
    turnonLightWordList = [word for line in intentTurnonLightSamples for word in line.split()]
    return list(set([item for sublist in [turnoffLightWordList,turnonLightWordList] for item in sublist]))

def extractFeatures(utterance):
    features={}
    for word in getVocabulary():
        features[word]=(word in set(utterance))
    return features

def classify(classifierInstance,utterance):
    problemInstance = utterance.split()
    problemFeatures = extractFeatures(problemInstance)
    return classifierInstance.classify(problemFeatures)

def load():
    instanceFile = 'intent4.pickle'
    if(not os.path.isfile(instanceFile)):
        turnOffLightTaggedTrainingList = [{'tokens': phrase.split(), 'label': 'intent-turnofflight'} for phrase in
                                          intentTurnoffLightSamples]
        turnOnLightTaggedTrainingList = [{'tokens': phrase.split(), 'label': 'intent-turnonlight'} for phrase in
                                         intentTurnonLightSamples]
        fullTaggedTrainingData = [item for sublist in [turnOffLightTaggedTrainingList, turnOnLightTaggedTrainingList]
                                  for item in sublist]
        trainingData = [(review['tokens'], review['label']) for review in fullTaggedTrainingData]
        trainingFeatures = nltk.classify.apply_features(extractFeatures, trainingData)
        classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
        fileWrite = open(instanceFile, 'wb')
        pickle.dump(classifier, fileWrite, pickle.HIGHEST_PROTOCOL)
        fileWrite.close()

    fileRead = open(instanceFile, 'rb')
    classifier = pickle.load(fileRead)
    fileRead.close()
    return classifier


#print(classify(instance, 'switch the light off'))
#print(classify(instance, 'turn it off'))
#print(classify(instance, 'please turn off the light'))
#print(classify(instance, 'turn it on'))
#print(classify(instance, 'light on'))
#print(classify(instance,'please would you turn on the light'))

#print (extract_features('hi there'))


app = Flask(__name__)

@app.route('/intent')
def hello_world():
	instance = load()
	return classify(instance, 'switch the light off')

#if __name__ == '__main__':
#  app.run()


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)