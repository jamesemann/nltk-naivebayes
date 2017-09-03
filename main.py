import nltk
import pickle
import os.path
import os
import flask

def getSamples(file):
    with open(file, 'r') as f:
        return f.readlines()

def getVocabulary():
    turnoffLightWordList = [word for line in getSamples('intent-turnofflight.txt') for word in line.split()]
    turnonLightWordList = [word for line in getSamples('intent-turnonlight.txt') for word in line.split()]
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
    instanceFile = 'intent.pickle'
    if(not os.path.isfile(instanceFile)):
        print('Training model started');
        turnOffLightTaggedTrainingList = [{'tokens': phrase.split(), 'label': 'intent-turnofflight'} for phrase in getSamples('intent-turnofflight.txt')]
        turnOnLightTaggedTrainingList = [{'tokens': phrase.split(), 'label': 'intent-turnonlight'} for phrase in getSamples('intent-turnonlight.txt')]
        fullTaggedTrainingData = [item for sublist in [turnOffLightTaggedTrainingList, turnOnLightTaggedTrainingList] for item in sublist]
        trainingData = [(review['tokens'], review['label']) for review in fullTaggedTrainingData]
        trainingFeatures = nltk.classify.apply_features(extractFeatures, trainingData)
        classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
        fileWrite = open(instanceFile, 'wb')
        pickle.dump(classifier, fileWrite, pickle.HIGHEST_PROTOCOL)
        fileWrite.close()
        print('Training model complete');

    fileRead = open(instanceFile, 'rb')
    classifier = pickle.load(fileRead)
    fileRead.close()
    return classifier

app = flask.Flask(__name__)

@app.route('/intent')
def intent():
	utterance = flask.request.args.get('utterance')
	instance = load()
	return flask.jsonify(intent=classify(instance, utterance))

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)