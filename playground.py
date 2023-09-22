import pickle

weatWordSentenceDict = pickle.load(open("./results/results.pkl", "rb"))

print(len(weatWordSentenceDict))


def createSmallerDataset():
    newDict = {}
    sortedDict = sorted(
        weatWordSentenceDict, key=lambda x: len(weatWordSentenceDict[x]), reverse=True
    )
    for word in sortedDict:
        newDict[word] = weatWordSentenceDict[word][:10]
    pickle.dump(newDict, open("./results/results_small.pkl", "wb"))


createSmallerDataset()
