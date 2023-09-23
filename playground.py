import pickle

weatWordSentenceDict = pickle.load(open("./results/results.pkl", "rb"))

print(len(weatWordSentenceDict))

total = 0

for word in weatWordSentenceDict:
    total += min(len(weatWordSentenceDict[word]), 100000)

print("With Gpu:")
print(total / (40 * 60 * 60), "hrs")

print("Without Gpu:")
print(total / (10 * 60 * 60), "hrs")


def createSmallerDataset():
    newDict = {}
    sortedDict = sorted(
        weatWordSentenceDict, key=lambda x: len(weatWordSentenceDict[x]), reverse=True
    )
    for word in sortedDict:
        newDict[word] = weatWordSentenceDict[word][:10]
    pickle.dump(newDict, open("./results/results_small.pkl", "wb"))


# createSmallerDataset()
