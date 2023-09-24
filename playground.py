# import pickle

# weatWordSentenceDict = pickle.load(open("./results/results.pkl", "rb"))

# print(len(weatWordSentenceDict))

# total = 0

# for word in weatWordSentenceDict:
#     total += min(len(weatWordSentenceDict[word]), 100000)

# print("With Gpu:")
# print(total / (40 * 60 * 60), "hrs")

# print("Without Gpu:")
# print(total / (10 * 60 * 60), "hrs")

# print("Sir's PC:")
# print(total / (85 * 60 * 60), "hrs")


def createSmallerDataset():
    newDict = {}
    sortedDict = sorted(
        weatWordSentenceDict, key=lambda x: len(weatWordSentenceDict[x]), reverse=True
    )
    for word in sortedDict:
        newDict[word] = weatWordSentenceDict[word][:10]
    pickle.dump(newDict, open("./results/results_small.pkl", "wb"))


# createSmallerDataset()


from wordFinder import *
import json
from extractSentences import normalizeWeatDict
from Stemmer import *
import numpy as np
import random
from tqdm import tqdm
import pickle

weatWordDict = json.load(open("weatWordsWithSuffix.jsonl", "r", encoding="utf-8"))
weatWordDict = normalizeWeatDict(weatWordDict)
evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)

sent = "সারাটাদিন ব্যপক ধোয়ামোছার পর রুমে খাট ঢুকানো হইল। টিকটিকির বিরুদ্ধে যুদ্ধে নামিয়া সেইদিন বহুত কামলা খাটিতে হইয়াছিল সেই ৪ জনের! অবশেষে সুখের দিন আসিল।"
word = "টিকটিকি"

print(evaluator.getIndex(sent, word))
