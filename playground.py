import pickle
from normalizer import normalize


def removeUnwantedWords():
    weatWordSentenceDict = pickle.load(open("./results/results_augmented_V2.pkl", "rb"))
    print(len(weatWordSentenceDict))

    with open("./results/all_ceat_words.txt", "r") as file:
        wordList = []
        for word in file.readlines():
            wordList.append(
                normalize(
                    word.strip(), unicode_norm="NFKC", apply_unicode_norm_last=True
                )
            )
    with open("./results/all_ceat_words_norm.txt", "w") as file:
        for word in wordList:
            file.write(word + "\n")

    wordList = sorted(wordList)

    for word in wordList:
        print(f"{word}: {len(weatWordSentenceDict[word])}")

    unwanted = []
    for word in weatWordSentenceDict:
        if word not in wordList:
            unwanted.append(word)
    print("\n\n\nUnwanted Words:")
    for word in unwanted:
        print(f"{word}: {len(weatWordSentenceDict[word])}")

    for word in unwanted:
        weatWordSentenceDict.pop(word)

    print(len(weatWordSentenceDict))
    pickle.dump(weatWordSentenceDict, open("./results/result_final_v2.pkl", "wb"))


def printExecutionTime():
    total = 0
    weatWordSentenceDict = pickle.load(open("./results/results_augmented_V2.pkl", "rb"))
    for word in weatWordSentenceDict:
        total += min(len(weatWordSentenceDict[word]), 100000)

    print("With Gpu:")
    print(total / (40 * 60 * 60), "hrs")

    print("Without Gpu:")
    print(total / (10 * 60 * 60), "hrs")

    print("Sir's PC:")
    print(total / (60 * 60 * 60), "hrs")


def createSmallerDataset():
    newDict = {}
    weatWordSentenceDict = pickle.load(open("./results/results_augmented_V2.pkl", "rb"))
    print(len(weatWordSentenceDict))
    sortedDict = sorted(
        weatWordSentenceDict, key=lambda x: len(weatWordSentenceDict[x]), reverse=True
    )
    for word in sortedDict:
        newDict[word] = weatWordSentenceDict[word][:10]
    pickle.dump(newDict, open("./results/results_small.pkl", "wb"))


# createSmallerDataset()


def examinEmbedding():
    weatWordEmbedidngDict = pickle.load(open("./embeddings/embeddings.pkl", "rb"))
    print(len(weatWordEmbedidngDict))

    print(weatWordEmbedidngDict["গোলাপ"][1].size)


from wordFinder import *
import json
from extractSentences import normalizeWeatDict
from Stemmer import *
import numpy as np
import random
from tqdm import tqdm
import pickle

# weatWordDict = json.load(open("weatWordsWithSuffix.jsonl", "r", encoding="utf-8"))
# weatWordDict = normalizeWeatDict(weatWordDict)
# evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)

# sent = "সারাটাদিন ব্যপক ধোয়ামোছার পর রুমে খাট ঢুকানো হইল। টিকটিকির বিরুদ্ধে যুদ্ধে নামিয়া সেইদিন বহুত কামলা খাটিতে হইয়াছিল সেই ৪ জনের! অবশেষে সুখের দিন আসিল।"
# word = "টিকটিকি"

# print(evaluator.getIndex(sent, word))

# removeUnwantedWords()

# printExecutionTime()

examinEmbedding()
