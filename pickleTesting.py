import pickle

weatWordSentenceDict = pickle.load(open("./results/results2.pkl", "rb"))
# for word in weatWordSentenceDict:
#     print(word)


same_words_dict = {
    "উঁকুন": ["উকুন"],
    "চলচ্চিত্র": ["চলচিত্র"],
    "প্রাণিবিদ্যা": ["প্রাণীবিদ্যা"],
    "তরবারি": ["তরবারী"],
    "মাকড়সা": ["মাকড়শা"],
    "বারুদ": ["গোলাবারুদ"],
    "বেলি": ["বেলী"],
    "উইপোকা": ["উঁইপোকা"],
}

for masterWord, value in same_words_dict.items():
    for word in value:
        if word in weatWordSentenceDict:
            weatWordSentenceDict.pop(word)

pickle.dump(weatWordSentenceDict, open("results/results.pkl", "wb"))


for word in weatWordSentenceDict:
    print(word)
