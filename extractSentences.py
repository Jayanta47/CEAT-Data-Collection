from dataScrapper import * 
import pandas as pd
from wordFinder import *

def get_all_files(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def getWeatWords(filepath):
    with open(filepath, "r") as file:
        return [w.rstrip(', ') for w in file.read().split("\n")]

def getSuffixList(filepath):
    with open(filepath, "r") as file:
        return [w.rstrip(', ') for w in file.read().split("\n")]

if __name__ == "__main__":
    weatWordList = getWeatWords("allWeatWords.txt")
    suffixList = getSuffixList("possibleSuffix.txt")

    filesList = []
    for i in range(1, len(sys.argv)):
        files = get_all_files(sys.argv[i])
        filesList.extend(files)

    stemmerCore = RafiStemmer()
    stemmerWrapper = StemmerRK(stemmerCore)

    # scrapper = DataScrapper(filesList, stemmerWrapper, weatWordList)
    evaluator = WordEvaluatorRegex(weatWordList)
    scrapper = DataScrapper(filesList, evaluator=evaluator)

    weatWordDict, sentenceList, filesIndexList = scrapper.scrapeData()

    saveData = {
        "WEAT word": [],
        "Sentences": []
    }
    for word in weatWordList:
        values = weatWordDict[word]
        value_str = '-'.join(str(i) for i in values)
        saveData["WEAT word"].append(word)
        saveData["Sentences"].append(value_str)
    saveData["Index"] = range(len(weatWordList))
    weatWordDF = pd.DataFrame(saveData)

    sentencesDF = pd.DataFrame({"Index": range(len(sentenceList)), "Sentence": sentenceList, "SourceFile": filesIndexList})
    
    weatWordDF.to_csv("weatWordsSentences.csv", index=False)
    sentencesDF.to_csv("sentences.csv", index=False)