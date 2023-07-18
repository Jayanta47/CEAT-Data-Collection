from Stemmer import *
from stemmerParser import RafiStemmer, WordDict
import pybmoore
import os 
import sys 

class DataScrapper():
    def __init__(self,
                 rootDir: str,
                 filenames: list[str],
                 stemmer: StemmerWrapper,
                 weatWordList: list[str]) -> None:
        self.rootDir = rootDir
        self.filenames = filenames
        self.stemmer = stemmer
        self.weatWordList = weatWordList
        self.weatWordDict = {}
        for word in weatWordList:
            self.weatWordDict[word] = []
        self.resetLists()

    def resetLists(self):
        self.sentenceList = []
        self.filesIndexList = []
        self.sentSerial = 0
        self.currentFileName = ""

    def evaluateWords(self, sent:str):
        isMatchFound = False
        for word in self.weatWordList:
            matches = pybmoore.search(word, sent)
            if (len(matches) == 1):
                self.weatWordDict[word].append(self.sentSerial)
                isMatchFound = True
        return isMatchFound
            

    def lookIntoFile(self, file):
        currentSent = ""
        prevSent = ""
        nextSent = ""
        for line in file.readlines():
            line = line.strip()
            nextSent = self.stemmer.stemSentence(line)
            if currentSent == "":
                continue
            # check if the combination of the sentences has the word
            sentence = ' '.join([prevSent, currentSent, nextSent])
            if self.evaluateWords(sentence):
                self.sentenceList.append(sentence)
                self.sentSerial += 1
                self.filesIndexList.append(self.currentFileName)
            prevSent = currentSent
            currentSent = nextSent

    def scrapeData(self):
        self.resetLists()
        for filename in self.filenames:
            with open(os.path.join(self.rootDir, filename), "r") as file:
                self.currentFileName = filename
                self.lookIntoFile(file)

        return self.weatWordDict, self.sentenceList, self.filesIndexList
                

if __name__ == "__main__":
    # scp = DataScrapper("data", ["test.txt"], None, ["a", "b", "c"])
    filesList = []
    for i in range(1, len(sys.argv)):
        files = os.listdir(sys.argv[i])
        filesList.extend(files)

    wordDict = WordDict()
    priorityRules = {
        "replace": 1,
        "remove": [0,2,3],
        "ambiguous": 4
    }

    stemmerCore = RafiStemmer(wordDict, priorityRules)
    stemmerWrapper = StemmerRK(stemmerCore)

    weatWordList = ["প্রকৃতি", "পর্যটক", "কুয়াকাটা", "বৌদ্ধ", "লালবাগ", "মোগল"]

    dsc = DataScrapper("", filesList, stemmerWrapper, weatWordList)

    print(dsc.scrapeData())