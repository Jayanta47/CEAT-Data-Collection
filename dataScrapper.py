from Stemmer import Stemmer
import pybmoore

class DataScrapper():
    def __init__(self,
                 rootDir: str,
                 filenames: list[str],
                 stemmer: Stemmer,
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
            if self.evaluateWords(' '.join([prevSent, currentSent, nextSent])):
                self.sentenceList.append(' '.join([prevSent, currentSent, nextSent]))
                self.sentSerial += 1
                self.filesIndexcList.append(file.name)
            prevSent = currentSent
            currentSent = nextSent

    def scrapeData(self):
        self.resetLists()
        for filename in self.filenames:
            with open(self.rootDir + "/" + filename, "r") as file:
                self.lookIntoFile(file)
                

if __name__ == "__main__":
    scp = DataScrapper("data", ["test.txt"], None, ["a", "b", "c"])