from abc import ABC, abstractmethod
import pybmoore
import re

class WordFinder(ABC):
    def __init__(self, weatWordList: list[str]):
        self.weatWordList = weatWordList
        self.weatWordDict = {}
        for word in weatWordList:
            self.weatWordDict[word] = []

    def getWeatWordDict(self):
        return self.weatWordDict

    @abstractmethod
    def evaluate(self, sent: str, serial: int) -> bool:
        pass

class WordEvaluatorBMoore(WordFinder):
    def __init__(self, weatWordList: list[str]):
        super().__init__(weatWordList)

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            matches = pybmoore.search(word, sent)
            if (len(matches) >= 1):
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound
    

class WordEvaluatorRegex(WordFinder):
    def __init__(self, weatWordList: list[str]):
        super().__init__(weatWordList)
        self.weatWordPatterns = {}

        for word in weatWordList:
            self.weatWordPatterns[word] = r'\b' + word + r'\w*'

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            pattern = self.weatWordPatterns[word]
            matches = re.match(pattern, sent)
            if matches:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound
    
class WordEvaluatorRegexSuffix(WordFinder):
    def __init__(self, weatWordList: list[str], suffixList: list[str]):
        super().__init__(weatWordList)
        self.weatWordPatterns = {}

        suffixString = '|'.join(suffixList)

        for word in weatWordList:
            self.weatWordPatterns[word] = r'\b' + word + f"(?:{suffixString})?" + r'\W'

    def evaluate(self, sent: str, serial: int) -> bool:
        isMatchFound = False
        for word in self.weatWordList:
            pattern = self.weatWordPatterns[word]
            matches = re.match(pattern, sent)
            if matches:
                isMatchFound = True
                self.weatWordDict[word].append(serial)
        return isMatchFound