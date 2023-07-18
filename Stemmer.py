from abc import ABC, abstractmethod
from stemmerParser import RafiStemmer, StemmerCore, WordDict

class StemmerWrapper(ABC):
    @abstractmethod
    def stemSentence(self, sent):
        pass

class StemmerRK(StemmerWrapper):
    def __init__(self, stemmerCore: StemmerCore):
        self.stemmerCore = stemmerCore

    def stemSentence(self, sent):
        tokens = sent.split()
        stemmed_tokens = []
        for token in tokens:
            stemmed_tokens.append(self.stemmerCore.stem_word(token))
        return " ".join(stemmed_tokens)
    

if __name__ == "__main__":
    wordDict = WordDict()
    priorityRules = {
        "replace": 1,
        "remove": [0,2,3],
        "ambiguous": 4
    }

    stemmerCore = RafiStemmer(wordDict, priorityRules)
    stemmerWrapper = StemmerRK(stemmerCore)
    
    print(stemmerWrapper.stemSentence("সে কারণেই পর্যটকরা কুয়াকাটায় এসে প্রকৃতির নিয়মের সাথে নিজের মনকে একাকার করে"))