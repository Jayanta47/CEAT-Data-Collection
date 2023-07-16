from abc import ABC, abstractmethod

class Stemmer(ABC):
    @abstractmethod
    def stemSentence(self, sent):
        pass

class StemmerRK(Stemmer):
    def __init__(self, stemmerCore):
        self.stemmerCore = stemmerCore

    def stemSentence(self, sent):
        tokens = sent.split()
        stemmed_tokens = []
        for token in tokens:
            stemmed_tokens.append(self.stemmerCore.stem_word(token))
        return " ".join(stemmed_tokens)