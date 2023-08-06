from wordFinder import *
import json
from extractSentences import normalizeWeatDict
from Stemmer import *
from abc import ABC, abstractmethod
import numpy as np
import random
from tqdm import tqdm
import pickle
from normalizer import normalize

# weatWordDict = json.load(open('weatWordsWithSuffix.jsonl', 'r', encoding='utf-8'))
# weatWordDict = normalizeWeatDict(weatWordDict)
# weatWordList = list(weatWordDict.keys())
# evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)
# sentence = "খাওয়া দাওয়া পার ডে ২২০ - ৩০০ টাকা। গোলাপ রাজ্য (ভ্রমণ কাহিনী) অভিজিৎ সাগর A rose for my rose.......এটার বদলে যদি বলি a kingdom of rose for my beautiful rose ? কেমন হবে বলুন তো?- যা হবে তা ভাবনাতেই থাকুক।"
# indices = evaluator.getIndex(sentence,
#                          "গোলাপ")
# print(indices)

# print(sentence.split()[indices[0]])

def getSentencesSample(sentenceList, maxItems = 1000):
    numItems = min(maxItems, len(sentenceList))
    items = sentenceList if numItems < maxItems else random.sample(sentenceList, maxItems)
    return items

class SentenceProcessor():
    def __init__(self, evaluator: WordFinder, 
                 stemmer: StemmerWrapper = None) -> None:
        self.evaluator = evaluator 
        self.stemmer = stemmer

    def setLength(self, length: int) -> None:
        self.length = length

    def shortenSentence(self, sentence: str, word: str) -> str:
        indices = self.evaluator.getIndex(sentence, word)
        if len(indices) == 0:
            raise Exception("No match found")
        
        index = indices[0]
        if self.length == -1:
            return sentence, index
        words = sentence.split()
        modifiedIndex = index
        if len(words) >= self.length:
            if index < self.length//2:
                wordsUsed = words[:self.length]
            elif (len(words) - index) < self.length//2 + 1:
                wordsUsed = words[-self.length:]
                modifiedIndex = self.length - (len(words) - index)
            else:
                wordsUsed = words[index - self.length//2: index + self.length//2]
                modifiedIndex = self.length//2
            newSentence = ' '.join(wordsUsed)
        else:
            newSentence = sentence 

        if self.stemmer:
            newSentence = self.stemmer.stemSentence(newSentence)
        return newSentence, modifiedIndex 
    
class ModelWrapper(ABC):
    @abstractmethod
    def getWordVector(self, word: str, sent: str, index: int) -> np.array:
        pass

class EmbeddingExtractor():
    def __init__(self, 
                 sentenceProcessor: SentenceProcessor,
                 model: ModelWrapper) -> None:
        self.sentenceProcessor = sentenceProcessor
        self.model = model

    def savePickle(self, filename: str, data: dict[str, list[np.array]]):
        pickle.dump(data, open(filename, "wb"))

    def extract(self, weatWordDict: dict[str, list[str]]) -> dict[str, list[np.array]]:
        weatWordEmbeddings = {}

        for word in weatWordDict:
            for index, sentence in tqdm(enumerate(weatWordDict[word]), desc="Processing Sentences"):
                sentence, index = self.sentenceProcessor.shortenSentence(sentence, word)
                weatWordEmbeddings[word].append(self.model.getWordVector(word, sentence, index))

        return weatWordEmbeddings


class BanglaBertEmbeddingExtractor(ModelWrapper):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer #add_special_tokens=False

    def getWordVector(self, word: str, sent: str, index: int) -> np.array:
        normalized_sentence = normalize(sent) # no additional params needed?
        input_tokens = self.tokenizer.encode(normalized_sentence, add_special_tokens=False, return_tensors="pt")
        if torch.cuda.is_available():
            input_tokens = input_tokens.to('cuda')
        with torch.no_grad():
            output = self.model(**input_tokens)
            return output[1][24][0].detach().cpu().numpy()[index]

if __name__ == "__main__":
    weatWordDict = json.load(open('weatWordsWithSuffix.jsonl', 'r', encoding='utf-8'))
    weatWordDict = normalizeWeatDict(weatWordDict)
    evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)
    processor = SentenceProcessor(evaluator)
    processor.setLength(9)

    # test index
    sent1 = "১৫০ টাকা নিয়েছিল। গোলাপ গ্রামের মজার একটা ব্যাপার লক্ষ করেছিলাম। সেখানে সব বাড়ির সাথেই লাগোয়া ছোটছোট গোলাপের বাগান আছে। গাড়ি নিয়ে স্বপরিবারে বেড়াতে যাওয়ার প্ল্যান করার আগে অবশ্যই নিরাপত্তার ব্যপারটি মাথায় রাখতে হবে। পরিবারের নিরাপত্তায় সবার সাথে ফোন এবং ফোনে রিচার্জ করে নিলে ভাল হয়।"
    sent2 = "যথাযথ কর্তৃপক্ষের উচিত এই সকল নিদর্শনসমুহের নিয়মিত পরিচর্যা করা, নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপ রাজ্য (ভ্রমণ কাহিনী) অভিজিৎ সাগর A rose for my rose.......এটার বদলে যদি বলি a kingdom of rose for my beautiful rose ? কেমন হবে বলুন তো?- যা হবে তা ভাবনাতেই থাকুক।"
    sent3 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের রাজ্য"
    sent4 = "গোলাপের রাজ্য"
    sent5 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের। রাজ্য"

    print(processor.shortenSentence(sent1, "গোলাপ"))
    print(processor.shortenSentence(sent2, "গোলাপ"))
    print(processor.shortenSentence(sent3, "গোলাপ"))
    print(processor.shortenSentence(sent4, "গোলাপ"))
    print(processor.shortenSentence(sent5, "গোলাপ"))

    