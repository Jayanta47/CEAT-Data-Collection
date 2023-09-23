from wordFinder import *
import json
from extractSentences import normalizeWeatDict
from Stemmer import *
import numpy as np
import random
from tqdm import tqdm
import pickle
from models import ModelWrapper, BanglaBertEmbeddingExtractor


def getSentencesSample(sentenceList, maxItems=1000):
    numItems = min(maxItems, len(sentenceList))
    items = (
        sentenceList if numItems < maxItems else random.sample(sentenceList, maxItems)
    )
    return items


class SentenceProcessor:
    def __init__(self, evaluator: WordFinder, stemmer: StemmerWrapper = None) -> None:
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
            if index < self.length // 2:
                wordsUsed = words[: self.length]
            elif (len(words) - index) < self.length // 2 + 1:
                wordsUsed = words[-self.length :]
                modifiedIndex = self.length - (len(words) - index)
            else:
                wordsUsed = words[index - self.length // 2 : index + self.length // 2]
                modifiedIndex = self.length // 2
            newSentence = " ".join(wordsUsed)
        else:
            newSentence = sentence

        if self.stemmer:
            newSentence = self.stemmer.stemSentence(newSentence)
        return newSentence, modifiedIndex

    def getSpan(self, sentence: str, word: str, index: int) -> list[int]:
        return self.evaluator.getSpanByIndex(sentence, word, index)


class EmbeddingExtractor:
    def __init__(
        self, sentenceProcessor: SentenceProcessor, model: ModelWrapper
    ) -> None:
        self.sentenceProcessor = sentenceProcessor
        self.model = model

    def savePickle(self, filename: str, data: dict[str, list[np.array]]):
        pickle.dump(data, open(filename, "wb"))

    def extract(
        self, weatWordSentenceDict: dict[str, list[str]]
    ) -> dict[str, list[np.array]]:
        weatWordEmbeddings = {}

        for word in weatWordSentenceDict:
            weatWordEmbeddings[word] = []

        for word in weatWordSentenceDict:
            for index, sentence in tqdm(
                enumerate(weatWordSentenceDict[word]), desc="Processing Sentences"
            ):
                sentence, index = self.sentenceProcessor.shortenSentence(sentence, word)
                span = self.sentenceProcessor.getSpan(sentence, word, index)
                try:
                    weatWordEmbeddings[word].append(
                        self.model.getWordVector(word, sentence, index, span)
                    )
                except:
                    print("Error with: ", sentence)

        return weatWordEmbeddings


if __name__ == "__main__":
    weatWordDict = json.load(open("weatWordsWithSuffix.jsonl", "r", encoding="utf-8"))
    weatWordDict = normalizeWeatDict(weatWordDict)
    evaluator = WordEvaluatorRegexSuffixFixed(weatWordDict)
    processor = SentenceProcessor(evaluator)
    processor.setLength(9)

    model = BanglaBertEmbeddingExtractor(
        model_name="csebuetnlp/banglabert_large_generator",
        tokenizer_name="csebuetnlp/banglabert_large_generator",
    )

    extractor = EmbeddingExtractor(processor, model)

    # load the pickle file
    weatWordSentenceDict = pickle.load(open("./results/results.pkl", "rb"))
    embedding = extractor.extract(weatWordSentenceDict)
    pickle.dump(embedding, open("./embeddings/embeddings.pkl", "wb"))

    # test index
    # sent1 = "১৫০ টাকা নিয়েছিল। গোলাপ গ্রামের মজার একটা ব্যাপার লক্ষ করেছিলাম। সেখানে সব বাড়ির সাথেই লাগোয়া ছোটছোট গোলাপের বাগান আছে। গাড়ি নিয়ে স্বপরিবারে বেড়াতে যাওয়ার প্ল্যান করার আগে অবশ্যই নিরাপত্তার ব্যপারটি মাথায় রাখতে হবে। পরিবারের নিরাপত্তায় সবার সাথে ফোন এবং ফোনে রিচার্জ করে নিলে ভাল হয়।"
    # sent2 = "যথাযথ কর্তৃপক্ষের উচিত এই সকল নিদর্শনসমুহের নিয়মিত পরিচর্যা করা, নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপ রাজ্য (ভ্রমণ কাহিনী) অভিজিৎ সাগর A rose for my rose.......এটার বদলে যদি বলি a kingdom of rose for my beautiful rose ? কেমন হবে বলুন তো?- যা হবে তা ভাবনাতেই থাকুক।"
    # sent3 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের রাজ্য"
    # sent4 = "গোলাপের রাজ্য"
    # sent5 = "নতুবা এই সকল নিদর্শনসমুহ একসময় কালের গর্ভে বিলীন হয়ে যাবে। গোলাপের। রাজ্য"

    # short_sent_1 = processor.shortenSentence(sent1, "গোলাপ")[0]
    # short_sent_2 = processor.shortenSentence(sent2, "গোলাপ")[0]
    # short_sent_3 = processor.shortenSentence(sent3, "গোলাপ")[0]
    # short_sent_4 = processor.shortenSentence(sent4, "গোলাপ")[0]
    # short_sent_5 = processor.shortenSentence(sent5, "গোলাপ")[0]

    # print(short_sent_1, evaluator.getSpan(short_sent_1, "গোলাপ"))
    # print(short_sent_2, evaluator.getSpan(short_sent_2, "গোলাপ"))
    # print(short_sent_3, evaluator.getSpan(short_sent_3, "গোলাপ"))
    # print(short_sent_4, evaluator.getSpan(short_sent_4, "গোলাপ"))
    # print(short_sent_5, evaluator.getSpan(short_sent_5, "গোলাপ"))

    # print(processor.shortenSentence(sent1, "গোলাপ"))
    # print(processor.shortenSentence(sent2, "গোলাপ"))
    # print(processor.shortenSentence(sent3, "গোলাপ"))
    # print(processor.shortenSentence(sent4, "গোলাপ"))
    # print(processor.shortenSentence(sent5, "গোলাপ"))

    # print(re.sub("গোলাপ", "কাঠগোলাপ", sent4))
