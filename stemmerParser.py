import os
import re
from typing import List, Dict, Tuple, IO
from abc import ABC, abstractmethod
from wordNormalizer import word_normalize

st = {'া', 'ি', 'ী', 'ে', 'ু', 'ূ', 'ো'}

class StemmerCore(ABC):
    @abstractmethod
    def stem_word(self, word: str):
        pass

class WordDict:
    def __init__(self) -> None:
        me = os.path.realpath(__file__)
        directory = os.path.dirname(me)

        for word in open(os.path.join(directory, 'banglaWords/ner_static.txt'), "r"):
            word = word.replace('\n', '')
            segment = word.split(' ')
            word = segment[:-1]
            for i in word:
                self.__ner_word_dict[i]=1

        for word in open(os.path.join(directory, 'banglaWords/rootword_list.txt'), "r"):
            word=word.replace('\n','')
            self.__rootword_dict[word]=1

        self.__normalizedSearchWords = {}

    def __normalize(self, word:str) -> str:
        if word in self.__normalizedSearchWords:
            return self.__normalizedSearchWords[word]
        
        normalizedWord = word_normalize(word)
        self.__normalizedSearchWords[word] = normalizedWord
        return normalizedWord

    def checkInVocab(self, word: str) -> bool:
        return self.checkInRootWord(word) or self.checkIn_NER_Vocab(word)

    def checkInRootWord(self, word: str) -> bool:
        return (word in self.__rootword_dict) or (self.__normalize(word) in self.__rootword_dict)

    def checkIn_NER_Vocab(self, word: str) -> bool:
        return (word in self.__ner_word_dict) or (self.__normalize(word) in self.__ner_word_dict)


class RafiStemmerRuleParser:
    TAB_AND_SPACE = re.compile(r'\s*')
    COMMENTS = re.compile(r'#.*')
    REPLACE_RULE = re.compile(r'.*->.*')
    LINE_REPLACE_RULE = re.compile(r'->.*')

    lines: List[str]

    groups: List[List[str]]
    replace_rules: Dict[str, str]

    def __init__(self, rules_content: str):
        self.lines = []
        self.groups = []
        self.replace_rules = {}

        self.parse_content(rules_content)
        self.group_rules()

    def group_rules(self):
        group, i = 0, 0
        line_count = len(self.lines)

        # Dear angry Pythonista, this nested bit will be refactored!

        while i < line_count:
            if self.lines[i] == '{':
                self.groups.append([])
                i += 1
                while i < line_count and not self.lines[i] == '}':
                    self.groups[group].append(self.lines[i])
                    i += 1
                group += 1
            i += 1

    def parse_content(self, rules_content):
        for line in rules_content.splitlines():
            try:
                parsed_line, rule = self.parse_line_and_rule(line)

                if parsed_line:
                    self.lines.append(parsed_line)

                if rule:
                    self.replace_rules[parsed_line] = rule

            except ValueError:
                continue

    def parse_line_and_rule(self, line) -> Tuple[str, str]:
        line = line.strip()
        line = self.remove_whitespace(line)
        line = self.remove_comments(line)

        if not line:
            raise ValueError('Not a proper line')

        replace_rule = self.extract_replace_rule(line)
        line = self.LINE_REPLACE_RULE.sub('', line)

        return line, replace_rule

    def extract_replace_rule(self, line: str):
        if self.REPLACE_RULE.fullmatch(line):
            _, suf = line.split('->')
            return suf

    def remove_whitespace(self, line: str):
        return self.TAB_AND_SPACE.sub('', line)

    def remove_comments(self, line: str):
        return self.COMMENTS.sub('', line)


class RafiStemmer(StemmerCore):
    groups: List[List[str]]
    replace_rules: Dict[str, str]

    def __init__(self, wordDict: WordDict, readable_rules: IO[str] = None):
        if readable_rules is None:
            me = os.path.realpath(__file__)
            directory = os.path.dirname(me)

            with open(os.path.join(directory, 'common.rules'), 'rb') as f:
                content = f.read().decode('utf-8')
        else:
            content = readable_rules.read()

        parser = RafiStemmerRuleParser(content)
        self.groups = parser.groups
        self.replace_rules = parser.replace_rules
        self.wordDict = wordDict

    def check(self, word: str):
        word_length = 0

        for c in word:
            if c in st:
                continue
            word_length += 1

        return word_length >= 1

    def stem_with_replace_rule(self, index, replace_prefix, word):
        replace_suffix = self.replace_rules[replace_prefix]
        word_as_list = list(word)
        word_char_idx, current = index, 0

        while word_char_idx < index + len(replace_suffix):

            if replace_suffix[current] != '.':
                word_as_list[word_char_idx] = replace_suffix[current]

            word_char_idx += 1
            current += 1

        return "".join(word_as_list[0:word_char_idx])

    def stem_word(self, word: str):
        if self.wordDict.checkInVocab(word):
            return word
        
        for group in self.groups:
            for replace_prefix in group:

                if not word.endswith(replace_prefix):
                    continue

                index = len(word) - len(replace_prefix)

                if replace_prefix in self.replace_rules:
                    word = self.stem_with_replace_rule(index, replace_prefix, word)  # noqa: E501

                elif self.check(word[0:index]):
                    word = word[0:index]

                break

        return word
