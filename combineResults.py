import os
import pandas as pd

def findResultFolders(directoryPath, filename):
    resultFolders = []
    for root, dirs, files in os.walk(directoryPath):
        if filename in files:
            resultFolders.append(root)
    return resultFolders

directory_path = './'
file_name = 'sentences.csv'
folders_with_file = findResultFolders(directory_path, file_name)
print(folders_with_file)

def create_word_sentence_dict(sentences_file, words_file):
    # Read sentences.csv into a DataFrame
    sentences_df = pd.read_csv(sentences_file)

    # Read words.csv into a DataFrame
    words_df = pd.read_csv(words_file)

    # Create an empty dictionary to store the word-sentence mapping
    word_sentence_dict = {}

    # Iterate through each row in words_df
    for index, row in words_df.iterrows():
        word = row['WEAT word']
        print(row['Sentences'])
        sentence_ids = row['Sentences'].split('-')

        # Get the corresponding sentences from sentences_df using sentence_ids
        sentences_list = [sentences_df.loc[int(sid)]['Sentence'] for sid in sentence_ids]

        # Add the word-sentence mapping to the dictionary
        word_sentence_dict[word] = sentences_list

    return word_sentence_dict

sentences_file = 'porjotonlipi_3/sentences.csv'
words_file = 'porjotonlipi_3/weatWordsSentences.csv'
word_sentence_dict = create_word_sentence_dict(sentences_file, words_file)
print(word_sentence_dict)

# for folder in folders_with_file:
#     sentences = pd.read_csv(os.path.join(folder, "sentences.csv"))
#     weatWords = pd.read_csv(os.path.join(folder, "weatWordsSentences.csv"))