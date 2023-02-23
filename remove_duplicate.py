import pickle
import numpy as np
from data_cleaning import DataCleaning


class RemoveDuplicates:
    def __init__(self):
        self.__DC = DataCleaning()
        self.__config = {
            'model': "en_core_web_sm",
            'threshold_first_sentence': 2.225,
            'threshold_last_sentence': 2.21,
            'threshold_in_between_sentence': 2.133,
            'breakpoint': 7
        }
        with open('Vectorizer.pkl', 'rb') as file:
            self.__vectorizer, self.__maxidf = pickle.load(file)

    def __transform(self, data: str, clean_data: str) -> str:
        """
        to transform the data that is provided into their own sentence embeddings
        :param data: input string that is provided
        :param clean_data: input string of user but with appropriate period
        :return: duplicate sentences
        """
        duplicate_sentence, res = "", []
        idf_values, max_idf_value = self.__vectorizer, self.__maxidf
        total_sentences = len(data.split("."))
        for i in range(total_sentences):
            sentence = data.split(".")[i]
            clean_sentence = clean_data.split(".")[i]
            avg_idf = 0
            for j in range(len(sentence.split(" "))):
                word = sentence.split(" ")[j]
                if word in idf_values.keys():
                    avg_idf += np.sqrt(idf_values[word])
                else:
                    avg_idf += np.sqrt(max_idf_value)
            avg_idf = avg_idf / len(sentence.split(" "))
            if i == 0 and avg_idf < self.__config['threshold_first_sentence']:
                duplicate_sentence += clean_sentence + "."
            elif i == total_sentences - 1 and avg_idf < self.__config['threshold_last_sentence']:
                duplicate_sentence += clean_sentence + "."
            elif avg_idf < self.__config['threshold_in_between_sentence']:
                duplicate_sentence += clean_sentence + "."
            res.append([sentence, clean_sentence, avg_idf, i])
        return duplicate_sentence

    def getDuplicates(self, data: str) -> tuple[str, str]:
        """
        this function finds out the duplicate sentence from the data
        :param data: string from which the duplicate sentences have to be removed
        :return: duplicates which are the duplicate sentence and actual data without duplicate sentence
        """
        clean_data = self.__DC.clean_data_nf(data)
        clean_data, structured_data = self.__DC.sentence_boundary_detection_text(clean_data)

        duplicates = self.__transform(structured_data, clean_data)
        no_of_sentences = len(clean_data.split("."))
        without_duplicates = ""
        for i in range(no_of_sentences):
            sentence = clean_data.split(".")[i].strip()
            if sentence not in duplicates.split("."):
                without_duplicates += sentence + ". "
        return duplicates, without_duplicates


if __name__ == "__main__":
    rd = RemoveDuplicates()
    text = input("Enter text: ")
    duplicate, without_duplicate = rd.getDuplicates(text)
    print(duplicate)
    print(without_duplicate)
