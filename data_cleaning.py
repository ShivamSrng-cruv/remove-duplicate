import re
import time
import spacy
import string
import warnings
import contractions
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')


class DataCleaning:
    def __init__(self) -> None:
        self.__stopwords = stopwords.words("english")
        self.__punctuations = string.punctuation.replace(".", "")
        self.__config = {
            'breakpoint': 7,
            'model': "en_core_web_lg"
        }
        self.__model = spacy.load(self.__config['model'])

    def __remove_html_tags_entities(self, data: str) -> str:
        """
        to remove HTML tags, entities and links if any from the data
        :param data: string with HTML tags, entities and links
        :return: string without HTML tags, entities and links
        """
        data = BeautifulSoup(str(data), "lxml").get_text(strip=True)
        data = data.encode("ascii", "ignore").decode()
        return data.strip()

    def __remove_last_fullstop(self, data: str) -> str:
        """
        to remove the last period if present, because in later stages
        while splitting the data on the basis of period, the last
        period leads to generation of empty string
        :param data: which may have fullstop at end
        :return: string without fullstop at the end
        """
        data = self.__remove_html_tags_entities(data)
        if data[-1] == '.':
            data = data[:-1]
            return data
        else:
            return data

    def __remove_links(self, data: str) -> str:
        """
        to remove any website links that might get scrapped with text
        :param data: string with or without links
        :return: string without links
        """
        data = self.__remove_last_fullstop(data)
        data = re.sub(r"(http(s)?:\/\/.)+(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}(\.)*[ a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/=]*)", '', data)
        return data

    def __general_preprocessing(self, data: str) -> str:
        """
        to remove any \n, \t, ]r characters
        :param data: string with any irrelevant characters
        :return: string without any irrelevant characters
        """
        data = self.__remove_links(data)
        data = re.sub(r"(\n)+", ".", data)
        data = re.sub("[\t\r]", ' ', data)
        data = re.sub("(  )+", " ", data)
        return data

    def __retain_fullstops_within_quotes(self, data: str) -> str:
        """
        to retain the fullstops without the double quotes
        :param data: string with fullstops inside double quotes
        :return: string in which fullstops inside double quotes are replaced with ''
        """
        data = self.__general_preprocessing(data)
        return re.sub(r'"[^"]*"', lambda m: m.group(0).replace('.', '') + ".", data)

    def __remove_contractions(self, data: str) -> str:
        """
        to convert contractions to expanded words
        :param data: string with contractions, ex: "It's"
        :return: string without contractions, ex: "It's" -> "It is"
        """
        data = self.__retain_fullstops_within_quotes(data)
        return " ".join([contractions.fix(i) for i in data.split(" ")])

    def __remove_empty_sentences(self, data: str):
        """
        to remove any sentence that has length zero, or it's just more than 1 consecutive fullstops
        :param data: string with or without consecutive fullstops
        :return: string without empty sentence
        """
        data = self.__remove_contractions(data)
        data = re.sub(" \.", ".", data)
        data = re.sub("\. ", ".", data)
        data = re.sub(" \. ", ".", data)
        data = re.sub("(\.)+", ".", data)
        return ".".join(sentence for sentence in data.split(".") if len(sentence) != 0)

    def __remove_punctuations(self, data: str) -> str:
        """
        to remove punctuations
        Note: Here, we are neither inserting nor removing any fullstop. Therefore, no. of sentences
        in clean data as well as structured data remains same after applying the pre-processing.
        :param data: string which has punctuations
        :return: string without punctuations
        """
        data = data.lower()
        return data.translate(data.maketrans('', '', self.__punctuations))

    def __remove_stopwords(self, data: str) -> str:
        """
        to remove stopwords from the sentence
        :param data: string with stopwords, e.g.: 'This is great'
        :return: string without stopwords, e.g.: 'This great'
        """
        data = self.__remove_punctuations(data)
        return " ".join([i for i in data.split(" ") if i not in self.__stopwords])

    def clean_data_nf(self, data: str) -> str:
        """
        to perform the appropriate preprocessing on the data
        :param data: raw data
        :return: clean data which can be used for further processing
        """
        data = self.__remove_empty_sentences(data)
        return data

    def structured_data(self, data: str) -> str:
        """
        to remove the punctuations and stopwords from the cleaned data
        :param data: cleaned data
        :return: structured data
        """
        data = self.__remove_stopwords(data)
        return data

    def __sentence_formatting(self, data: str) -> str:
        """
        to merge the sentences that are much smaller than the breakpoint value according to conditions
        :param data: string with fullstops inside double quotes
        :return: a proper formatted string
        """
        flag, sentences, data = 0, [], data.split(".")
        breakpoint = self.__config['breakpoint']
        for i in range(len(data)):
            if flag == 1:
                flag = 0
                continue
            sentence = data[i]
            total_words = len(sentence.split(" "))
            if total_words > breakpoint and sentence not in sentences:
                sentences.append(sentence)
            elif 0 < total_words <= breakpoint and sentence not in sentences:
                if i == 0:
                    sentences.append(data[i] + " " + data[i + 1])
                elif i == len(data) - 1:
                    sentences.append(data[i - 1] + " " + data[i])
                elif len(data[i - 1].split(" ")) <= len(data[i + 1].split(" ")):
                    sentences.pop(-1)
                    sentences.append(data[i - 1] + " " + data[i])
                elif len(data[i - 1].split(" ")) > len(data[i + 1].split(" ")):
                    sentences.append(data[i] + " " + data[i + 1])
                    flag = 1
        return '.'.join(sentences)

    def __sentence_boundary_detection_model(self, data: str) -> str:
        """
        to add period at appropriate positions in any sentence
        :param data: string without period
        :return: string with period at appropriate locations
        """
        doc = self.__model(data)
        sentences = list(doc.sents)
        data = ""
        for i in sentences:
            data += str(i).strip() + "."
        sentences = self.__sentence_formatting(data)
        data = []
        for i in sentences.split("."):
            if len(i) != 0:
                data.append(i)
        sentences = ".".join(data)
        sentences = re.sub("(  )+", " ", sentences)
        return sentences

    def sentence_boundary_detection(self) -> None:
        """
        to add period at appropriate positions in any sentence
        """
        for no_of_sentence in range(1, 11):
            indices = df[df['no_of_sentence'] == no_of_sentence].index
            for i in indices:
                data = df['clean_data'][i]
                df['clean_data'][i] = self.__sentence_boundary_detection_model(data)
                df['structured_data'][i] = self.structured_data(df['clean_data'][i])
            df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))

    def remove_unnecessary_fullstops(self) -> None:
        """
        to remove any unnecessary period from positions in any sentence
        """
        max_sentences_in_a_row = df.sort_values(by='no_of_sentence').max()[2] + 1
        for no_of_sentence in range(max_sentences_in_a_row, 29, -1):
            indices = df[df['no_of_sentence'] == no_of_sentence].index
            for i in indices:
                data = df['clean_data'][i]
                df['clean_data'][i] = self.__sentence_formatting(data)
                df['structured_data'][i] = self.structured_data(df['clean_data'][i])
            df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))

    def clean_data_f(self, data: str) -> str:
        """
        to perform the final cleaning of data, after all the pre-processing
        :param data: string with period at appropriate location
        :return: clean data
        """
        return self.__sentence_formatting(data)

    def sentence_boundary_detection_text(self, clean_data) -> tuple[str, str]:
        """
        to add period at appropriate positions in any sentence
        (this function is used later)
        :param clean_data: string with pre-processing
        :return: string with period at appropriate locations
        """
        clean_data = self.__sentence_formatting(clean_data)
        structured_data = self.structured_data(clean_data)
        no_of_sentence = structured_data.split(".")
        if len(no_of_sentence) < 6:
            while True:
                prev = len(structured_data.split("."))
                clean_data = self.__sentence_boundary_detection_model(clean_data)
                structured_data = self.structured_data(clean_data)
                end = len(structured_data.split("."))
                if prev == end or end > 6:
                    clean_data = self.__sentence_formatting(clean_data)
                    structured_data = self.structured_data(clean_data)
                    return clean_data, structured_data
        else:
            return clean_data, structured_data


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("./news.csv")
    df.head()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(['title', 'published_at', 'source', 'topic'], axis=1, inplace=True)
    dc = DataCleaning()
    df['clean_data'] = df['content'].apply(dc.clean_data_nf)
    df['structured_data'] = df['clean_data'].apply(dc.structured_data)
    df.drop(['content'], axis=1, inplace=True)
    df['no_of_sentence'] = df['structured_data'].apply(lambda x: len(x.split(".")))
    dc.sentence_boundary_detection()
    dc.remove_unnecessary_fullstops()
    df['clean_data'] = df['clean_data'].apply(dc.clean_data_f)
    df['structured_data'] = df['clean_data'].apply(dc.structured_data)
    df.drop(['no_of_sentence'], axis=1, inplace=True)
    df.to_csv('./clean_and_structured_news.csv', index=False)
    print(f"Total time taken in complete program execution: {int((time.time() - start) // 60)} mins {int((time.time() - start) % 60)} secs")
