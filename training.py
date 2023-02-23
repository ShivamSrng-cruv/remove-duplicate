import time
import pickle
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")


class RemoveDuplicates:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.__df = dataframe
        self.__config = {
            'analyzer': 'word',
            'ngram_range': (1, 1),
            'stop_words': 'english',
            'max_features': 2000
        }
        self.__vectorizer = TfidfVectorizer(analyzer=self.__config['analyzer'],
                                            ngram_range=self.__config['ngram_range'],
                                            stop_words=self.__config['stop_words'],
                                            max_features=self.__config['max_features'])

    def __combine_text(self) -> list[str]:
        """
        this function takes 'structured_data' column and converts it into list of string
        :return: list of string of sentences
        """
        self.__corpus = '.'.join(self.__df['structured_data'].to_list())
        return self.__corpus.split(".")

    def train(self) -> None:
        """
        this function, fits and transform tfidf vectorizer using the data under 'structured_data' column
        :return: vectors generated on the basis of data present under 'structured_data' column
        """
        __corpus = self.__combine_text()  # corpus: collection of texts
        self.__vectorizer.fit_transform(__corpus)
        __only_idf, __vocabulary = self.__vectorizer.idf_, self.__vectorizer.vocabulary_
        __idf_values, keys, values = dict(), list(__vocabulary.keys()), list(__vocabulary.values())
        for i in range(len(keys)):
            key, value = keys[i], values[i]
            __idf_values[key] = __only_idf[value]
        max_idf_value = max(list(__idf_values.values())) + 1
        __vectorizer = [__idf_values, max_idf_value]
        with open('Vectorizer.pkl', 'wb') as file:
            pickle.dump(__vectorizer, file)


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("./clean_and_structured_news.csv")
    rd = RemoveDuplicates(df)
    rd.train()
    print(f"Total time taken in complete program execution: {int((time.time() - start) // 60)} mins {int((time.time() - start) % 60)} secs")
