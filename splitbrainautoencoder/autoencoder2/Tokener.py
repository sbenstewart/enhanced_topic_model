from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.data

class Pyxter():

    """This Class contains implementation for Term Frequency"""

    def __init__(self, f):
	
        # this private var contains tokenized terms.
        self.__text = []

        # nltk module for tokenizing
        tk = nltk.data.load('tokenizers/punkt/english.pickle')
        rawData = "Biomedical disease research summary"     # Query Text
        rawData += f.read()
        self.__text = tk.tokenize(rawData)

        # stop words for tf-idf module
        self.__stopwords = []
        with open('stopwords/long.txt', 'r') as sw:
            self.__stopwords.extend(sw.read().split())

    def tf(self):

        """Returns the tf-idf document-term matrix"""

        tfidf = TfidfVectorizer(stop_words=self.__stopwords, max_features=200, smooth_idf=True)
        self.__X = tfidf.fit_transform(self.__text)

        return self.__X.toarray().transpose()

    def printSentence(self, indexList, file=None, outFileName=None):

        """Prints ranked sentences to stdout or specified file"""

        if file is None:

            # Print to stdout
            print("Summary")
            for index in indexList:
                print(self.__text[index])

        else:

            # Prints to the file

            file.write("--------------------------------\n")
            file.write(outFileName+'\n')
            file.write("--------------------------------\n")

            for index in indexList:
                file.write(self.__text[index])

            file.write('\n--------------------------------\n')
            print("Summary written to file")

    def test(self):
        """Now printing Tf - Idf values"""
        for i in self.__text:
            print(i)


if __name__ == "__main__":
    with open("data.txt", 'r') as f:
        pyxter = Pyxter(f)
        pyxter.test()
