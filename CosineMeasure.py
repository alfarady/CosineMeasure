import re
import copy
import math
from collections import Counter
from collections import OrderedDict

class CosineMeasure():
    # constructor for CosineMeasure
    def __init__(self, query, documents):
        self.query = query  # Query to find
        self.dir_path = documents  # Documents

        # Tokenize documents
        self.tokenized_documents = [re.findall(r'\w+', d.lower()) for d in documents]
        self.tokenized_query = re.findall(r'\w+', query.lower())
        self.word_set = sorted(set(sum(self.tokenized_documents, []))) # sorting all words in tokenized document
        self.vector_template = OrderedDict((token, 0) for token in self.word_set) # creating array template
        self.doc_tfidf_vectors = []
        self.query_tfidf_vectors = copy.copy(self.vector_template) # copy template that we made before
        self.is_docs_tokenized = 0

    def find_tfidf(self, is_query):
        if is_query:
            if self.is_docs_tokenized:
                print("=============" + 'Q' + "=============")
                token_counts = Counter(self.tokenized_query)
                for key, value in token_counts.items():
                    docs_containing_key = 0
                    for _doc in self.tokenized_documents:
                        if key in _doc:
                            docs_containing_key += 1
                    if docs_containing_key == 0:
                        continue
                    tf = value
                    idf = math.log10(len(self.tokenized_documents) / docs_containing_key)
                    self.query_tfidf_vectors[key] = tf * idf
                    print("WORD: " + key + " TF: " + str(tf) + " IDF: " + str(idf) + " BOBOT: " + str(tf * idf))
            else:
                print("Error: Need tokenize documents before query -> use find_tfidf(is_query=0) first")

        else:
            index = 1
            for doc_tokens in self.tokenized_documents: # iterate all tokenized documents
                print("============D" + str(index) + "=============")
                vec = copy.copy(self.vector_template) # make a temp variable to save vector
                token_counts = Counter(doc_tokens) # make counter each value of tokenized docs
                for key, value in token_counts.items(): # iterate all item of counter
                    docs_containing_key = 0 # init variable to check containing word inside sentence
                    for _doc_tokens in self.tokenized_documents: # iterate to check containing word inside sentence
                        if key in _doc_tokens:
                            docs_containing_key += 1

                    # measure tf
                    tf = value # used value as tf

                    # measure idf
                    if docs_containing_key:
                        idf = math.log10(len(self.tokenized_documents) / docs_containing_key)  # formula log base 10 (size of documents / freq of occurance word in sentence)
                    else:
                        idf = 0
                    vec[key] = tf * idf
                    print("WORD: " + key + " TF: " + str(tf) + " IDF: " + str(idf) + " BOBOT: " + str(tf * idf))
                index += 1
                self.doc_tfidf_vectors.append(vec)

            self.is_docs_tokenized = 1

    def find_cosin_sim(self, vec1, vec2):
        vec1 = list(vec1.values())
        vec2 = list(vec2.values())

        dot_prod = 0
        for i, v in enumerate(vec1):
            dot_prod += v * vec2[i]
        mag_1 = math.sqrt(sum([x**2 for x in vec1]))
        mag_2 = math.sqrt(sum([x**2 for x in vec2]))
        return dot_prod / (mag_1 * mag_2)

    def print_cosin_sim(self):
        print("============CosineSim=============")
        index = 1
        sorted_vector = {}
        for vectors in self.doc_tfidf_vectors:
            sorted_vector.update({
                f'sim(Q, D{index})': self.find_cosin_sim(self.query_tfidf_vectors, vectors)
            })
            print(f'sim(Q, D{index}): {self.find_cosin_sim(self.query_tfidf_vectors, vectors)}')
            index += 1
        print("\033[93m========CosineSim(Sorted)=========")
        sorted_vector = {k: v for k, v in sorted(sorted_vector.items(), reverse=True, key=lambda item: item[1])}
        for key, value in sorted_vector.items() :
            print (key, value)

        print("\033[0m")

    