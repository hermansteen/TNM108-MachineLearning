from sklearn.feature_extraction.text import CountVectorizer
d1 = "The sky is blue"
d2 = "The sun is bright"
d3 = "The sun in the sky is bright"
d4 = "We can see the shining sun, the bright sun"
Z = (d1, d2, d3, d4)

vectorizer = CountVectorizer()
print(vectorizer)

my_stop_words = {"the", "is"}
my_vocabulary = {"blue": 0, "sun": 1, "bright": 2, "sky": 3}
vectorizer = CountVectorizer(stop_words=my_stop_words, vocabulary=my_vocabulary)
print(vectorizer.stop_words)
print(vectorizer.vocabulary)

smatrix = vectorizer.transform(Z)
print(smatrix)
matrix = smatrix.todense()
print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)
# print idf values
feature_names = vectorizer.get_feature_names()
import pandas as pd
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'], ascending=False)
#tf_idf scores
tf_idf_vector=tfidf_transformer.transform(smatrix)
#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]
# print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
