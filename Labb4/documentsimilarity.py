from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math
d1 = "The sky is blue"
d2 = "The sun is bright"
d3 = "The sun in the sky is bright"
d4 = "We can see the shining sun, the bright sun"
Z = (d1, d2, d3, d4)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

# take the angle between the first and third document
angle_in_radians = math.acos(cos_similarity.item(2))
print(math.degrees(angle_in_radians))

data = fetch_20newsgroups()
print(data.target_names)

my_categories = ["rec.sport.baseball",
                 "rec.motorcycles", "sci.space", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=my_categories)
test = fetch_20newsgroups(subset="test", categories=my_categories)

print(len(train.data))
print(len(test.data))
print(train.data[9])

cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB().fit(X_train_tfidf, train.target)

docs_new = ['Pierangelo is a really good baseball player', 'Maria rides her motorcycle', 'OpenGL on the GPU is fast',
            'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))
