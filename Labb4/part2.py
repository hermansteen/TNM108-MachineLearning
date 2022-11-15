from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_files
import nltk

moviedir = "./movie_reviews"
movie = load_files(moviedir, shuffle=True)
len(movie.data)
movie.target_names
movie.data[0][:500]
movie.filenames[0]
movie.target[0]

docs_train, docs_test, y_train, y_test = train_test_split(
    movie.data, movie.target, test_size=0.20, random_state=12)
movieVzer = CountVectorizer(
    min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
docs_train_counts = movieVzer.fit_transform(docs_train)

docs_train_counts.shape
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)
docs_train_tfidf.shape

docs_test_count = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_count)

clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

y_pred = clf.predict(docs_test_tfidf)
sklearn.metrics.accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(
    reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(
    reviews_new_counts)  # turn into tfidf vector

pred = clf.predict(reviews_new_tfidf)
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

# create pipeline for vectorizer, tfidf, and classifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf.fit(docs_train, y_train)  # train on training set
# use grid search to find best parameters
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train, y_train)
print(gs_clf.best_score_)
print(gs_clf.best_params_)
# use best parameters to predict
y_pred = gs_clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
# use best parameters to predict new reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']
pred = gs_clf.predict(reviews_new)
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))
