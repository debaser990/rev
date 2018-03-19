import csv
from textblob import TextBlob
import pandas
import pickle
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import sklearn
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
import MySQLdb as mdb

db = mdb.connect(host="localhost" , user="root", passwd="pragya", db="test")
cursor = db.cursor()

reviews = pandas.read_csv('review.csv', names=["label", "message"])
 
def split_into_tokens(review):
    review = str(review)
    review = unicode(review, 'utf8')  
    return TextBlob(review).words

def split_into_lemmas(review):
    review = str(review)
    review = unicode(review, 'utf8').lower()
    words = TextBlob(review).words 
    return [word.lemma for word in words]

#print "Script Start"
print reviews.groupby('label').describe()
 
review_train = reviews['message']
label_train = reviews['label']
print "pipeline"
pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas,stop_words='english')),('tfidf', TfidfTransformer()),   ('classifier', MultinomialNB()),  ])
print "cross validation"
scores = cross_val_score(pipeline,review_train,label_train,cv=10,scoring='accuracy',n_jobs=1,)
print "scores"
print scores
params = {'tfidf__use_idf': (True, False),
          #s'vect__ngram_range': [(1, 1), (1, 2)],
          #'clf__alpha': (1e-2, 1e-3),
'bow__analyzer': (split_into_lemmas, split_into_tokens),
}
print "grid"
grid = GridSearchCV(pipeline,params,refit=True,n_jobs=1,scoring='accuracy',cv=StratifiedKFold(label_train, n_folds=5),)
print "nb_detector"
grid = grid.fit(review_train, label_train)
nb_detector_filename = "train_2novagain.pkl"
nb_detector_pkl = open(nb_detector_filename, 'ab')

pickle.dump(grid , nb_detector_pkl)

nb_detector_pkl.close()

print " script done"    

 

