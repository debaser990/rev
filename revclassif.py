import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
#from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import model_selection
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")



url = "C:/Users/Akshat/Downloads/reviewPrediction/review_dataset.csv"
names = ['label','review']
reviews = pandas.read_csv(url,names = names)
#print reviews.head()

#print len(reviews.dropna())

X = reviews['review']
Y = reviews['label']

x1 = X.tolist()
y1 = Y.tolist()
#print type(x1)


#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x1, y1, test_size=validation_size, random_state=seed)
#seed = 7

#pipeline = Pipeline([('bow', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])

#print X_train

cv = CountVectorizer(stop_words='english',decode_error = 'ignore')

print x1
try:
    for x in x1:
        vecTrain = cv.fit_transform(x)
except AttributeError :
    print x
      
#transformer = TfidfTransformer()
#t = transformer.fit_transform(i)
print vecTrain
