from textblob import TextBlob
import pickle
import MySQLdb as mdb
import csv

print "hello"
def split_into_tokens(review):
    review = unicode(review, 'utf8')  
    return TextBlob(review).words

def split_into_lemmas(review):
    review = str(review)
    review = unicode(review, 'utf8').lower()
    words = TextBlob(review).words 
    return [word.lemma for word in words]
 
nb_detector_filename = "train_2novagain.pkl"

nb_detector= pickle.load(open(nb_detector_filename,'rb'))
 

db = mdb.connect(host="localhost" , user="root", passwd="pragya", db="test")
cursor = db.cursor()

cursor.execute("select coalesce(likes,dislikes) from new_review;")
rows = cursor.fetchall()
csv = open("filter_review.csv", "w") 
columnTitleRow = "label,reviewText \n"
csv.write(columnTitleRow)

for review in rows:
    prediction =  nb_detector.predict(review)[0]
    #print r
    result = prediction + "," + str(review) +"\n"
    csv.write(result)
     
 
"""for i in range(100):
    r = str(raw_input("\nenter text to test review\n"))
    print nb_detector.predict([r])[0]
    #print nb_detector.predict_proba([r])[0]"""
    
print "done"

