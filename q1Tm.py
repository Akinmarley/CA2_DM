import re, nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import joblib as joblib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
 

def preprocess_text(MedReviews): #### Cleaning Medreviews
    soup = BeautifulSoup(MedReviews, 'lxml')   # removing HTML encoding such as ‘&amp’,’&quot’
    souped = soup.get_text()
    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t3])|(\w+:\/\\\S+)"," ", souped) # removing @mentions, hashtags, urls.

 

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/library/re.html
    https://www.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html
    """
    tokens =nltk.word_tokenize(only_words)
    removed_characters = [word for word in tokens if len(word)>2] # removing words with length less than or equal to 2
    lower_case = [l.lower() for l in removed_characters]

 

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

 

    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_vector = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemm_vector

 

def Cross_validation(data, targets, tfidf, clf_cv, model_name): #### Performs cross-validation on SVC

 

    kf = KFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
    scores=[]
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data[train_index], targets[train_index]
        data_test_cv, targets_test_cv = data[test_index], targets[test_index]
        data_train_list.append(data_train_cv) # appending training data for each iteration
        data_test_list.append(data_test_cv) # appending test data for each iteration
        targets_train_list.append(targets_train_cv) # appending training targets for each iteration
        targets_test_list.append(targets_test_cv) # appending test targets for each iteration
        tfidf.fit(data_train_cv) # learning vocabulary of training set
        data_train_tfidf_cv = tfidf.transform(data_train_cv)
        print("Shape of training data: ", data_train_tfidf_cv.shape)
        data_test_tfidf_cv = tfidf.transform(data_test_cv)
        print("Shape of test data: ", data_test_tfidf_cv.shape)
        clf_cv.fit(data_train_tfidf_cv, targets_train_cv) # Fitting SVC
        score = clf_cv.score(data_test_tfidf_cv, targets_test_cv) # Calculating accuracy
        scores.append(score) # appending cross-validation accuracy for each iteration
    print("List of cross-validation accuracies for {}: ".format(model_name), scores)
    mean_accuracy = np.mean(scores)
    print("Mean cross-validation accuracy for {}: ".format(model_name), mean_accuracy)
    print("Best cross-validation accuracy for {}: ".format(model_name), max(scores))
    max_acc_index = scores.index(max(scores)) # best cross-validation accuracy
    max_acc_data_train = data_train_list[max_acc_index] # training data corresponding to best cross-validation accuracy
    max_acc_data_test = data_test_list[max_acc_index] # test data corresponding to best cross-validation accuracy
    max_acc_targets_train = targets_train_list[max_acc_index] # training targets corresponding to best cross-validation accuracy
    max_acc_targets_test = targets_test_list[max_acc_index] # test targets corresponding to best cross-validation accuracy

 

    return mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test,scores

 
def visualize_results(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, clf, model_name): #### Creates Confusion matrix for SVC
    tfidf.fit(max_acc_data_train)
    max_acc_data_train_tfidf = tfidf.transform(max_acc_data_train)
    max_acc_data_test_tfidf = tfidf.transform(max_acc_data_test)
    clf.fit(max_acc_data_train_tfidf, max_acc_targets_train) # Fitting SVC
    targets_pred = clf.predict(max_acc_data_test_tfidf) # Prediction on test data
    conf_mat = confusion_matrix(max_acc_targets_test, targets_pred)
    d={-1:'Negative', 0: 'Neutral', 1: 'Positive'}
    sentiment_df = targets.drop_duplicates().sort_values()
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=sentiment_df.values, yticklabels=sentiment_df.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Accuracy) - {}".format(model_name))
    plt.show()
    return
 
def SVC_Save(data, targets, tfidf):
    tfidf.fit(data) # learn vocabulary of entire data
    data_tfidf = tfidf.transform(data)
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary_SVC.csv', header=False)
    print("Shape of tfidf matrix for saved SVC Model: ", data_tfidf.shape)
    clf = LinearSVC().fit(data_tfidf, targets)
    joblib.dump(clf, 'svc.sav')
    return tfidf



def main():
#### Reading training dataset as dataframe

    df = pd.read_csv("MedReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
### # Normalizing tweets
    df['normalized_MedReviews'] = df.Review.apply(preprocess_text)
    df = df[df['normalized_MedReviews'].map(len) > 0] # removing rows with normalized tweets of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned tweets....")
    print(df[['Review','normalized_MedReviews']].head())
    df.drop(['Medicine', 'Condition', 'Review'], axis=1, inplace=True)
    #### Saving cleaned tweets to csv
    df.to_csv('cleaned_data.csv', encoding='utf-8', index=False)
    #### Reading cleaned tweets as dataframe
    cleaned_data = pd.read_csv("cleaned_data.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_MedReviews
    targets = cleaned_data.Rating
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=30, norm='l2', ngram_range=(1,3)) # min_df=30 is a clever way of feature engineering

    NBC_clf = MultinomialNB() # NBC Model
    NBC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test,nbc_scores = Cross_validation(data, targets, tfidf, NBC_clf, "NBC") # NBC cross-validation
    visualize_results(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, NBC_clf, "NBC") # NBC confusion matrix

    SVC_clf = LinearSVC() # SVC Model
    SVC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test,svm_scores = Cross_validation(data, targets, tfidf, SVC_clf, "SVC") # SVC cross-validation
    visualize_results(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, SVC_clf, "SVC") # SVC confusion matrix
    
    if SVC_mean_accuracy > NBC_mean_accuracy:
        SVC_Save(data, targets, tfidf)
    else:
        NBC_Save(data, targets, tfidf)

 
if __name__ =='__main__':
    main()


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################


##model deployment



#import tweepy

#import re, nltk
#import time
#import numpy as np
#import pandas as pd
#import csv
#from bs4 import BeautifulSoup
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib

##def twitter_api():
##    CONSUMER_KEY = "Your Key Here"
##    CONSUMER_SECRET = "Your Secret Here"
##    ACCESS_TOKEN = "Your Token Here"
##    ACCESS_TOKEN_SECRET = "Your Token Secret Here"
##    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
##    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
##    return tweepy.API(auth)

##def twitter_search(api, query, max_tweets):
##    tweets = []
##    initial_tweets = api.search(q=query, lang='en', count=100, tweet_mode = "extended", truncated = False)
##    tweets.extend(initial_tweets)
##    max_id = initial_tweets[-1].id
##    tweet_counter = 0
##    while len(tweets) < max_tweets:
##        new_tweets = api.search(q=query, lang='en', count=100, tweet_mode = "extended", truncated = False, max_id=str(max_id-1))
##        if not new_tweets:
##            break
##        else:
##            tweets.extend(new_tweets)
##            max_id = new_tweets[-1].id
##            tweet_counter += len(new_tweets)
##            if tweet_counter > 15000:
##                print("sleeping for 16 minutes")
##                time.sleep(16*60)
##                tweet_counter = 0

##    print("Number of tweets retrieved: ", len(tweets))
##    return tweets

##def create_csv(query, outtweets):
##    with open('%s_tweets.csv' % query, 'w', newline='') as f:
##        writer = csv.writer(f)
##        writer.writerow(["id","created_at","tweets"])
##        writer.writerows(outtweets)

#def normalizer(NoRatings):
#    soup = BeautifulSoup(NoRatings, 'lxml')   # removing HTML encoding such as ‘&amp’,’&quot’
#    souped = soup.get_text()
#    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)"," ", souped) # removing @mentions, hashtags, urls

#    tokens = nltk.word_tokenize(only_words)
#    removed_letters = [word for word in tokens if len(word)>2]
#    lower_case = [l.lower() for l in removed_letters]

#    stop_words = set(stopwords.words('english'))
#    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

#    wordnet_lemmatizer = WordNetLemmatizer()
#    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
#    return lemmas

#def main():
#    #### Loading the saved model
#    model = joblib.load('svc.sav')
#    vocabulary_model = pd.read_csv('vocabulary_SVC.csv', header=None)
#    vocabulary_model_dict = {}
#    for i, word in enumerate(vocabulary_model[0]):
#         vocabulary_model_dict[word] = i
#    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary = vocabulary_model_dict, min_df=5, norm='l2', ngram_range=(1,3)) # min_df=5 is clever way of feature engineering
    

    
#    NoRatings_df = pd.read_csv('NoRatings.csv', encoding = "ISO-8859-1")
#    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
#    #### Normalizing retrieved tweets
#    NoRatings_df['normalized_Review'] = NoRatings_df.Review.apply(normalizer)
#    NoRatings_df = NoRatings_df[NoRatings_df['normalized_Review'].map(len) > 0] # removing rows with normalized tweets of length 0
#    print("Number of Reviews: ", NoRatings_df.normalized_Review.shape[0])
#    print(NoRatings_df[['Review','normalized_Review']].head())
#    #### Saving cleaned tweets to csv file
#    NoRatings_df.drop(['Medicine', 'Condition'], axis=1, inplace=True)
#    NoRatings_df.to_csv('cleaned_Review.csv', encoding='utf-8', index=False)
#    cleaned_Review = pd.read_csv("cleaned_Review.csv", encoding = "ISO-8859-1")
#    pd.set_option('display.max_colwidth', -1)
#    cleaned_Review_tfidf = tfidf.fit_transform(cleaned_Review['normalized_Review'])
#    targets_pred = model.predict(cleaned_Review_tfidf)
#    #### Saving predicted sentiment of tweets to csv
#    cleaned_Review['predicted_sentiment'] = targets_pred.reshape(-1,1)
#    cleaned_Review.to_csv('predicted_sentiment.csv', encoding='utf-8', index=False)

#if __name__ == "__main__":
#    main()

