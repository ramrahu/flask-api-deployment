import re
import pickle
import collections
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
ps = PorterStemmer()


def clean_reviews(review):
  review = review.lower()
  review = re.sub('[^a-zA-Z]', ' ', review)
  return review


def stemming(review):
  review = [ps.stem(word) for word in review]
  return review


def remove_stopwords(review):
  review = review.split()
  review = [word for word in review if not word in set(stopwords.words('english'))]
  review = stemming(review)
  return review


def feature_extraction(final_corpus):
    tf_idf = TfidfVectorizer()
    model = pickle.load(open('models/tfidfvec.pkl','rb'))
    features = model.transform(final_corpus['reviews_text']).toarray()
    return features


def preprocess_text(reviews):
    preprocessed_reviews = []
    cleaned_reviews = reviews.apply(clean_reviews)
    cleaned_reviews.tolist()
    for review in cleaned_reviews:
        preprocessed_reviews.append(' '.join(remove_stopwords(review)))
    final_corpus = pd.DataFrame(preprocessed_reviews)
    final_corpus.columns = ['reviews_text']
    return final_corpus


def get_recommend(user_id):
    user_ratings = pd.read_csv('models/user-user-final-ratings.csv')
    user_ratings = user_ratings.set_index('id')
    recommendations = user_ratings.loc[user_id]
    recommendations = recommendations.sort_values(ascending=False)[0:20]
    return recommendations


def sentiment_based_filtering(recommendations):
    final_recommendations = {}
    model = pickle.load(open('models/logreg.pkl','rb'))
    prod_dataset = pd.read_csv('models/sample30.csv')
    new_prod_dataset = prod_dataset[['name', 'reviews_text']].copy()
    recommendations = recommendations.to_frame().reset_index()
    recommendations.rename(columns = {'index':'product'}, inplace = True)
    new_prod_dataset = new_prod_dataset.groupby(new_prod_dataset['name'])
    for product in recommendations['product']:
        reviews = new_prod_dataset.get_group(product)['reviews_text']
        final_corpus = preprocess_text(reviews)
        features = feature_extraction(final_corpus)
        result = model.predict(features)
        counter = collections.Counter(result)
        counter = dict(counter)
        if 1 in counter.keys() and 0 in counter.keys():
            final_recommendations[product] = counter[1]
    #return result
    sorted_final_recommendations = collections.OrderedDict(final_recommendations)
    return list(sorted_final_recommendations.keys())[0:5]


if __name__=='__main__':
    recommendations = get_recommend('AV13O1A8GV-KLJ3akUyj')
    filtered_recommendations = sentiment_based_filtering(recommendations)
    print(filtered_recommendations)