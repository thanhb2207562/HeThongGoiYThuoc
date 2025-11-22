import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from scripts.recommend import compute_stats
import warnings
warnings.filterwarnings('ignore')

PROCESSED = 'D:/NIENLUAN/drug-recommender/data/raw/drugsComTrain_clean.csv'
MODELS_DIR = 'data/models'
STATS_DIR = 'data/stats'

def prepare_text_features(df):
    texts = df['condition'].fillna('') + ' ||| ' + df['review'].fillna('')
    return texts.tolist()

def train_classifier(df_train, df_test):
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train = prepare_text_features(df_train)
    X_test = prepare_text_features(df_test)
    y_train = (df_train['rating'] >= 6).astype(int)
    y_test = (df_test['rating'] >= 6).astype(int)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(f"Classifier F1: {f1:.4f}, Acc: {acc:.4f}")

    joblib.dump(vectorizer, os.path.join(MODELS_DIR,'tfidf_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(MODELS_DIR,'logistic_classifier.joblib'))

def build_stats(df):
    os.makedirs(STATS_DIR, exist_ok=True)
    stats = compute_stats(df, min_reviews=1)
    stats.to_csv(os.path.join(STATS_DIR,'stats_condition_drug.csv'), index=False)
    print('Stats saved.')

def main():
    os.makedirs('data/processed', exist_ok=True)
    df = pd.read_csv(PROCESSED)
    # split if not already split
    train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    train_classifier(train, test)
    build_stats(df)

if __name__ == '__main__':
    main()
