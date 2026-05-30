import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def load_test(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_tweet_train(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    labels = []
    # my_tweets.json may be a dict of records (id -> {..}) or a list
    records = data.values() if isinstance(data, dict) else data
    for item in records:
        if not isinstance(item, dict):
            continue
        texts.append(item.get("text_of_tweet") or item.get("text") or "")
        labels.append(item.get("sentiment_label"))
    pairs = [(t, l) for t, l in zip(texts, labels) if t and l]
    if not pairs:
        return [], []
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def simple_preprocess(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s'\.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def train_sentiment_classifier(texts, labels):
    texts = [simple_preprocess(t) for t in texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    clf.fit(X, labels)
    return vec, clf


def rule_based_topic(text):
    t = text.lower()
    if any(w in t for w in ["movie", "film", "shot", "actor", "played", "cinema"]):
        return "movie"
    if any(w in t for w in ["book", "novel", "author", "austen", "novels", "read"]):
        return "book"
    if any(w in t for w in ["restaurant", "diner", "eat", "food", "menu", "place to eat"]):
        return "restaurant"
    return "movie"


def evaluate_sentiment(vec, clf, df):
    texts = [simple_preprocess(t) for t in df["text"].fillna("")]
    X = vec.transform(texts)
    preds = clf.predict(X)
    gold = df["sentiment"].values
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")
    report = classification_report(gold, preds, zero_division=0)
    labels = ["negative", "neutral", "positive"]
    try:
        cm = confusion_matrix(gold, preds, labels=labels)
    except Exception:
        cm = confusion_matrix(gold, preds)
    return preds, acc, f1, report, cm


def evaluate_topic(df):
    preds = [rule_based_topic(t) for t in df["text"].fillna("")]
    gold = df["topic"].values
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")
    report = classification_report(gold, preds, zero_division=0)
    labels = sorted(list(set(gold)))
    cm = confusion_matrix(gold, preds, labels=labels)
    return preds, acc, f1, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=False, help="Path to Sentiment-topic-test.tsv")
    parser.add_argument(
        "--tweets",
        default="lab_sessions/lab3/my_tweets.json",
        help="Path to tweet training JSON (from lab sessions)",
    )
    parser.add_argument("--out", default="results/traditional_baseline_results.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # If --test not provided, try to find the test file in common locations
    def find_test_file():
        candidates = []
        # project-relative
        candidates.append(os.path.join(os.getcwd(), "Sentiment-topic-test.tsv"))
        # user's Downloads
        user_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        candidates.append(os.path.join(user_downloads, "Sentiment-topic-test.tsv"))
        candidates.append(os.path.join(user_downloads, "text_mining_project", "Sentiment-topic-test.tsv"))
        candidates.append(os.path.join(user_downloads, "text_mining_for_AI_group3-main", "Sentiment-topic-test.tsv"))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    test_path = args.test if args.test else find_test_file()
    if not test_path:
        raise RuntimeError("Test file not provided and could not be found. Pass --test <path> or place Sentiment-topic-test.tsv in Downloads or project root.")

    test = load_test(test_path)

    # locate tweet training file if not provided or missing
    def find_tweets_file(provided):
        candidates = []
        if provided:
            candidates.append(provided)
        # project-relative
        candidates.append(os.path.join(os.getcwd(), "lab_sessions", "lab3", "my_tweets.json"))
        # known repo location (if script run from other folder)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates.append(os.path.join(repo_root, "lab_sessions", "lab3", "my_tweets.json"))
        # user's Downloads
        user_downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        candidates.append(os.path.join(user_downloads, "my_tweets.json"))
        candidates.append(os.path.join(user_downloads, "text_mining_for_AI_group3-main", "lab_sessions", "lab3", "my_tweets.json"))
        candidates.append(os.path.join(user_downloads, "text_mining_project", "my_tweets.json"))
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    tweets_path = find_tweets_file(args.tweets)
    if not tweets_path:
        raise RuntimeError("Tweet training file not provided and could not be found. Pass --tweets <path> or place my_tweets.json in lab_sessions/lab3 or Downloads.")

    train_texts, train_labels = load_tweet_train(tweets_path)
    if not train_texts:
        raise RuntimeError(f"No training tweets found at {tweets_path}")

    vec, clf = train_sentiment_classifier(train_texts, train_labels)

    sent_preds, sent_acc, sent_f1, sent_report, sent_cm = evaluate_sentiment(vec, clf, test)
    topic_preds, topic_acc, topic_f1, topic_report, topic_cm = evaluate_topic(test)

    # collect error examples for sentiment
    errors = []
    for i, (gold, pred, text) in enumerate(zip(test["sentiment"], sent_preds, test["text"])):
        if gold != pred:
            errors.append((i, text, gold, pred))
        if len(errors) >= 2:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("Sentiment classification (TF-IDF + LogisticRegression)\n")
        f.write(f"Accuracy: {sent_acc:.4f}\n")
        f.write(f"Macro F1: {sent_f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(sent_report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(sent_cm))
        f.write("\n\nTopic classification (rule-based baseline)\n")
        f.write(f"Accuracy: {topic_acc:.4f}\n")
        f.write(f"Macro F1: {topic_f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(topic_report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(topic_cm))
        f.write("\n\nExample sentiment errors (first 2):\n")
        for idx, text, gold, pred in errors:
            f.write(f"Index {idx}: GOLD={gold} PRED={pred} TEXT={text}\n")

    print("Results written to", args.out)


if __name__ == "__main__":
    main()
