import pandas as pd
import json
from collections import Counter

sent_topic = pd.read_csv("./data/Sentiment-topic-test.tsv", sep="\t")

print("=== Sentiment-topic-test.tsv ===")
print("Number of sentences:", len(sent_topic))
print("\nSentiment distribution:")
print(sent_topic["sentiment"].value_counts())
print("\nTopic distribution:")
print(sent_topic["topic"].value_counts())

ner = pd.read_csv("./data/NER-test.tsv", sep="\t")

print("\n=== NER-test.tsv ===")
print("Number of sentences:", ner["sentence id"].nunique())
print("Number of tokens:", len(ner))
print("\nBIO label distribution:")
print(ner["BIO NER tag"].value_counts())

spans = []

for sent_id, group in ner.groupby("sentence id"):
    current_type = None
    current_tokens = []

    for _, row in group.iterrows():
        tag = row["BIO NER tag"]
        token = row["token"]

        if tag == "O":
            if current_type is not None:
                spans.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []
        else:
            prefix, entity_type = tag.split("-", 1)

            if prefix == "B" or entity_type != current_type:
                if current_type is not None:
                    spans.append((current_type, " ".join(current_tokens)))

                current_type = entity_type
                current_tokens = [token]
            else:
                current_tokens.append(token)

    if current_type is not None:
        spans.append((current_type, " ".join(current_tokens)))

print("\nEntity span counts:")
print(Counter([entity_type for entity_type, _ in spans]))

with open("./data/my_tweets.json", "r") as file:
    tweets = json.load(file)

if isinstance(tweets, dict):
    records = tweets.values()
else:
    records = tweets

tweet_labels = [item["sentiment_label"] for item in records]

print("\n=== my_tweets.json ===")
print("Number of training tweets:", len(tweet_labels))
print("Sentiment label distribution:")
print(Counter(tweet_labels))
