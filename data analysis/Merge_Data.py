import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# 1.Download Stopwords From nltk Library
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

#print(stop_words)


# 2.Cleaning Function
def clean_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def clean_punctuation(text):
    if isinstance(text, str):
        return re.sub(r"[^\w\s]", "",text)
    return text

def clean_stopwords(text):
    if isinstance(text, str):
        return " ".join([w for w in text.split() if w not in stop_words])
    return text

def clean_whitespace(text):
    if isinstance(text, str):
        return re.sub(r"\s+", " ",text).strip()
    return text

def clean_text(text):
    text = clean_lowercase(text)
    text = clean_stopwords(text)
    text = clean_punctuation(text)
    text = clean_whitespace(text)
    return text


# 3.Loading Files
df_flip = pd.read_csv("DataForProject/flipkart_product.csv", encoding="latin1", engine="python")
df_amazon = pd.read_excel("DataForProject/Amazon_datasheet.xlsx")


# 4.Extra Cleaning Required in the Flipkart Dataset
def clean_text_flip(t):
    t = str(t)
    #remove non-ASCII garbled characters
    t = re.sub(r"[^\x00-\x7F]", " ", t)
    #remove all special chars except letters/numbers/space 
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    #collapse multiple spaces into one
    t = re.sub(r"\s+", " ", t)

    return t.strip()

df_flip = df_flip.map(lambda x: clean_text_flip(x) if isinstance (x, str) else x)


# 5.Standardize Flipkart Dataset Columns
df_flip = df_flip.rename(columns={
    "ProductName" : "product",
    "Review" : "review_title",
    "Summary" : "review_text",
    "Rate" : "rating"
})
df_flip["source"] = "flipkart"
df_flip["review_date"] = ""
df_flip["sentiment_label"] = ""
df_flip["category"] = ""


# 6.Standardize Amazon Dataset Columns
df_amazon = df_amazon.rename(columns={
    "Product Name" : "product",
    "User Review" : "review_text",
    "Star Rating" : "rating",
    "Date of Review" : "review_date",
    "Category" : "category",
    "Sentiment" : "sentiment_label"
})
df_amazon["source"] = "amazon"
df_amazon["review_title"] = ""
df_amazon["source"] = "amazon"


# 8.Keeping Required Columns Only
required = [
    "source",
    "product",
    "review_text",
    "review_title",
    "rating",
    "category",
    "review_date",
    "sentiment_label"
]

df_flip = df_flip[required]
df_amazon = df_amazon[required]


# 9.Combine Both Datesets
df = pd.concat([df_flip, df_amazon], ignore_index=True)


# 10.Cleaning The Review Text
df["cleaned_text"] = df["review_text"].astype(str).apply(clean_text)


# 11.Cleaning Product Name
df["product"] = df["product"].astype(str).apply(clean_text)


# 12.Sentiment Score
df["sentiment_score"] = ""


# 13.Saving the Result
df.to_csv("DataForProject/combined_cleaned_dataset.csv", index=False)

print("Final dataset saved as combined_cleaned_dataset.csv")