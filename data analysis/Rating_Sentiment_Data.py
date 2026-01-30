import pandas as pd
import numpy as np

# 1.Load Data
df = pd.read_csv("DataForProject/categorized_reduced_combined_cleaned_dataset.csv")

# 2.Rating Based Sentiment for Flipkart
def rating_sentiment(row):
    if str(row['source']).lower() =='flipkart':
        try:
            rating = int(row['rating'])
        except:
            return "unknown"
        if rating in [1,2]:
            return "negative"
        elif rating == 3:
            return "neutral"
        elif rating in [4,5]:
            return "positive"
        else:
            return "unknown"
    else:
        return row.get("sentiment_label", None) #Keep previous sentiment label for Non-Flipkart
     
df['sentiment_label'] = df.apply(rating_sentiment, axis=1)


# 3.Add Random Review Dates (2021-01-01 to 2025-01-01)
start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2025-01-01')


# 4.In Mask, Keep all dates that are NaN or Empty
mask = df['review_date'].isna() | (df['review_date'] == '')

random_dates = pd.to_datetime(
    np.random.randint(
        start_date.value // 10**9,
        end_date.value // 10**9,
        size = mask.sum()
    ),
    unit='s'
)

df.loc[mask, 'review_date'] = random_dates.strftime('%m/%d/%Y')


# 5.Save The Updated Dataframe
df.to_csv("DataForProject/sentiment_categorized_reduced_combined_cleaned_dataset.csv", index=False)

print("Rating-based sentiment analysis completed. Saved as sentiment_categorized_reduced_combined_cleaned_dataset.csv")