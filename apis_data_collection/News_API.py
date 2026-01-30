import requests
import pandas as pd 
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from dotenv import load_dotenv
import notifications.Notification as Notification
from apis_data_collection import sentiment_news_spike

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()


# 1. API Configuration
api_key = os.getenv("News_api_key")
base_url = "https://newsapi.org/v2/everything"

# HOMEWORK
#from_date = datetime.utcnow().date() - timedelta(days=5)
from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
language = "en"
page_size = 50


# 3. Category Keywords
category_keywords = {
    "Electricals_Power_Backup" : ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances" : ["air conditioner", "refrigerator", "washing machine", "air cooler"],
    "Kitchen_Appliances" : ["mixer", "grinder", "microwave", "oven", "juicer"],
    "Furniture" : ["sofa", "bed", "table", "chair"],
    "Home_Storage_Organization" : ["storage box", "wardrobe", "organizer"],
    "Computers_Tablets" : ["laptop", "tablet", "desktop"],
    "Mobile_Accessories" : ["charger", "earphones", "powerbank"],
    "Wearables" : ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment" : ["smart", "speaker", "soundbar"],
    "Networking_Devices" : ["router", "wifi modem"],
    "Toys_Kids" : ["kid's toys", "children's games"],
    "Gardening_Outdoor" : ["gardening", "lawn tools"],
    "Kitchen_Dining" : ["cookware", "utensils"],
    "Mens_Clothing" : ["mens clothing", "mens fashion"],
    "Footwear" : ["shoes", "sneakers"],
    "Beauty_Personal_Care" : ["skincare", "beauty products"],
    "Security_Surveillance" : ["cctv", "security camera"],  # stopped at cctv
    "Office_Printer_Supplies" : ["printer", "scanner"],
    "Software" : ["software", "saas"],
    "Fashion_Accessories" : ["handbag", "watch", "wallet"],
    "Sports_And_Fitness" : ["fitness", "gym", "sports equipment", "yoga"]
    #,
    #"Health_Care",
    #"Home_Furnishing",
    #"Grocery",
    #"Tools",
    #"Party_Accessories"
}


# 4. Fetch NEWS Function
def fetch_news(query, category):
    params = {
        "q" : query,
        "from" : from_date,
        "language" : language,
        "sortBy" : "popularity",
        "pageSize" : page_size,
        "apiKey" : api_key
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "source" : a["source"]["name"],
            "author" : a.get("author"),
            "title" : a.get("title"),
            "description" : a.get("description"),
            "content" : a.get("content"),
            "url" : a.get("url"),
            "image_url" : a.get("urlToImage"),
            "published_at" : a.get("publishedAt"),
            "category" : category,
            "query_used" : query,
            "collected_at" : datetime.utcnow()
        })

    return articles


# 4..5 Sentiment Prediction Function
def get_sentiment(text):
    MODEL_NAME = "ProsusAI/finbert"

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_availabe() else "cpu") 
    model.to(device)

    if pd.isna(text) or text.strip() == "":
        return "Neutral"
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs - {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_idx = torch.argmax(probs).item()

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return label_map[sentiment_idx]


# 5. Main Pipeline
def get_news_data():
    try:
        all_articles = []

        for category, keywords in tqdm(category_keywords.items()):
            for keyword in keywords:
                try:
                    articles = fetch_news(keyword, category)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Error fetching {keyword} : {e}")


        # 6. Save (Append) To CSV
        OUTPUT_FILE = "DataForProject/news_categorized_data.csv"
        news_df = pd.DataFrame(all_articles)
        news_df.drop_duplicates(subset="url", inplace=True)  #Remove duplicates based on URL

        #write_header = not os.path.exists("DataForProject/news_categorized_data.csv")
        #news_df.to_csv("DataForProject/news_categorized_data.csv", mode="a", index=False, header=write_header)
        #print(f"Saved {len(news_df)} articles to news_categorized_data.csv")

        # combine text field
        news_df["combined_text"] = (
            news_df["title"].fillna("") + ". " +
            news_df["description"].fillna("") + ". " +
            news_df["content"].fillna("") 
        )

        #apply sentiment model
        tqdm.pandas()
        news_df["sentiment_label"] = news_df["combined_text"].progress_apply(get_sentiment)

        #save output
        news_df.drop(columns=["combined_text"], inplace=True)

        if os.path.exists(OUTPUT_FILE):
            #append without header
            news_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
        else:
            #create file with header
            news_df.to_csv(OUTPUT_FILE, index=False)
        
        result_df = sentiment_news_spike.new_sentiment_spike(news_df)
        if result_df.empty:
            Notification.send_mail("News Data Alert", "News Data Extracted Successfully and No major weekly Reddit sentiment spikes or trend shifts detected." )
        else:
            Notification.send_mail("News Data Alert", "News Data Extracted Successfully and please find the attached report of sentiment spike and trend shift of this week", result_df )
            print("✅ Sentiment analysis completed and saved successfully.")

    
    except Exception as e:
        Notification.send_mail(f"Failed to Extract News Data and the reason is: {e}", "News Data Alert")