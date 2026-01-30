# 0. Import Header Libraries
import os
import requests
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import random
from dotenv import load_dotenv

load_dotenv()

# top review parameter

# 1. Labels(Categories)
labels = [
    #"Electricals_Power_Backup",
    #"Home_Appliances",
    #"Kitchen_Appliances",
    #"Furniture",
    #"Home_Storage_Organization",
    #"Computers_Tablets",
    #"Mobile_Accessories",
    #"Wearables",
    #"TV_Audio_Entertainment",
    #"Networking_Devices",
    #"Toys_Kids",
    #"Gardening_Outdoor",
    #"Kitchen_Dining",   # stopped at kitchen dining
    "Mens_Clothing",
    "Footwear",
    "Beauty_Personal_Care",
    "Security_Surveillance",
    "Office_Printer_Supplies",
    "Software",
    "Fashion_Accessories",
    "Sports_And_Fitness",
    "Health_Care",
    "Home_Furnishing",
    "Grocery",
    "Tools",
    "Party_Accessories"
]



# 2. Rapid API ASIN Request
url = "https://real-time-amazon-data.p.rapidapi.com/search"

headers = {
    'X-RapidAPI-Host': 'real-time-amazon-data.p.rapidapi.com',
    'X-RapidAPI-Key': os.getenv("Rapid_api_key")
}

all_asins = []



# 3. Amazon ASIN Request API -> Request Data For Each Category 
for label in labels:
    print(f"Fetching Data For : {label}")

    params = {
        "query" : label,
        "page" : 1,
        "country" : "US",
        "sort_by" : "RELEVANCE",
        "product_condition" : "ALL",
        "is_prime" : "false",
        "deals_and_discounts" : "NONE"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        products = data.get("data", {}).get("products", [])    
        limit = 0

        for product in products:
            asin = product.get("asin")
            if asin and limit<5:
                all_asins.append({
                    "asin" : asin,
                    "category" : label
                })
                limit = limit + 1
        
        print(f"Collected {len(products)} ASINs")   

    elif response.status_code == 401:
        print("Unauthorized : stopping script to protect API key.")
        break
    
    elif response.status_code == 429:
        print("API1 Rate time limit hit, sleeping....")
        time.sleep(60)
        continue

    else:
        print(f"Failed For API1 : {label} - {response.status_code}")

    time.sleep(20 + random.randint(5,10))

all_asins = list({item["asin"] : item for item in all_asins}.values())


# 2. Rapid API Product Review Request
url2 = "https://real-time-amazon-data.p.rapidapi.com/top-product-reviews"

print(os.getenv("Rapid_api_key"))

headers2 = {
    'X-RapidAPI-Host': 'real-time-amazon-data.p.rapidapi.com',
    'X-RapidAPI-Key': os.getenv("Rapid_api_key")
}



# 4. Amazon Product Review Request API -> Function
def product_review_request(asin, category):
    print(f"Fetching data for : {asin} - {category}")

    params2={
        "asin" : asin,
        "country" : "US",
        "star_rating" : "ALL"
    }

    response = requests.get(url2, headers=headers2, params=params2)
    
    product_reviews = []
    if response.status_code == 200:
        data = response.json()
        reviews = data.get("data", {}).get("reviews", [])
        for review in reviews:
            product_reviews.append({
                "product" : asin,
                "category" : category,
                "review_title" : review.get("review_title"),
                "review_text" : review.get("review_comment"),
                "rating" : review.get("review_star_rating"),
                "review_date" : review.get("review_date"),
                "review_url" : review.get("review_link"),
                "review_author" : review.get("review_author"),
                "collected_at" : datetime.utcnow().date()
            })
    
    elif response.status_code == 404:
        print(f"Failed for API2 : {asin} - {category}")
    
    elif response.status_code == 429:
        print("API2 Rate time limit hit, sleeping....")
        time.sleep(60)

    elif response.status_code == 401:
        print("Unauthorized : stopping.")
        return []

    else:
        print(f"Failed for API2 : {asin} - {category} - {response.status_code}")

    time.sleep(20 + random.randint(5, 10))

    return product_reviews



# 6. Amazon Product Review Request API
all_product_reviews = []
MAX_ASINS = 5

for item in tqdm(all_asins):
    all_product_reviews.extend(
        product_review_request(item["asin"], item["category"])
        )



# 7. Save (Append) Output to CSV
rapid_df = pd.DataFrame(all_product_reviews)

write_header = not os.path.exists("DataForProject/rapid_categorized_data.csv")

rapid_df.to_csv("DataForProject/rapid_categorized_data.csv", mode="a", index=False, header=write_header)
print(f"Saved {len(rapid_df)} articles to rapid_categorized_data.csv")