# 0. Header Libraries
import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai 
from google.genai import types
from groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()


# -. SCHEDULER
import schedule
import threading
import time
import notifications.Notification as Notification
from apis_data_collection import News_API, Reddit_API

def run_scheduler():
    #call function every 10 seconds
    #schedule.every(10).seconds.do(Notification.testing_function)

    #call function every 10 minutes
    #schedule.every(10).minutes.do(Notification.testing_function)

    #call function every day at a given time
    #schedule.every().day.at("00:00").do(Notification.testing_function)

    #call function every week at a given day and time
    schedule.every().sunday.at("02:00").do(News_API.get_news_data)
    schedule.every().monday.at("02:00").do(Reddit_API.reddit_api)

    while True:
        schedule.run_pending()
        time.sleep(5)

#start scheduler on a different thread to run simultaneously
threading.Thread(target=run_scheduler, daemon=True).start()

if __name__=="__main__":
    # 1. Page Configuration
    st.set_page_config(
        page_title="AI Market Trend & Consumer Sentiment Forecaster",
        layout="wide"  
    )

    st.title("AI-Powered Market Trend & Consumer Sentiment Dashboard")
    st.markdown("Consumer sentiment, topic trend, and social insights from reivews, news and Reddit data")


    # 2. Loading Data
    @st.cache_data  #caching serializable data, loading/transforming data, qying db, new copy safe from mutations
    def load_data():
        reviews = pd.read_csv("final data/category_wise_lda_output_with_topic_labels.csv")
        reddit = pd.read_excel("final data/reddit_category_trend_data.xlsx")
        news = pd.read_csv("final data/news_data_with_sentiment.csv")

        #print(reddit.columns)

        if "review_date" in reviews.columns:
            reviews["review_date"] = pd.to_datetime(
                reviews["review_date"], errors="coerce"   # convert invalid date to NaT
            )
        
        if "published_at" in news.columns:
            news["published_at"] = pd.to_datetime(
                news["published_at"], errors="coerce"   # convert invalid date to NaT
            )

        if "created_date" in reddit.columns:
            reddit["created_date"] = pd.to_datetime(
                reddit["created_date"], errors="coerce"   # convert invalid date to NaT
            )

        return reviews, reddit, news

    reviews_df, reddit_df, news_df = load_data()


    # 3.1    Load Vector Database
    @st.cache_resource  #caching large unserializable obj, ML models, Db connections, shares single inst of obj across all reruns and sessions without copying 
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "consumer_sentiment_faiss",
            embeddings,
            allow_dangerous_deserialization=True  #pickle file, binary format to freeze and unfreeze large amt of python code data in byte stream, para allows deserialization of the frozen file
        )

        return vector_db

    vector_db = load_vector_db()


    # 3.2 Load Gemini
    @st.cache_resource
    def load_gemini_client():
        client = genai.Client(api_key=os.getenv("Gemini_api_key"))  #??
        return client

    gemini_client = load_gemini_client()

    main_col, right_sidebar = st.columns([3,1])


    # 3.3 Load Groq
    @st.cache_resource
    def load_groq_client():
        client = Groq(
            api_key = os.getenv("Groq_api_key")
        )
        return client
    
    groq_client = load_groq_client()



    # 3. SideBar Filters
    with main_col:
        st.sidebar.header("Filters")

        source_filter = st.sidebar.multiselect(
            "Select Source",
            options=reviews_df["source"].unique(),
            default=reviews_df["source"].unique()
        )

        category_filter = st.sidebar.multiselect(
            "Select Category",
            options=reviews_df["category"].unique(),
            default=reviews_df["category"].unique()
        )

        filtered_reviews = reviews_df[
            (reviews_df["source"].isin(source_filter)) &
            (reviews_df["category"].isin(category_filter))
        ]


        # 4. Key Performance Indicator Metrics
        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(filtered_reviews))
        col2.metric("Positive %", round((filtered_reviews["sentiment_label"]=="Positive").mean()*100, 1))
        col3.metric("Negative %", round((filtered_reviews["sentiment_label"]=="Negative").mean()*100, 1))
        col4.metric("Neutral %", round((filtered_reviews["sentiment_label"]=="Neutral").mean()*100, 1))


        # 5. Sentiment Distribution
        col1, col2 = st.columns(2)

        with col1:
            sentiment_dist = filtered_reviews["sentiment_label"].value_counts().reset_index()
            sentiment_dist.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment_dist,
                names="Sentiment",
                values="Count",
                title="Overall Sentiment Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            category_sentiment = (
                filtered_reviews.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
            )

            fig = px.bar(
                category_sentiment,
                x="category",
                y="count",
                color="sentiment_label",
                title="Category-Wise Sentiment Comparison",
                barmode="group"
            )
            st.plotly_chart(fig, width='stretch')


        # 6. Sentiment Trend Over Time
        st.subheader("Sentiment Trend Over Time")

        sentiment_trend = (
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="W"), "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig_trend = px.line(
            sentiment_trend,
            x="review_date",
            y="count",
            color="sentiment_label",
            title="Weekly Sentiment Trend"
        )
        st.plotly_chart(fig_trend, width='stretch')


        # 7. Category Trend Over Time
        st.subheader("Category Trend Over Time (Product Demand)")

        category_trend = (
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="ME"), "category"])
            .size()
            .reset_index(name="count")
        )

        fig_category_trend = px.line(
            category_trend,
            x="review_date",
            y="count",
            color="category",
            title="Monthly Category Demand Trend"
        )
        st.plotly_chart(fig_category_trend, width='stretch')


        # 8. Category Vs. Sentiment
        #    Category-Wise Sentiment Distribution
        st.subheader("Category-Wise Sentiment Distribution")

        cat_sent = (
            filtered_reviews
            .groupby(["category", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig_cat = px.bar(
            cat_sent,
            x="category",
            y="count",
            color="sentiment_label",
            barmode="group"
        )
        st.plotly_chart(fig_cat, width='stretch')


        # 9. Topic Distribution
        st.subheader("Topic Insights")

        topic_dist = (
            filtered_reviews["topic_label"]
            .value_counts()
            .reset_index()
        )

        topic_dist.columns = ["Topic", "Count"]

        fig_topic = px.bar(
            topic_dist,
            x="Topic",
            y="Count",
            title="Topic Distribution"
        )
        st.plotly_chart(fig_topic, width='stretch')


        # 10. Reddit Category Trend
        st.subheader("Reddit Category Popularity")

        reddit_trend = (
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )

        fig_reddit = px.bar(
            reddit_trend, 
            x="category_label",
            y="mentions",
            title="Trending categories on Reddit"
        )

        st.plotly_chart(fig_reddit, width='stretch')


        # 11. News Sentiment 
        st.subheader("News Sentiment Overview")

        news_sent=(
            news_df
            .groupby("sentiment_label")
            .size()
            .reset_index(name="count")
        )

        fig_news=px.pie(
            news_sent,
            names="sentiment_label",
            values="count",
            title="News Sentiment distribution"
        )

        st.plotly_chart(fig_news, width='stretch')


        # 12. News Category Distribution
        st.subheader("News Category Popularity")

        news_categ = (
            news_df
            .groupby("category")
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )

        fig_news_categ = px.bar(
            news_categ,
            x="category",
            y="mentions",
            title="Trending Categories on News"
        )
        st.plotly_chart(fig_news_categ, width='stretch')


        # 13. Trending On All Platforms
        st.subheader("Cross-Source Category Camparision")

        # review category count
        review_cat=(
            reviews_df
            .groupby("category")
            .size()
            .reset_index(name="Review Mentions")
        )

        # reddit category count
        reddit_cat=(
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="Reddit Mentions")
            .rename(columns={"category_label":"category"})
        )

        # news category count
        news_cat=(
            news_df
            .groupby("category")
            .size()
            .reset_index(name="News Mentions")
        )

        # merge all 
        category_compare = review_cat\
            .merge(reddit_cat, on="category", how="outer")\
            .merge(news_cat, on="category", how="outer")\
            .fillna(0)
            
        # Reshaping data
        category_compare["Total"] = (
            category_compare["Review Mentions"]+
            category_compare["Reddit Mentions"]+
            category_compare["News Mentions"]
        )
        category_compare = (
            category_compare
            .sort_values("Total", ascending=False)
        )
        category_long = category_compare.melt(
            id_vars="category",
            value_vars=["Review Mentions", "Reddit Mentions", "News Mentions"],
            var_name="Source",
            value_name="Mentions"
        )

        fig_compare = px.bar(
            category_long,
            x="Mentions",
            y="category",
            color="Source",
            orientation="h",
            barmode="group",
            title="Category Presence Across Reviews, Reddit and News",
            template="plotly_dark",
            height=700
        )

        fig_compare.update_xaxes(
            type="log",
            title="Mentions (log scale)"
        )

        fig_compare.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=220, r=40, t=80, b=40),
            title_font_size=22
        )

        fig_compare.update_traces(
            opacity=0.9,
            hovertemplate="<b>%{y}</b><br>%{x:,} mentions<extra></extra>"
        )

        st.plotly_chart(fig_compare, width='stretch')
        st.dataframe(
            category_compare.drop(columns="Total"),
            width='stretch'
        )


    # 14. AI chatbot
    with right_sidebar:
        st.markdown("## AI Insight Panel")
        st.caption("Ask Question using Reviews, News & Reddit data!")

        user_query = st.text_area(
            "Your Question",
            height=140
        )

        ask_btn = st.button(
            "Get Insight",
            width='stretch'
        )

        if ask_btn and user_query:
            with st.spinner("Analyzing Market Intelligence..."):  #temp loading msg withs pinning icon
                results = vector_db.similarity_search(user_query, k=10)  # k from knn, top relevant results to be returned
                retreived_docs = [r.page_content for r in results]

                prompt = f""" 
                    You are a market intelligence analyst
                    Using only the information from the provided context
                    Give a response based on the question
                    Do not use bullet points, headings or sections
                    Do not add external knowledge
                    Context:
                    {retreived_docs}
                    Question:
                    {user_query}
                    Answer:
                """

                try:
                    response = gemini_client.models.generate_content(              #make a mistake in spelling gemini
                        model="gemini-2.5-flash",
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            temperature=0.2
                        ),
                    )
                    print("Using Gemini")
                    st.success("Insight Generated")
                    st.write(response.text)

                except Exception as e:
                    print(f"Gemini ran into error : {e}")
                    print("Using Groq")
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role" : "user",
                                "content" : prompt
                            }
                        ],
                        model="llama-3.3-70b-versatile"
                    )
                    st.success("Insight Generated")
                    st.write(chat_completion.choices[0].message.content) 

            