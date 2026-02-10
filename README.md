<p align="center">
  <img src="https://img.shields.io/badge/Project-Sentiment_Analysis-blue?style=for-the-badge" alt="Project">
  <img src="https://img.shields.io/badge/Tech-AI_RAG_Pipeline-green?style=for-the-badge" alt="Tech">
  <img src="https://img.shields.io/badge/Year-2026-orange?style=for-the-badge" alt="Year">
</p>

# AI-Powered Sentiment Analysis & Trend Monitoring Platform
> **Project Documentation** | *Developed by: Aryan Sasane*

---

## 📖 Project Overview
A comprehensive end-to-end data intelligence platform designed to track consumer sentiment, market trends, and industry-specific shifts. By aggregating data from E-commerce reviews (Amazon/Flipkart), Social Media (Reddit), and Global News APIs, the system provides a unified dashboard for visual analysis and a RAG-powered chatbot for natural language insights.



---

## 🚀 Key Features

* **Multi-Channel Data Aggregation:** Automated pipelines to fetch data from Reddit (PRAW), NewsAPI, and Amazon/Flipkart product reviews.
* **Intelligent Data Processing:**
    * **Cleaning & Normalization:** Automated text cleaning, punctuation removal, and stop-word filtering.
    * **Zero-Shot Classification:** Categorizes raw text into industry sectors using `facebook/bart-large-mnli`.
    * **Topic Modeling (LDA):** Extracts specific consumer "aspects" (e.g., product quality, pricing) from large review datasets.
* **Trend & Spike Detection:** Custom algorithms monitor sentiment shifts over a 2-week rolling window and identify "Positive Spikes," "Negative Spikes," and "Trend Shifts."
* **RAG-Powered Chatbot:** A Retrieval-Augmented Generation interface built with LangChain and FAISS, enabling users to ask questions like *"What are the main complaints about home appliances this week?"*
* **Automated Alerting:** A multi-threaded scheduler (`schedule` library) runs weekly data collection and sends detailed Excel-based reports via SMTP if a sentiment spike is detected.

---

## 🛠️ Tech Stack

| Category | Tools & Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, Plotly |
| **Machine Learning** | Transformers (Hugging Face), Scikit-learn (LDA), VADER |
| **LLM Orchestration** | LangChain, FAISS, Google Gemini 1.5 Flash, Groq (Llama 3.3) |
| **Data Handling** | Pandas, NumPy, OpenPyXL |
| **APIs** | Reddit, NewsAPI, RapidAPI (Amazon Data) |

---

## 📋 Project Development Pipeline



### **Phase 1: Data Engineering**
**Step 1: Data Cleaning & Pre-processing (`cleaning.py`)** Standardized raw text by removing noise, including punctuation, special characters, and stop words. This step ensures that the downstream NLP models focus only on the most meaningful terms for analysis.

**Step 2: Data Unification & Merging (`Merge_Data.py`)** Aggregated datasets from multiple e-commerce sources (Amazon and Flipkart) into a unified schema. This involved mapping disparate column names to a consistent format to create a single source of truth for product reviews.

**Step 3: Dataset Optimization (`Reduce_Data.py`)** Managed computational efficiency by performing a representative sampling of the merged dataset. This reduced the data volume while maintaining the statistical integrity and diversity of the reviews.

### **Phase 2: NLP & Intelligence**
**Step 4: Zero-Shot Industry Classification (`Categorize_Data.py`)** Implemented a `facebook/bart-large-mnli` transformer model to automatically categorize products into specific industry sectors (e.g., Home Appliances, Electronics). This allowed for granular, category-wise trend monitoring.

**Step 5: Sentiment Enrichment & Temporal Mapping (`Rating_Sentiment_Data.py`)** Applied a hybrid sentiment analysis approach using star ratings and VADER/Transformer scoring. Additionally, temporal data was normalized to enable rolling window analysis and time-series trend detection.

**Step 6: Aspect-Based Topic Modeling (`Topic_Modelling_LDA.py`)** Utilized Latent Dirichlet Allocation (LDA) to extract underlying discussion themes. This step identifies specific "aspects" of consumer feedback, such as battery life, pricing, or build quality, beyond just basic sentiment.

### **Phase 3: Retrieval & Deployment**
**Step 7: Semantic Indexing & Vector Database (`Vector_db.py`)** Converted the enriched text data into high-dimensional embeddings using Hugging Face models. These embeddings were stored in a FAISS vector database to enable fast, semantic similarity searches for the RAG pipeline.

**Step 8: Interactive RAG Dashboard (`Dashboard.py`)** Developed a Streamlit-based interface that integrates data visualization with a Retrieval-Augmented Generation (RAG) system. Users can visualize sentiment spikes and query the LLM (Gemini/Llama) for natural language insights based on the collected data.

**Step 9: Automated Monitoring & Alerting System (`Notification.py`)** Configured a multi-threaded scheduler to trigger weekly data collection and sentiment analysis. If a significant sentiment "spike" or "trend shift" is detected, the system automatically dispatches an email alert with a detailed Excel report attached.

---
<p align="center">
  <i>© 2026 Aryan Sasane | NIT Goa</i>
</p>