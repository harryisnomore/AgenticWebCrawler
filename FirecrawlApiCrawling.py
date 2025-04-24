from firecrawl import FirecrawlApp
from pymongo import MongoClient
import logging
from urllib.parse import urlparse
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from googlesearch import search
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import pipeline
import pandas as pd
from langdetect import detect
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import re
import tweepy
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app with SocketIO
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

class CrawlerState(TypedDict):
    urls_to_visit: List[str]
    visited_urls: List[str]
    content_vectors: List[np.ndarray]
    priority_queue: List[str]
    current_content: Optional[str]
    page_count: int
    max_pages: int
    domain: str
    status: Optional[str]
    entities: Optional[List[dict]]
    sentiments: Optional[List[dict]]
    languages: Optional[List[str]]
    social_posts: Optional[List[dict]]

def setup_mongodb(db_name="crawler_db", collection_name="pages", query_collection_name="queries"):
    """Connect to MongoDB with environment variable or local fallback."""
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logging.info(f"MongoDB connection successful: {mongo_uri}")
        return client[db_name][collection_name], client[db_name][query_collection_name]
    except Exception as e:
        logging.error(f"MongoDB connection error with {mongo_uri}: {e}", exc_info=True)
        if mongo_uri != "mongodb://localhost:27017/":
            logging.info("Falling back to local MongoDB")
            try:
                client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=10000)
                client.admin.command('ismaster')
                logging.info("Local MongoDB connection successful")
                return client[db_name][collection_name], client[db_name][query_collection_name]
            except Exception as local_e:
                logging.error(f"Local MongoDB connection error: {local_e}", exc_info=True)
        raise Exception(f"Failed to connect to MongoDB: {str(e)}")

def setup_firecrawl():
    """Initialize Firecrawl."""
    try:
        firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        logging.info("Firecrawl setup successful.")
        return firecrawl
    except Exception as e:
        logging.error(f"Firecrawl setup error: {e}", exc_info=True)
        raise

def setup_azure_openai():
    """Configure Azure OpenAI client."""
    try:
        client = AzureChatOpenAI(
            openai_api_key=os.getenv("API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT"),
            deployment_name="gpt-4o-mini",
            api_version="2024-02-15-preview",
            temperature=0.7
        )
        logging.info("Azure OpenAI client setup successful.")
        return client
    except Exception as e:
        logging.error(f"Azure OpenAI setup error: {e}", exc_info=True)
        raise

def setup_embedding_model():
    """Initialize embedding model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Embedding model setup successful.")
        return model
    except Exception as e:
        logging.error(f"Embedding model setup error: {e}", exc_info=True)
        raise

def setup_nlp():
    """Initialize spaCy for NER with fallback."""
    try:
        # Check if en_core_web_sm is installed
        spacy.util.get_installed_models()
        if 'en_core_web_sm' not in spacy.util.get_installed_models():
            logging.info("en_core_web_sm not found, attempting to download")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logging.info("spaCy NLP setup successful.")
        return nlp
    except Exception as e:
        logging.error(f"spaCy setup error: {e}", exc_info=True)
        logging.info("Falling back to regex-based NER")
        return None  # Indicates fallback to regex

def setup_sentiment_analyzer():
    """Initialize sentiment analysis model."""
    try:
        analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logging.info("Sentiment analyzer setup successful.")
        return analyzer
    except Exception as e:
        logging.error(f"Sentiment analyzer setup error: {e}", exc_info=True)
        raise

def setup_azure_translator():
    """Initialize Azure Translator client."""
    try:
        credential = AzureKeyCredential(os.getenv("AZURE_TRANSLATOR_KEY"))
        client = TextAnalyticsClient(
            endpoint=os.getenv("AZURE_TRANSLATOR_ENDPOINT"),
            credential=credential
        )
        logging.info("Azure Translator client setup successful.")
        return client
    except Exception as e:
        logging.error(f"Azure Translator setup error: {e}", exc_info=True)
        raise

def setup_twitter_api():
    """Initialize Twitter/X API client."""
    try:
        client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        )
        logging.info("Twitter API setup successful.")
        return client
    except Exception as e:
        logging.error(f"Twitter API setup error: {e}", exc_info=True)
        raise

def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ['http', 'https'] and parsed.netloc)
    except Exception:
        return False

def fetch_urls_for_query(query: str, max_urls: int = 5) -> List[str]:
    """Fetch URLs for product/company."""
    try:
        logging.info(f"Fetching URLs for query: {query}")
        urls = [url for url in search(query, num_results=max_urls, lang="en") if validate_url(url)]
        if not urls:
            logging.warning(f"No URLs found for query: {query}. Using fallback sources.")
            urls = [
                "https://www.bbc.com/news",
                "https://www.reuters.com/",
                "https://www.theguardian.com/"
            ]
        logging.info(f"Fetched {len(urls)} URLs: {urls}")
        return urls[:max_urls]
    except Exception as e:
        logging.error(f"Error fetching URLs for {query}: {e}", exc_info=True)
        return [
            "https://www.bbc.com/news",
            "https://www.reuters.com/",
            "https://www.theguardian.com/"
        ]

def fetch_social_posts(query: str, twitter_client, max_posts: int = 10) -> List[dict]:
    """Fetch Twitter/X posts for query."""
    try:
        tweets = twitter_client.search_recent_tweets(query=query, max_results=max_posts, tweet_fields=['created_at', 'text'])
        if not tweets.data:
            logging.info(f"No Twitter/X posts found for query: {query}")
            return []
        posts = [{"text": tweet.text, "created_at": tweet.created_at.isoformat()} for tweet in tweets.data]
        logging.info(f"Fetched {len(posts)} Twitter/X posts for query: {query}")
        return posts
    except Exception as e:
        logging.error(f"Error fetching Twitter/X posts: {e}", exc_info=True)
        return []

def regex_ner(text: str) -> List[dict]:
    """Basic regex-based NER for fallback."""
    entities = []
    # Simple patterns for PERSON, ORG, GPE
    person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    org_pattern = r'\b[A-Z][a-z]*(?: [A-Z][a-z]*)*\b(?=\s+(Inc|Corp|LLC|Company))'
    gpe_pattern = r'\b(?:New York|California|London|Paris|Tokyo|United States|China)\b'
    
    for match in re.finditer(person_pattern, text):
        entities.append({"text": match.group(), "label": "PERSON"})
    for match in re.finditer(org_pattern, text):
        entities.append({"text": match.group(), "label": "ORG"})
    for match in re.finditer(gpe_pattern, text):
        entities.append({"text": match.group(), "label": "GPE"})
    
    return entities

def translate_text(text: str, client, target_language: str = "en") -> str:
    """Translate text to target language using Azure Translator."""
    try:
        result = client.translate(text[:5000], to_language=target_language)
        return result[0].translations[0].text if result else text
    except Exception as e:
        logging.error(f"Translation error: {e}", exc_info=True)
        return text

def fetch_url(state: CrawlerState, firecrawl, collection, nlp, sentiment_analyzer, translator, twitter_client, context: str):
    """Node: Fetch and process a URL using Firecrawl."""
    logging.debug(f"Entering fetch_url. State: {state}")
    if state["page_count"] >= state["max_pages"]:
        state["status"] = "stop"
        return "stop"

    url_to_fetch = None
    if state['priority_queue']:
        url_to_fetch = state['priority_queue'].pop(0)
    elif state['urls_to_visit']:
        potential_url = state['urls_to_visit'].pop(0)
        while potential_url in state['visited_urls'] and state['urls_to_visit']:
            potential_url = state['urls_to_visit'].pop(0)
        if potential_url not in state['visited_urls']:
            url_to_fetch = potential_url

    if not url_to_fetch:
        logging.info("No more URLs to fetch.")
        return {**state, "status": "queue_empty", "current_content": None}

    try:
        logging.info(f"Fetching ({state['page_count'] + 1}/{state['max_pages']}): {url_to_fetch}")
        response = firecrawl.crawl_url(
            url_to_fetch,
            params={
                'pageOptions': {'onlyMainContent': True},
                'browserOptions': {'waitUntil': 'networkidle2'}
            }
        )
        content = response[0]['markdown'][:50000] if response and response[0].get('markdown') else ""
        title = response[0]['metadata']['title'] or "No Title"

        # Detect language and translate if not English
        language = detect(content[:500]) if content else "en"
        translated_content = content
        if language != "en":
            translated_content = translate_text(content, translator, target_language="en")

        # Extract entities
        entities = []
        if nlp:
            doc = nlp(translated_content[:10000])
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
        else:
            entities = regex_ner(translated_content[:10000])

        # Perform sentiment analysis
        sentiment = sentiment_analyzer(translated_content[:512])[0]
        sentiment_info = {"label": sentiment['label'], "score": sentiment['score']}

        collection.update_one(
            {"url": url_to_fetch},
            {"$set": {
                "url": url_to_fetch,
                "title": title,
                "content": content,
                "translated_content": translated_content,
                "language": language,
                "crawled_at": datetime.utcnow().isoformat(),
                "domain": state['domain'],
                "entities": entities,
                "sentiment": sentiment_info
            }},
            upsert=True
        )
        new_urls_found = []
        for link in response[0].get('links', []):
            full_url = link.get('url')
            if full_url and validate_url(full_url):
                parsed_url = urlparse(full_url)
                if (parsed_url.netloc == state['domain'] and
                    full_url not in state['visited_urls'] and
                    '#' not in full_url):
                    if full_url not in state['priority_queue'] and full_url not in state['urls_to_visit']:
                        new_urls_found.append(full_url)
        
        # Fetch social posts if first page
        social_posts = []
        if state['page_count'] == 0 and context.startswith(("product:", "company:")):
            query = context.split(":")[1].strip()
            social_posts = fetch_social_posts(query, twitter_client)

        # Emit WebSocket update
        socketio.emit('crawl_update', {
            'url': url_to_fetch,
            'page_count': state['page_count'] + 1,
            'max_pages': state['max_pages'],
            'entities': entities,
            'language': language,
            'sentiment': sentiment_info,
            'social_posts': social_posts
        })

        updated_state = {
            **state,
            "current_content": translated_content,
            "urls_to_visit": list(set(state['urls_to_visit'] + new_urls_found)),
            "visited_urls": state['visited_urls'] + [url_to_fetch],
            "page_count": state['page_count'] + 1,
            "priority_queue": state['priority_queue'],
            "status": "fetched",
            "entities": entities,
            "sentiments": [sentiment_info],
            "languages": [language],
            "social_posts": social_posts
        }
        logging.debug(f"Exiting fetch_url successfully. State: {updated_state}")
        return updated_state
    except Exception as e:
        logging.error(f"Error fetching {url_to_fetch}: {e}", exc_info=True)
        socketio.emit('crawl_update', {
            'url': url_to_fetch,
            'error': str(e),
            'page_count': state['page_count'],
            'max_pages': state['max_pages']
        })
        return {
            **state,
            "visited_urls": state['visited_urls'] + [url_to_fetch],
            "current_content": None,
            "priority_queue": state['priority_queue'],
            "status": "fetch_error",
            "entities": [],
            "sentiments": [],
            "languages": [],
            "social_posts": []
        }

def analyze_content(state: CrawlerState, embedding_model, collection):
    """Node: Analyze content and generate embeddings."""
    logging.debug("Entering analyze_content.")
    if state.get("current_content"):
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            vectors = vectorizer.fit_transform([state['current_content']])
            new_vector = vectors.toarray()[0]
            existing_vectors = state.get('content_vectors', [])
            embedding = embedding_model.encode([state['current_content']])[0]
            collection.update_one(
                {"url": state['visited_urls'][-1]},
                {"$set": {"embedding": embedding.tolist()}}
            )
            updated_state = {
                **state,
                "content_vectors": existing_vectors + [new_vector]
            }
            logging.debug("Content analyzed and embedding stored successfully.")
            return updated_state
        except Exception as e:
            logging.error(f"Error during content analysis: {e}", exc_info=True)
            return {**state}
    else:
        logging.debug("No current content to analyze.")
        return {**state}

def generate_wordcloud(content: str) -> str:
    """Generate a word cloud."""
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(content)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}", exc_info=True)
        return ""

def export_to_csv(documents, context: str):
    """Export crawled data to CSV."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        context_safe = context.replace(":", "_").replace(" ", "_")
        filename = f"export_{context_safe}_{timestamp}.csv"
        data = [{
            "url": doc['url'],
            "title": doc['title'],
            "content": doc['content'][:1000],
            "translated_content": doc.get('translated_content', '')[:1000],
            "language": doc.get('language', 'en'),
            "crawled_at": doc['crawled_at'],
            "entities": "; ".join([f"{e['text']} ({e['label']})" for e in doc.get('entities', [])]),
            "sentiment": f"{doc['sentiment']['label']} ({doc['sentiment']['score']:.2f})" if 'sentiment' in doc else "N/A"
        } for doc in documents]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Data exported to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}", exc_info=True)
        return ""

def export_to_pdf(report: str, context: str) -> str:
    """Export report to PDF."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        context_safe = context.replace(":", "_").replace(" ", "_")
        filename = f"report_{context_safe}_{timestamp}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        for line in report.split('\n'):
            if y < 50:
                c.showPage()
                y = 750
            c.drawString(50, y, line[:100])  # Truncate long lines
            y -= 15
        c.save()
        logging.info(f"Report exported to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error exporting to PDF: {e}", exc_info=True)
        return ""

def report_agent(pages_collection, azure_client, domain: str, context: str) -> str:
    """Generate a detailed report with entities, sentiment, language, and social posts."""
    logging.debug("Entering report_agent.")
    try:
        documents = list(pages_collection.find({"content": {"$exists": True}, "domain": domain}))
        if not documents:
            logging.warning("No content available to generate report.")
            return f"# Crawl Report\n\nNo content was crawled for {context}. Please check the logs."

        aggregated_content = "\n\n".join([doc.get('translated_content', doc['content'])[:5000] for doc in documents])
        if not aggregated_content.strip():
            logging.warning("Aggregated content is empty.")
            return f"# Crawl Report\n\nNo valid content was found for {context}."

        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([doc.get('translated_content', doc['content']) for doc in documents if doc['content']])
        feature_names = vectorizer.get_feature_names_out()
        key_terms = ", ".join(feature_names) if feature_names.size > 0 else "None"

        entities = []
        sentiments = []
        languages = []
        social_posts = []
        for doc in documents:
            entities.extend(doc.get('entities', []))
            if 'sentiment' in doc:
                sentiments.append(doc['sentiment'])
            languages.append(doc.get('language', 'en'))
            social_posts.extend(doc.get('social_posts', []))
        
        entity_summary = "\n".join([f"- {e['text']} ({e['label']})" for e in set(tuple(e.items()) for e in entities)][:10])
        sentiment_summary = f"- Average Sentiment: {'Positive' if sum(s['score'] for s in sentiments if s['label'] == 'POSITIVE') > sum(s['score'] for s in sentiments if s['label'] == 'NEGATIVE') else 'Negative'}"
        language_summary = f"- Languages Detected: {', '.join(set(languages))}"
        social_summary = "\n".join([f"- {p['text'][:100]}... (Posted: {p['created_at']})" for p in social_posts][:5]) or "None"

        wordcloud_base64 = generate_wordcloud(aggregated_content)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an analyst generating a detailed report based on crawled web content and social media posts."),
            ("user", """
            Below is aggregated content from crawled webpages and social media posts related to {context}:

            Web Content:
            {content}

            Social Media Posts:
            {social_posts}

            Generate a detailed report with the following sections:
            - **Overview**: Summarize the main focus or purpose of the content (1-2 sentences).
            - **Key Findings**: List 3-5 key insights or topics found in the content (bullet points).
            - **Recommendations**: Suggest 1-2 ways to explore the content further (e.g., specific queries).

            Format the report in Markdown for clarity.
            """)
        ])

        chain = prompt_template | azure_client
        report = chain.invoke({
            "content": aggregated_content[:20000],
            "social_posts": "\n".join([p['text'] for p in social_posts])[:5000],
            "context": context
        }).content.strip()

        stats = f"""
## Crawl Statistics
- **Pages Crawled**: {len(documents)}
- **Key Terms Identified**: {key_terms}
- **Entities Extracted**:
{entity_summary or "None"}
- **Sentiment Analysis**:
{sentiment_summary}
- **Language Analysis**:
{language_summary}
- **Social Media Insights**:
{social_summary}
- **Crawl Timestamp**: {datetime.utcnow().isoformat()}
"""
        if wordcloud_base64:
            stats += f"\n\n![Word Cloud](data:image/png;base64,{wordcloud_base64})"

        final_report = f"# Crawl Report for {context}\n\n{report}\n\n{stats}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        context_safe = context.replace(":", "_").replace(" ", "_")
        filename = f"report_{context_safe}_{timestamp}.md"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_report)
            logging.info(f"Report saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save report to file: {e}", exc_info=True)

        # Export to CSV and PDF
        csv_filename = export_to_csv(documents, context)
        pdf_filename = export_to_pdf(final_report, context)

        logging.info("Report generated successfully.")
        return final_report
    except Exception as e:
        logging.error(f"Error generating report: {e}", exc_info=True)
        return f"# Crawl Report\n\nError generating report for {context}. Please check the logs."

def query_agent(query: str, azure_client, query_collection, embedding_model, pages_collection, domain: str) -> str:
    """Process a query using MongoDB Vector Search."""
    logging.debug("Entering query_agent.")
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            },
            {"$match": {"domain": domain}},
            {"$project": {"translated_content": {"$substr": [{"$ifNull": ["$translated_content", "$content"]}, 0, 5000]}, "social_posts": 1, "_id": 0}}
        ]
        documents = list(pages_collection.aggregate(pipeline))
        context = "\n\n".join([doc['translated_content'] for doc in documents]) or "No relevant content found."
        social_context = "\n".join([p['text'] for doc in documents for p in doc.get('social_posts', [])])[:5000]

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers queries based on crawled web content and social media posts."),
            ("user", """
            Below is relevant content from crawled webpages and social media posts:

            Web Content:
            {context}

            Social Media Posts:
            {social_context}

            User Query: {query}

            Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
            """)
        ])

        chain = prompt_template | azure_client
        answer = chain.invoke({"context": context, "social_context": social_context, "query": query}).content.strip()

        query_collection.insert_one({
            "query": query,
            "response": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain
        })

        logging.info(f"Query processed: {query}")
        return answer
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return f"Error processing query: {str(e)}"

def should_continue(state: CrawlerState):
    """Decision node: Continue crawling?"""
    logging.debug("Entering should_continue.")
    if state['page_count'] >= state['max_pages']:
        logging.info("Decision: Stop (max pages reached).")
        return "stop"
    if not state['priority_queue'] and not state['urls_to_visit']:
        logging.info("Decision: Stop (no more URLs in queues).")
        return "stop"
    logging.debug("Decision: Continue.")
    return "continue"

def prioritize_urls(state: CrawlerState):
    """Node: Prioritize URLs."""
    logging.debug("Entering prioritize_urls.")
    urls_to_consider = list(set(state['urls_to_visit']) - set(state['visited_urls']))
    if not urls_to_consider:
        logging.debug("No new URLs to prioritize.")
        return {**state, "urls_to_visit": []}
    sorted_urls = sorted(
        urls_to_consider,
        key=lambda x: len(urlparse(x).path.split('/')),
        reverse=False
    )
    new_priority_queue = sorted_urls + state.get('priority_queue', [])
    unique_priority_queue = []
    seen = set()
    for item in new_priority_queue:
        if item not in seen:
            unique_priority_queue.append(item)
            seen.add(item)
    MAX_QUEUE_SIZE = 1000
    final_priority_queue = unique_priority_queue[:MAX_QUEUE_SIZE]
    logging.info(f"Prioritized {len(urls_to_consider)} URLs. Priority queue size: {len(final_priority_queue)}")
    updated_state = {
        **state,
        "priority_queue": final_priority_queue,
        "urls_to_visit": []
    }
    return updated_state

def create_workflow(firecrawl, pages_collection, embedding_model, nlp, sentiment_analyzer, translator, twitter_client, context):
    """Create LangGraph workflow."""
    workflow = StateGraph(CrawlerState)
    workflow.add_node("fetch_url", lambda state: fetch_url(state, firecrawl, pages_collection, nlp, sentiment_analyzer, translator, twitter_client, context))
    workflow.add_node("analyze_content", lambda state: analyze_content(state, embedding_model, pages_collection))
    workflow.add_node("prioritize_urls", prioritize_urls)
    workflow.set_entry_point("fetch_url")
    workflow.add_edge("fetch_url", "analyze_content")
    workflow.add_edge("analyze_content", "prioritize_urls")
    workflow.add_conditional_edges(
        "prioritize_urls",
        should_continue,
        {"continue": "fetch_url", "stop": END}
    )
    app = workflow.compile()
    logging.info("Workflow compiled.")
    return app

# Flask Endpoints
@app.route('/')
def serve_index():
    try:
        if not os.path.exists('static/index.html'):
            logging.error("index.html not found in static folder")
            return jsonify({"error": "Index file not found"}), 404
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logging.error(f"Error serving index.html: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.json
    crawl_type = data.get('crawl_type')
    input_value = data.get('input_value')
    max_pages = int(data.get('max_pages', 10))

    if not crawl_type or not input_value:
        return jsonify({"error": "Missing crawl_type or input_value"}), 400

    if crawl_type == 'website':
        urls = [url.strip() for url in input_value.split(',') if url.strip()]
        valid_urls = [url for url in urls if validate_url(url)]
        if not valid_urls:
            return jsonify({"error": "No valid URLs provided"}), 400
        domain = urlparse(valid_urls[0]).netloc
        context = "website"
    elif crawl_type == 'product':
        urls = fetch_urls_for_query(f"{input_value} news", max_urls=5)
        domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
        context = f"product: {input_value}"
    else:  # company
        urls = fetch_urls_for_query(f"{input_value} news", max_urls=5)
        domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
        context = f"company: {input_value}"

    try:
        firecrawl = setup_firecrawl()
        pages_collection, query_collection = setup_mongodb()
        azure_client = setup_azure_openai()
        embedding_model = setup_embedding_model()
        nlp = setup_nlp()
        sentiment_analyzer = setup_sentiment_analyzer()
        translator = setup_azure_translator()
        twitter_client = setup_twitter_api()

        initial_state = CrawlerState(
            urls_to_visit=[],
            visited_urls=[],
            content_vectors=[],
            priority_queue=urls,
            current_content=None,
            page_count=0,
            max_pages=max_pages,
            domain=domain,
            status="initial",
            entities=[],
            sentiments=[],
            languages=[],
            social_posts=[]
        )
        workflow_app = create_workflow(firecrawl, pages_collection, embedding_model, nlp, sentiment_analyzer, translator, twitter_client, context)
        final_state = None
        crawl_start_time = datetime.now()
        for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
            for node_name, result_state in step_output.items():
                if isinstance(result_state, dict) and 'page_count' in result_state:
                    final_state = result_state
        crawl_end_time = datetime.now()
        
        report = report_agent(pages_collection, azure_client, domain, context)
        visited_urls = final_state['visited_urls'] if final_state else initial_state.get('visited_urls', [])
        
        return jsonify({
            "report": report,
            "visited_urls": visited_urls,
            "crawl_time": str(crawl_end_time - crawl_start_time),
            "domain": domain,
            "context": context
        })
    except Exception as e:
        logging.error(f"Crawl error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    domain = data.get('domain')
    if not query_text or not domain:
        return jsonify({"error": "Missing query or domain"}), 400
    try:
        azure_client = setup_azure_openai()
        embedding_model = setup_embedding_model()
        _, query_collection = setup_mongodb()
        pages_collection, _ = setup_mongodb()
        response = query_agent(query_text, azure_client, query_collection, embedding_model, pages_collection, domain)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Query error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    logging.info("Client connected to WebSocket")
    emit('connection_status', {'status': 'connected'})

# Create static folder and index.html
try:
    os.makedirs('static', exist_ok=True)
    with open('static/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Web Crawler</title>
    <script src="https://cdn.jsdelivr.net/npm/react@17/umd/react.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@17/umd/react-dom.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/babel-standalone@6/babel.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/socket.io-client@4/dist/socket.io.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div id="root"></div>
    <script type="text/babel">
        function App() {
            const [crawlType, setCrawlType] = React.useState('');
            const [inputValue, setInputValue] = React.useState('');
            const [maxPages, setMaxPages] = React.useState(10);
            const [report, setReport] = React.useState('');
            const [urls, setUrls] = React.useState([]);
            const [domain, setDomain] = React.useState('');
            const [context, setContext] = React.useState('');
            const [query, setQuery] = React.useState('');
            const [queryResponse, setQueryResponse] = React.useState('');
            const [error, setError] = React.useState('');
            const [crawlUpdates, setCrawlUpdates] = React.useState([]);

            React.useEffect(() => {
                const socket = io();
                socket.on('connection_status', (data) => {
                    console.log('WebSocket connected:', data.status);
                });
                socket.on('crawl_update', (data) => {
                    setCrawlUpdates((prev) => [...prev, data].slice(-10));
                });
                return () => socket.disconnect();
            }, []);

            const handleCrawl = async () => {
                setError('');
                setCrawlUpdates([]);
                try {
                    const response = await fetch('/crawl', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({crawl_type: crawlType, input_value: inputValue, max_pages: maxPages})
                    });
                    const data = await response.json();
                    if (data.error) {
                        setError(data.error);
                    } else {
                        setReport(data.report);
                        setUrls(data.visited_urls);
                        setDomain(data.domain);
                        setContext(data.context);
                    }
                } catch (err) {
                    setError('Failed to initiate crawl');
                }
            };

            const handleQuery = async () => {
                setError('');
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query, domain: domain})
                    });
                    const data = await response.json();
                    if (data.error) {
                        setError(data.error);
                    } else {
                        setQueryResponse(data.response);
                    }
                } catch (err) {
                    setError('Failed to process query');
                }
            };

            const handleExportCSV = () => {
                const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(
                    urls.map(url => `"${url}"`).join("\\n")
                );
                const link = document.createElement("a");
                link.setAttribute("href", csvContent);
                link.setAttribute("download", "crawled_urls.csv");
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };

            const handleExportPDF = () => {
                // Trigger server-side PDF generation
                alert("PDF report is being generated. Check the server directory for the file.");
            };

            return (
                <div className="container mx-auto p-4">
                    <h1 className="text-3xl font-bold mb-4">Advanced Web Crawler</h1>
                    <div className="mb-4">
                        <label className="block text-lg mb-2">Crawl Type</label>
                        <select
                            className="w-full p-2 border rounded"
                            value={crawlType}
                            onChange={(e) => setCrawlType(e.target.value)}
                        >
                            <option value="">Select Type</option>
                            <option value="website">Website</option>
                            <option value="product">Product</option>
                            <option value="company">Company</option>
                        </select>
                    </div>
                    <div className="mb-4">
                        <label className="block text-lg mb-2">Input</label>
                        <input
                            className="w-full p-2 border rounded"
                            type="text"
                            placeholder={crawlType === 'website' ? 'Enter URLs (comma-separated)' : 'Enter name'}
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                        />
                    </div>
                    <div className="mb-4">
                        <label className="block text-lg mb-2">Max Pages</label>
                        <input
                            className="w-full p-2 border rounded"
                            type="number"
                            value={maxPages}
                            onChange={(e) => setMaxPages(e.target.value)}
                        />
                    </div>
                    <button
                        className="bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
                        onClick={handleCrawl}
                    >
                        Start Crawl
                    </button>
                    {error && <p className="text-red-500 mt-2">{error}</p>}
                    {crawlUpdates.length > 0 && (
                        <div className="mt-4">
                            <h2 className="text-2xl font-bold">Crawl Progress</h2>
                            <div className="bg-white p-4 rounded shadow">
                                <div className="w-full bg-gray-200 rounded">
                                    <div
                                        className="bg-blue-500 text-xs font-medium text-white text-center p-0.5 leading-none rounded"
                                        style={{ width: `${(crawlUpdates[crawlUpdates.length - 1].page_count / crawlUpdates[crawlUpdates.length - 1].max_pages) * 100}%` }}
                                    >
                                        {`${Math.round((crawlUpdates[crawlUpdates.length - 1].page_count / crawlUpdates[crawlUpdates.length - 1].max_pages) * 100)}%`}
                                    </div>
                                </div>
                                <ul className="mt-2">
                                    {crawlUpdates.map((update, index) => (
                                        <li key={index} className="text-sm">
                                            {update.error ? (
                                                <span className="text-red-500">Error: {update.error}</span>
                                            ) : (
                                                <span>
                                                    Crawled: {update.url} (Page {update.page_count}/{update.max_pages}, Language: {update.language}, Sentiment: {update.sentiment.label})
                                                    {update.social_posts.length > 0 && (
                                                        <ul className="pl-4">
                                                            {update.social_posts.map((post, i) => (
                                                                <li key={i} className="text-xs">{post.text.slice(0, 50)}...</li>
                                                            ))}
                                                        </ul>
                                                    )}
                                                </span>
                                            )}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    )}
                    {report && (
                        <div className="mt-4">
                            <h2 className="text-2xl font-bold">Crawl Report</h2>
                            <div
                                className="prose"
                                dangerouslySetInnerHTML={{ __html: report.replace(/\n/g, '<br>').replace(/!\[Word Cloud\]\(data:image\/png;base64,(.+?)\)/, '<img src="data:image/png;base64,$1" alt="Word Cloud" />') }}
                            />
                            <h3 className="text-xl font-bold mt-4">Crawled URLs</h3>
                            <button
                                className="bg-green-500 text-white p-2 rounded hover:bg-green-600 mr-2"
                                onClick={handleExportCSV}
                            >
                                Export URLs to CSV
                            </button>
                            <button
                                className="bg-purple-500 text-white p-2 rounded hover:bg-purple-600"
                                onClick={handleExportPDF}
                            >
                                Export Report to PDF
                            </button>
                            <ul className="list-disc pl-5 mt-2">
                                {urls.map((url, index) => (
                                    <li key={index}><a href={url} className="text-blue-500" target="_blank">{url}</a></li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {domain && (
                        <div className="mt-4">
                            <h2 className="text-2xl font-bold">Query Content</h2>
                            <input
                                className="w-full p-2 border rounded mb-2"
                                type="text"
                                placeholder="Enter your query"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                            />
                            <button
                                className="bg-green-500 text-white p-2 rounded hover:bg-green-600"
                                onClick={handleQuery}
                            >
                                Submit Query
                            </button>
                            {queryResponse && (
                                <div className="mt-2">
                                    <h3 className="text-lg font-bold">Response</h3>
                                    <p>{queryResponse}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
        """)
    logging.info("Static folder and index.html created successfully.")
except Exception as e:
    logging.error(f"Failed to create static/index.html: {e}", exc_info=True)

# Main Execution
if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)