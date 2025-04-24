from scrapy import Spider, Request
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider
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
import schedule
import time
import threading
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import signal
import timeout_decorator

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

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

class ContentSpider(Spider):
    name = 'content_spider'
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 8,
    }

    def __init__(self, start_urls, domain, max_pages, collection, nlp, sentiment_analyzer, translator, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.domain = domain
        self.max_pages = max_pages
        self.collection = collection
        self.nlp = nlp
        self.sentiment_analyzer = sentiment_analyzer
        self.translator = translator
        self.context = context
        self.visited_urls = []
        self.page_count = 0
        self.stop_crawling = False

    def parse(self, response):
        if self.stop_crawling or self.page_count >= self.max_pages or self.stop_event.is_set():
            self.stop_crawling = True
            logging.info(f"Stopping spider: page_count={self.page_count}, max_pages={self.max_pages}, stop_event={self.stop_event.is_set()}")
            raise CloseSpider('Max pages reached or stop requested')

        url = response.url
        if url in self.visited_urls:
            return

        self.visited_urls.append(url)
        self.page_count += 1

        try:
            content = response.css('article, main, .content, .post, div[role="main"] ::text').getall()
            content = ' '.join([text.strip() for text in content if text.strip()])[:50000]
            title = response.css('title ::text').get(default='No Title').strip()

            language = detect(content[:500]) if content else "en"
            translated_content = content
            if language != "en":
                translated_content = translate_text(content, self.translator, target_language="en")

            entities = []
            if self.nlp:
                doc = self.nlp(translated_content[:10000])
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
            else:
                entities = regex_ner(translated_content[:10000])

            sentiment = self.sentiment_analyzer(translated_content[:512])[0]
            sentiment_info = {"label": sentiment['label'], "score": sentiment['score']}

            self.collection.update_one(
                {"url": url},
                {"$set": {
                    "url": url,
                    "title": title,
                    "content": content,
                    "translated_content": translated_content,
                    "language": language,
                    "crawled_at": datetime.utcnow().isoformat(),
                    "domain": self.domain,
                    "entities": entities,
                    "sentiment": sentiment_info
                }},
                upsert=True
            )

            socketio.emit('crawl_update', {
                'url': url,
                'page_count': self.page_count,
                'max_pages': self.max_pages,
                'entities': entities,
                'language': language,
                'sentiment': sentiment_info,
                'queued_urls': len(self.crawler.engine.slot.scheduler.pending_requests) if self.crawler else 0
            })

            if self.page_count < self.max_pages:
                for href in response.css('a::attr(href)').getall():
                    full_url = response.urljoin(href)
                    if (validate_url(full_url) and urlparse(full_url).netloc == self.domain and
                            full_url not in self.visited_urls and '#' not in full_url):
                        relevance_score = calculate_url_relevance(full_url, self.context)
                        if relevance_score > 0.5:  # Only yield relevant URLs
                            yield Request(full_url, callback=self.parse)

        except Exception as e:
            logging.error(f"Error parsing {url}: {e}", exc_info=True)
            socketio.emit('crawl_update', {
                'url': url,
                'error': str(e),
                'page_count': self.page_count,
                'max_pages': self.max_pages
            })

def calculate_url_relevance(url: str, context: str) -> float:
    """Calculate relevance of a URL based on context."""
    try:
        stop_words = set(stopwords.words('english'))
        context_tokens = [w.lower() for w in word_tokenize(context) if w.lower() not in stop_words]
        url_tokens = [w.lower() for w in word_tokenize(urlparse(url).path) if w.lower() not in stop_words]
        if not context_tokens or not url_tokens:
            return 0.0
        common_tokens = set(context_tokens) & set(url_tokens)
        return len(common_tokens) / len(set(context_tokens))
    except Exception as e:
        logging.error(f"Error calculating URL relevance: {e}", exc_info=True)
        return 0.0

def run_scrapy_crawl(start_urls, domain, max_pages, collection, nlp, sentiment_analyzer, translator, context):
    """Run Scrapy crawl and ensure it stops."""
    process = CrawlerProcess(settings={
        'LOG_LEVEL': 'INFO',
    })
    spider = ContentSpider(start_urls=start_urls, domain=domain, max_pages=max_pages,
                          collection=collection, nlp=nlp, sentiment_analyzer=sentiment_analyzer,
                          translator=translator, context=context)
    process.crawl(spider)
    process.start()
    process.join()  # Ensure process completes
    return spider.visited_urls

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

def setup_azure_openai():
    """Configure Azure OpenAI client."""
    try:
        client = AzureChatOpenAI(
            openai_api_key=os.getenv("API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT"),
            deployment_name="gpt-4o-mini",
            api_version="2024-02-15-preview",
            temperature=0.7,
            max_tokens=500
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
        if 'en_core_web_sm' not in spacy.util.get_installed_models():
            logging.info("en_core_web_sm not found, attempting to download")
            import subprocess
            result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to download en_core_web_sm: {result.stderr}")
        nlp = spacy.load("en_core_web_sm")
        logging.info("spaCy NLP setup successful.")
        return nlp
    except Exception as e:
        logging.error(f"spaCy setup error: {e}", exc_info=True)
        logging.info("Falling back to regex-based NER")
        return None

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

def regex_ner(text: str) -> List[dict]:
    """Basic regex-based NER for fallback."""
    entities = []
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

def fetch_url(state: CrawlerState, collection):
    """Node: Update state after Scrapy crawl."""
    logging.debug(f"Entering fetch_url. State: {state}")
    if state["page_count"] >= state["max_pages"]:
        state["status"] = "stop"
        return "stop"
    return {**state, "status": "fetched"}

def analyze_content(state: CrawlerState, embedding_model, collection):
    """Node: Analyze content and generate embeddings."""
    logging.debug("Entering analyze_content.")
    # Skip since Scrapy stores content directly
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

def report_agent(pages_collection, azure_client, domain: str, context: str) -> str:
    """Generate a detailed report with entities, sentiment, and language."""
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
        for doc in documents:
            entities.extend(doc.get('entities', []))
            if 'sentiment' in doc:
                sentiments.append(doc['sentiment'])
            languages.append(doc.get('language', 'en'))
        entity_summary = "\n".join([f"- {e['text']} ({e['label']})" for e in set(tuple(e.items()) for e in entities)][:10])
        sentiment_summary = f"- Average Sentiment: {'Positive' if sum(s['score'] for s in sentiments if s['label'] == 'POSITIVE') > sum(s['score'] for s in sentiments if s['label'] == 'NEGATIVE') else 'Negative'}"
        language_summary = f"- Languages Detected: {', '.join(set(languages))}"

        wordcloud_base64 = generate_wordcloud(aggregated_content)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an analyst generating a detailed report based on crawled web content."),
            ("user", """
            Below is aggregated content from crawled webpages related to {context}:

            {content}

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

        csv_filename = export_to_csv(documents, context)
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
            {"$project": {"translated_content": {"$substr": [{"$ifNull": ["$translated_content", "$content"]}, 0, 5000]}, "_id": 0}}
        ]
        documents = list(pages_collection.aggregate(pipeline))
        context = "\n\n".join([doc['translated_content'] for doc in documents]) or "No relevant content found."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers queries based on crawled web content."),
            ("user", """
            Below is relevant content from crawled webpages:

            {context}

            User Query: {query}

            Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
            """)
        ])

        chain = prompt_template | azure_client
        answer = chain.invoke({"context": context, "query": query}).content.strip()

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

def create_workflow(collection):
    """Create simplified LangGraph workflow."""
    workflow = StateGraph(CrawlerState)
    workflow.add_node("fetch_url", lambda state: fetch_url(state, collection))
    workflow.add_node("prioritize_urls", prioritize_urls)
    workflow.set_entry_point("fetch_url")
    workflow.add_conditional_edges(
        "fetch_url",
        should_continue,
        {"continue": "prioritize_urls", "stop": END}
    )
    workflow.add_edge("prioritize_urls", "fetch_url")
    app = workflow.compile()
    logging.info("Workflow compiled.")
    return app

def schedule_crawl(urls, domain, max_pages, collection, nlp, sentiment_analyzer, translator, context):
    """Schedule a crawl task."""
    logging.info(f"Scheduling crawl for {domain}")
    run_scrapy_crawl(urls, domain, max_pages, collection, nlp, sentiment_analyzer, translator, context)

def run_scheduler():
    """Run the scheduler in a separate thread."""
    while True:
        schedule.run_pending()
        time.sleep(60)

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
@timeout_decorator.timeout(10, use_signals=False) 
def crawl():
    global embedding_model
    data = request.json
    crawl_type = data.get('crawl_type')
    input_value = data.get('input_value')
    max_pages = int(data.get('max_pages', 10))
    schedule_interval = data.get('schedule_interval', None)

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
        pages_collection, query_collection = setup_mongodb()
        azure_client = setup_azure_openai()
        embedding_model = setup_embedding_model()
        nlp = setup_nlp()
        sentiment_analyzer = setup_sentiment_analyzer()
        translator = setup_azure_translator()

        # Run Scrapy crawl
        visited_urls = run_scrapy_crawl(urls, domain, max_pages, pages_collection, nlp,
                                        sentiment_analyzer, translator, context)

        # Schedule crawl if interval provided
        if schedule_interval:
            if schedule_interval == "daily":
                schedule.every().day.at("00:00").do(
                    schedule_crawl, urls, domain, max_pages, pages_collection, nlp,
                    sentiment_analyzer, translator, context
                )
            elif schedule_interval == "hourly":
                schedule.every().hour.do(
                    schedule_crawl, urls, domain, max_pages, pages_collection, nlp,
                    sentiment_analyzer, translator, context
                )
            logging.info(f"Crawl scheduled {schedule_interval} for {domain}")

        # Generate report immediately
        report = report_agent(pages_collection, azure_client, domain, context)

        initial_state = CrawlerState(
            urls_to_visit=[],
            visited_urls=visited_urls,
            content_vectors=[],
            priority_queue=[],
            current_content=None,
            page_count=len(visited_urls),
            max_pages=max_pages,
            domain=domain,
            status="initial",
            entities=[],
            sentiments=[],
            languages=[]
        )
        workflow_app = create_workflow(pages_collection)
        crawl_start_time = datetime.now()
        final_state = initial_state
        for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
            for node_name, result_state in step_output.items():
                if isinstance(result_state, dict) and 'page_count' in result_state:
                    final_state = result_state
        crawl_end_time = datetime.now()

        return jsonify({
            "report": report,
            "visited_urls": visited_urls,
            "crawl_time": str(crawl_end_time - crawl_start_time),
            "domain": domain,
            "context": context
        })
    except TimeoutError:
        logging.error("Crawl timed out after 5 minutes")
        return jsonify({"error": "Crawl timed out after 5 minutes"}), 504
    except Exception as e:
        logging.error(f"Crawl error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    global embedding_model
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
            const [scheduleInterval, setScheduleInterval] = React.useState('');
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
                        body: JSON.stringify({
                            crawl_type: crawlType,
                            input_value: inputValue,
                            max_pages: maxPages,
                            schedule_interval: scheduleInterval
                        })
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

            const handleExport = () => {
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
                    <div className="mb-4">
                        <label className="block text-lg mb-2">Schedule Crawl</label>
                        <select
                            className="w-full p-2 border rounded"
                            value={scheduleInterval}
                            onChange={(e) => setScheduleInterval(e.target.value)}
                        >
                            <option value="">None</option>
                            <option value="hourly">Hourly</option>
                            <option value="daily">Daily</option>
                        </select>
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
                                                    Crawled: {update.url} (Page {update.page_count}/{update.max_pages}, Language: {update.language}, Sentiment: {update.sentiment.label}, Queued: {update.queued_urls})
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
                                className="bg-green-500 text-white p-2 rounded hover:bg-green-600 mb-2"
                                onClick={handleExport}
                            >
                                Export URLs to CSV
                            </button>
                            <ul className="list-disc pl-5">
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

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Main Execution
if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)