# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from pymongo import MongoClient
# import logging
# from urllib.parse import urljoin, urlparse
# from datetime import datetime

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_mongodb(db_name="crawler_db", collection_name="pages", mongo_uri="mongodb://localhost:27017/"):
#     """Connect to MongoDB and return the collection."""
#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
#         client.admin.command('ping')  # Test connection
#         db = client[db_name]
#         collection = db[collection_name]
#         logging.info(f"Connected to MongoDB at {mongo_uri}")
#         return collection
#     except () as e:
#         logging.error(f"MongoDB connection error: {e}")
#         raise

# def setup_driver():
#     """Configure and return a Selenium WebDriver instance."""
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument('--headless')
#         chrome_options.add_argument('--disable-gpu')
#         chrome_options.add_argument('--no-sandbox')
#         chrome_options.add_argument('--log-level=3')
#         service = Service(ChromeDriverManager().install(), log_path='nul')
#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         return driver
#     except Exception as e:
#         logging.error(f"WebDriver setup error: {e}")
#         raise

# def crawl_website(start_url, max_pages=10, timeout=10, mongo_uri="mongodb://localhost:27017/", extract_content=False):
#     """Crawl a website and store data in MongoDB."""
#     driver = setup_driver()
#     collection = setup_mongodb(mongo_uri=mongo_uri)
#     visited_urls = set()
#     urls_to_visit = [start_url]
#     page_count = 0
#     domain = urlparse(start_url).netloc

#     try:
#         while urls_to_visit and page_count < max_pages:
#             url = urls_to_visit.pop(0)
            
#             if url in visited_urls:
#                 continue

#             logging.info(f"Visiting: {url}")
#             try:
#                 driver.get(url)
#                 WebDriverWait(driver, timeout).until(
#                     EC.presence_of_element_located((By.TAG_NAME, "body"))
#                 )

#                 # Extract data
#                 title = driver.title or "No Title"
#                 content = ""
#                 if extract_content:
#                     try:
#                         content = driver.find_element(By.TAG_NAME, "body").text.strip()
#                     except Exception as e:
#                         logging.warning(f"Could not extract content for {url}: {e}")

#                 # Store in MongoDB
#                 page_data = {
#                     "url": url,
#                     "title": title,
#                     "content": content if extract_content else None,
#                     "crawled_at": datetime.utcnow().isoformat(),
#                     "domain": domain
#                 }
#                 collection.update_one(
#                     {"url": url},
#                     {"$set": page_data},
#                     upsert=True
#                 )
#                 logging.info(f"Stored data for {url} in MongoDB")

#                 # Extract links
#                 links = driver.find_elements(By.TAG_NAME, 'a')
#                 for link in links:
#                     try:
#                         href = link.get_attribute('href')
#                         if href:
#                             full_url = urljoin(url, href)
#                             parsed_url = urlparse(full_url)
#                             if parsed_url.netloc == domain and full_url not in visited_urls and parsed_url.scheme in ['http', 'https']:
#                                 urls_to_visit.append(full_url)
#                     except Exception as e:
#                         logging.warning(f"Error processing link on {url}: {e}")

#                 visited_urls.add(url)
#                 page_count += 1

#             except Exception as e:
#                 logging.error(f"Error processing {url}: {e}")
#                 continue

#     except Exception as e:
#         logging.error(f"Crawler error: {e}")
#     finally:
#         driver.quit()

#     return visited_urls

# if __name__ == "__main__":
#     target_url = "https://www.python.org"
#     crawled_urls = crawl_website(
#         start_url=target_url,
#         max_pages=5,
#         extract_content=True  # Set to False to skip content extraction
#     )
#     logging.info(f"Crawled URLs: {crawled_urls}")


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pymongo import MongoClient
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    user_query: Optional[str]
    query_response: Optional[str]

def setup_mongodb(db_name="crawler_db", collection_name="pages", query_collection_name="queries", mongo_uri="mongodb://localhost:27017/"):
    """Connect to MongoDB and return the pages and queries collections."""
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logging.info("MongoDB connection successful.")
        return client[db_name][collection_name], client[db_name][query_collection_name]
    except Exception as e:
        logging.error(f"MongoDB connection error: {e}", exc_info=True)
        raise

def setup_driver():
    """Configure and return a headless ChromeDriver instance."""
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        log_path = os.devnull
        service = Service(ChromeDriverManager().install(), log_path=log_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logging.info("WebDriver setup successful.")
        return driver
    except Exception as e:
        logging.error(f"WebDriver setup error: {e}", exc_info=True)
        raise

def setup_azure_openai():
    """Configure and return Azure OpenAI client."""
    try:
        client = AzureChatOpenAI(
            openai_api_key=os.getenv("API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT"),
            deployment_name="gpt-4o-mini",
            api_version="2024-02-15-preview",
            temperature=0.7
        )
        logging.info(f"Azure OpenAI client setup successful. Client type: {type(client)}")
        return client
    except Exception as e:
        logging.error(f"Azure OpenAI setup error: {e}", exc_info=True)
        raise

def fetch_url(state: CrawlerState, driver, collection):
    """Node: Fetch and process a URL using Selenium"""
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
        driver.get(url_to_fetch)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        title = driver.title or "No Title"
        body_element = driver.find_elements(By.TAG_NAME, "body")
        content = body_element[0].text.strip() if body_element else ""
        collection.update_one(
            {"url": url_to_fetch},
            {"$set": {
                "url": url_to_fetch,
                "title": title,
                "content": content[:50000],
                "crawled_at": datetime.utcnow().isoformat(),
                "domain": state['domain']
            }},
            upsert=True
        )
        new_urls_found = []
        try:
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href:
                    try:
                        full_url = urljoin(url_to_fetch, href.strip())
                        parsed_url = urlparse(full_url)
                        if (parsed_url.netloc == state['domain'] and
                            parsed_url.scheme in ['http', 'https'] and
                            full_url not in state['visited_urls'] and
                            '#' not in full_url):
                            if full_url not in state['priority_queue'] and full_url not in state['urls_to_visit']:
                                new_urls_found.append(full_url)
                    except Exception as link_e:
                        logging.warning(f"Could not parse or process link '{href}': {link_e}")
        except Exception as find_link_e:
            logging.error(f"Error finding links on {url_to_fetch}: {find_link_e}")
        updated_state = {
            **state,
            "current_content": content,
            "urls_to_visit": list(set(state['urls_to_visit'] + new_urls_found)),
            "visited_urls": state['visited_urls'] + [url_to_fetch],
            "page_count": state['page_count'] + 1,
            "priority_queue": state['priority_queue'],
            "status": "fetched"
        }
        logging.debug(f"Exiting fetch_url successfully. State: {updated_state}")
        return updated_state
    except Exception as e:
        logging.error(f"Error fetching {url_to_fetch}: {e}", exc_info=True)
        return {
            **state,
            "visited_urls": state['visited_urls'] + [url_to_fetch],
            "current_content": None,
            "priority_queue": state['priority_queue'],
            "status": "fetch_error"
        }

def analyze_content(state: CrawlerState):
    """Node: Analyze content using TF-IDF vectors."""
    logging.debug("Entering analyze_content.")
    if state.get("current_content"):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            vectors = vectorizer.fit_transform([state['current_content']])
            new_vector = vectors.toarray()[0]
            existing_vectors = state.get('content_vectors', [])
            updated_state = {**state, "content_vectors": existing_vectors + [new_vector]}
            logging.debug("Content analyzed successfully.")
            return updated_state
        except ValueError as e:
            logging.warning(f"TF-IDF Vectorization failed: {e}. Skipping content analysis.")
            return {**state}
        except Exception as e:
            logging.error(f"Unexpected error during content analysis: {e}", exc_info=True)
            return {**state}
    else:
        logging.debug("No current content to analyze.")
        return {**state}

def query_agent(state: CrawlerState, azure_client, query_collection):
    """Node: Process user query using Azure OpenAI with LangChain."""
    logging.debug("Entering query_agent.")
    if not state.get("user_query"):
        logging.debug("No user query provided.")
        return {**state, "query_response": None}

    try:
        context = state.get("current_content", "")[:10000]
        if not context:
            context = "No relevant content crawled yet."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", """
            You are an assistant that answers user queries based on crawled web content.
            Below is the relevant content from a webpage:

            {context}

            User Query: {query}

            Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
            """)
        ])

        chain = prompt_template | azure_client
        answer = chain.invoke({"context": context, "query": state['user_query']}).content.strip()

        query_collection.insert_one({
            "query": state['user_query'],
            "response": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "domain": state['domain']
        })

        logging.info(f"Query processed: {state['user_query']}")
        logging.debug(f"Query response: {answer}")
        return {**state, "query_response": answer}
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return {**state, "query_response": f"Error processing query: {str(e)}"}

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
    """Node: Prioritize URLs from urls_to_visit into priority_queue."""
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

def create_workflow(driver, pages_collection, azure_client, query_collection):
    """Create LangGraph workflow with query agent."""
    workflow = StateGraph(CrawlerState)
    workflow.add_node("fetch_url", lambda state: fetch_url(state, driver, pages_collection))
    workflow.add_node("analyze_content", analyze_content)
    workflow.add_node("query_agent", lambda state: query_agent(state, azure_client, query_collection))
    workflow.add_node("prioritize_urls", prioritize_urls)
    workflow.set_entry_point("fetch_url")
    workflow.add_edge("fetch_url", "analyze_content")
    workflow.add_edge("analyze_content", "query_agent")
    workflow.add_edge("query_agent", "prioritize_urls")
    workflow.add_conditional_edges(
        "prioritize_urls",
        should_continue,
        {
            "continue": "fetch_url",
            "stop": END
        }
    )
    app = workflow.compile()
    logging.info("Workflow compiled.")
    return app

# --- Main Execution ---
driver = None
try:
    driver = setup_driver()
    pages_collection, query_collection = setup_mongodb()
    azure_client = setup_azure_openai()

    def run_crawler(start_url: str, max_pages: int = 10, user_query: Optional[str] = None):
        """Run the agentic crawler with query support."""
        parsed_url = urlparse(start_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logging.error(f"Invalid start URL: {start_url}")
            return [], None
        initial_state = CrawlerState(
            urls_to_visit=[],
            visited_urls=[],
            content_vectors=[],
            priority_queue=[start_url],
            current_content=None,
            page_count=0,
            max_pages=max_pages,
            domain=parsed_url.netloc,
            status="initial",
            user_query=user_query,
            query_response=None
        )
        workflow_app = create_workflow(driver, pages_collection, azure_client, query_collection)
        final_state = None
        logging.info(f"Starting crawl from: {start_url}, max pages: {max_pages}, query: {user_query}")
        for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
            for node_name, result_state in step_output.items():
                if isinstance(result_state, dict) and 'page_count' in result_state:
                    logging.info(f"Node '{node_name}' finished. Pages crawled: {result_state['page_count']}/{max_pages}. URLs in queue: {len(result_state.get('priority_queue', []))}")
                    if result_state.get("query_response"):
                        logging.info(f"Query response: {result_state['query_response']}")
                    final_state = result_state
                else:
                    logging.debug(f"Node '{node_name}' finished. Output type: {type(result_state)}")
        if final_state:
            logging.info(f"Crawling finished. Total pages visited: {len(final_state['visited_urls'])}")
            return final_state['visited_urls'], final_state.get('query_response', None)
        else:
            logging.warning("Crawling finished, but no final state was captured.")
            return initial_state.get('visited_urls', []), None

    if __name__ == "__main__":
        start_time = datetime.now()
        crawled_urls, query_response = run_crawler(
            start_url="https://www.bbc.com/",
            max_pages=10,
            user_query="What are the latest news headlines?"
        )
        end_time = datetime.now()
        logging.info(f"Finished crawling {len(crawled_urls)} URLs in {end_time - start_time}.")
        logging.info(f"Visited URLs:\n" + "\n".join(crawled_urls))
        if query_response:
            logging.info(f"Query response: {query_response}")

except Exception as e:
    logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
finally:
    if driver:
        try:
            driver.quit()
            logging.info("WebDriver quit successfully.")
        except Exception as e:
            logging.error(f"Error quitting WebDriver: {e}", exc_info=True)