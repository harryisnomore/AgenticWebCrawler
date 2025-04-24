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
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, List, Optional
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class CrawlerState(TypedDict):
#     urls_to_visit: List[str]
#     visited_urls: List[str]
#     content_vectors: List[np.ndarray]
#     priority_queue: List[str]
#     current_content: Optional[str]
#     page_count: int
#     max_pages: int
#     domain: str
#     status: Optional[str]
#     user_query: Optional[str]
#     query_response: Optional[str]

# def setup_mongodb(db_name="crawler_db", collection_name="pages", query_collection_name="queries", mongo_uri="mongodb://localhost:27017/"):
#     """Connect to MongoDB and return the pages and queries collections."""
#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
#         client.admin.command('ismaster')
#         logging.info("MongoDB connection successful.")
#         return client[db_name][collection_name], client[db_name][query_collection_name]
#     except Exception as e:
#         logging.error(f"MongoDB connection error: {e}", exc_info=True)
#         raise

# def setup_driver():
#     """Configure and return a headless ChromeDriver instance."""
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument('--headless')
#         chrome_options.add_argument('--disable-gpu')
#         chrome_options.add_argument('--no-sandbox')
#         chrome_options.add_argument('--disable-dev-shm-usage')
#         chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
#         log_path = os.devnull
#         service = Service(ChromeDriverManager().install(), log_path=log_path)
#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         logging.info("WebDriver setup successful.")
#         return driver
#     except Exception as e:
#         logging.error(f"WebDriver setup error: {e}", exc_info=True)
#         raise

# def setup_azure_openai():
#     """Configure and return Azure OpenAI client."""
#     try:
#         client = AzureChatOpenAI(
#             openai_api_key=os.getenv("API_KEY"),
#             azure_endpoint=os.getenv("ENDPOINT"),
#             deployment_name="gpt-4o-mini",
#             api_version="2024-02-15-preview",
#             temperature=0.7
#         )
#         logging.info(f"Azure OpenAI client setup successful. Client type: {type(client)}")
#         return client
#     except Exception as e:
#         logging.error(f"Azure OpenAI setup error: {e}", exc_info=True)
#         raise

# def setup_embedding_model():
#     """Initialize and return the embedding model."""
#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         logging.info("Embedding model setup successful.")
#         return model
#     except Exception as e:
#         logging.error(f"Embedding model setup error: {e}", exc_info=True)
#         raise

# def fetch_url(state: CrawlerState, driver, collection):
#     """Node: Fetch and process a URL using Selenium."""
#     logging.debug(f"Entering fetch_url. State: {state}")
#     if state["page_count"] >= state["max_pages"]:
#         state["status"] = "stop"
#         return "stop"

#     url_to_fetch = None
#     if state['priority_queue']:
#         url_to_fetch = state['priority_queue'].pop(0)
#     elif state['urls_to_visit']:
#         potential_url = state['urls_to_visit'].pop(0)
#         while potential_url in state['visited_urls'] and state['urls_to_visit']:
#             potential_url = state['urls_to_visit'].pop(0)
#         if potential_url not in state['visited_urls']:
#             url_to_fetch = potential_url

#     if not url_to_fetch:
#         logging.info("No more URLs to fetch.")
#         return {**state, "status": "queue_empty", "current_content": None}

#     try:
#         logging.info(f"Fetching ({state['page_count'] + 1}/{state['max_pages']}): {url_to_fetch}")
#         driver.get(url_to_fetch)
#         WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
#         title = driver.title or "No Title"
#         body_element = driver.find_elements(By.TAG_NAME, "body")
#         content = body_element[0].text.strip() if body_element else ""
#         collection.update_one(
#             {"url": url_to_fetch},
#             {"$set": {
#                 "url": url_to_fetch,
#                 "title": title,
#                 "content": content[:50000],
#                 "crawled_at": datetime.utcnow().isoformat(),
#                 "domain": state['domain']
#             }},
#             upsert=True
#         )
#         new_urls_found = []
#         try:
#             links = driver.find_elements(By.TAG_NAME, 'a')
#             for link in links:
#                 href = link.get_attribute('href')
#                 if href:
#                     try:
#                         full_url = urljoin(url_to_fetch, href.strip())
#                         parsed_url = urlparse(full_url)
#                         if (parsed_url.netloc == state['domain'] and
#                             parsed_url.scheme in ['http', 'https'] and
#                             full_url not in state['visited_urls'] and
#                             '#' not in full_url):
#                             if full_url not in state['priority_queue'] and full_url not in state['urls_to_visit']:
#                                 new_urls_found.append(full_url)
#                     except Exception as link_e:
#                         logging.warning(f"Could not parse or process link '{href}': {link_e}")
#         except Exception as find_link_e:
#             logging.error(f"Error finding links on {url_to_fetch}: {find_link_e}")
#         updated_state = {
#             **state,
#             "current_content": content,
#             "urls_to_visit": list(set(state['urls_to_visit'] + new_urls_found)),
#             "visited_urls": state['visited_urls'] + [url_to_fetch],
#             "page_count": state['page_count'] + 1,
#             "priority_queue": state['priority_queue'],
#             "status": "fetched"
#         }
#         logging.debug(f"Exiting fetch_url successfully. State: {updated_state}")
#         return updated_state
#     except Exception as e:
#         logging.error(f"Error fetching {url_to_fetch}: {e}", exc_info=True)
#         return {
#             **state,
#             "visited_urls": state['visited_urls'] + [url_to_fetch],
#             "current_content": None,
#             "priority_queue": state['priority_queue'],
#             "status": "fetch_error"
#         }

# def analyze_content(state: CrawlerState, embedding_model, collection):
#     """Node: Analyze content using TF-IDF and generate embeddings."""
#     logging.debug("Entering analyze_content.")
#     if state.get("current_content"):
#         try:
#             # TF-IDF Vectorization
#             vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#             vectors = vectorizer.fit_transform([state['current_content']])
#             new_vector = vectors.toarray()[0]
#             existing_vectors = state.get('content_vectors', [])

#             # Generate Embedding
#             embedding = embedding_model.encode([state['current_content']])[0]
#             # Store embedding in MongoDB
#             collection.update_one(
#                 {"url": state['visited_urls'][-1]},
#                 {"$set": {"embedding": embedding.tolist()}}
#             )

#             updated_state = {
#                 **state,
#                 "content_vectors": existing_vectors + [new_vector]
#             }
#             logging.debug("Content analyzed and embedding stored successfully.")
#             return updated_state
#         except ValueError as e:
#             logging.warning(f"TF-IDF Vectorization failed: {e}. Skipping content analysis.")
#             return {**state}
#         except Exception as e:
#             logging.error(f"Unexpected error during content analysis: {e}", exc_info=True)
#             return {**state}
#     else:
#         logging.debug("No current content to analyze.")
#         return {**state}

# def query_agent(state: CrawlerState, azure_client, query_collection, embedding_model, pages_collection):
#     """Node: Process user query using RAG with Azure OpenAI."""
#     logging.debug("Entering query_agent.")
#     if not state.get("user_query"):
#         logging.debug("No user query provided.")
#         return {**state, "query_response": None}

#     try:
#         # Generate query embedding
#         query_embedding = embedding_model.encode([state['user_query']])[0]

#         # Retrieve top-k relevant documents from MongoDB
#         documents = list(pages_collection.find({"embedding": {"$exists": True}}))
#         if not documents:
#             context = "No relevant content crawled yet."
#         else:
#             # Compute cosine similarity
#             scored_docs = []
#             for doc in documents:
#                 doc_embedding = np.array(doc['embedding'])
#                 similarity = 1 - cosine(query_embedding, doc_embedding)
#                 scored_docs.append((similarity, doc['content'][:5000]))  # Limit content size
#             # Sort by similarity and take top-k
#             scored_docs.sort(reverse=True)
#             top_k = min(3, len(scored_docs))  # Top 3 documents
#             context = "\n\n".join([doc[1] for doc in scored_docs[:top_k]])
#             if not context:
#                 context = "No relevant content found."

#         # Prepare prompt with retrieved context
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant that answers queries based on crawled web content."),
#             ("user", """
#             Below is relevant content from crawled webpages:

#             {context}

#             User Query: {query}

#             Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
#             """)
#         ])

#         chain = prompt_template | azure_client
#         answer = chain.invoke({"context": context, "query": state['user_query']}).content.strip()

#         query_collection.insert_one({
#             "query": state['user_query'],
#             "response": answer,
#             "timestamp": datetime.utcnow().isoformat(),
#             "domain": state['domain']
#         })

#         logging.info(f"Query processed: {state['user_query']}")
#         logging.debug(f"Query response: {answer}")
#         return {**state, "query_response": answer}
#     except Exception as e:
#         logging.error(f"Error processing query: {e}", exc_info=True)
#         return {**state, "query_response": f"Error processing query: {str(e)}"}

# def should_continue(state: CrawlerState):
#     """Decision node: Continue crawling?"""
#     logging.debug("Entering should_continue.")
#     if state['page_count'] >= state['max_pages']:
#         logging.info("Decision: Stop (max pages reached).")
#         return "stop"
#     if not state['priority_queue'] and not state['urls_to_visit']:
#         logging.info("Decision: Stop (no more URLs in queues).")
#         return "stop"
#     logging.debug("Decision: Continue.")
#     return "continue"

# def prioritize_urls(state: CrawlerState):
#     """Node: Prioritize URLs from urls_to_visit into priority_queue."""
#     logging.debug("Entering prioritize_urls.")
#     urls_to_consider = list(set(state['urls_to_visit']) - set(state['visited_urls']))
#     if not urls_to_consider:
#         logging.debug("No new URLs to prioritize.")
#         return {**state, "urls_to_visit": []}
#     sorted_urls = sorted(
#         urls_to_consider,
#         key=lambda x: len(urlparse(x).path.split('/')),
#         reverse=False
#     )
#     new_priority_queue = sorted_urls + state.get('priority_queue', [])
#     unique_priority_queue = []
#     seen = set()
#     for item in new_priority_queue:
#         if item not in seen:
#             unique_priority_queue.append(item)
#             seen.add(item)
#     MAX_QUEUE_SIZE = 1000
#     final_priority_queue = unique_priority_queue[:MAX_QUEUE_SIZE]
#     logging.info(f"Prioritized {len(urls_to_consider)} URLs. Priority queue size: {len(final_priority_queue)}")
#     updated_state = {
#         **state,
#         "priority_queue": final_priority_queue,
#         "urls_to_visit": []
#     }
#     return updated_state

# def create_workflow(driver, pages_collection, azure_client, embedding_model, query_collection):
#     """Create LangGraph workflow with RAG query agent."""
#     workflow = StateGraph(CrawlerState)
#     workflow.add_node("fetch_url", lambda state: fetch_url(state, driver, pages_collection))
#     workflow.add_node("analyze_content", lambda state: analyze_content(state, embedding_model, pages_collection))
#     workflow.add_node("query_agent", lambda state: query_agent(state, azure_client, query_collection, embedding_model, pages_collection))
#     workflow.add_node("prioritize_urls", prioritize_urls)
#     workflow.set_entry_point("fetch_url")
#     workflow.add_edge("fetch_url", "analyze_content")
#     workflow.add_edge("analyze_content", "query_agent")
#     workflow.add_edge("query_agent", "prioritize_urls")
#     workflow.add_conditional_edges(
#         "prioritize_urls",
#         should_continue,
#         {
#             "continue": "fetch_url",
#             "stop": END
#         }
#     )
#     app = workflow.compile()
#     logging.info("Workflow compiled.")
#     return app

# # --- Main Execution ---
# driver = None
# try:
#     driver = setup_driver()
#     pages_collection, query_collection = setup_mongodb()
#     azure_client = setup_azure_openai()
#     embedding_model = setup_embedding_model()

#     def run_crawler(start_url: str, max_pages: int = 10, user_query: Optional[str] = None):
#         """Run the agentic crawler with RAG query support."""
#         parsed_url = urlparse(start_url)
#         if not parsed_url.scheme or not parsed_url.netloc:
#             logging.error(f"Invalid start URL: {start_url}")
#             return [], None
#         initial_state = CrawlerState(
#             urls_to_visit=[],
#             visited_urls=[],
#             content_vectors=[],
#             priority_queue=[start_url],
#             current_content=None,
#             page_count=0,
#             max_pages=max_pages,
#             domain=parsed_url.netloc,
#             status="initial",
#             user_query=user_query,
#             query_response=None
#         )
#         workflow_app = create_workflow(driver, pages_collection, azure_client, embedding_model, query_collection)
#         final_state = None
#         logging.info(f"Starting crawl from: {start_url}, max pages: {max_pages}, query: {user_query}")
#         for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
#             for node_name, result_state in step_output.items():
#                 if isinstance(result_state, dict) and 'page_count' in result_state:
#                     logging.info(f"Node '{node_name}' finished. Pages crawled: {result_state['page_count']}/{max_pages}. URLs in queue: {len(result_state.get('priority_queue', []))}")
#                     if result_state.get("query_response"):
#                         logging.info(f"Query response: {result_state['query_response']}")
#                     final_state = result_state
#                 else:
#                     logging.debug(f"Node '{node_name}' finished. Output type: {type(result_state)}")
#         if final_state:
#             logging.info(f"Crawling finished. Total pages visited: {len(final_state['visited_urls'])}")
#             return final_state['visited_urls'], final_state.get('query_response', None)
#         else:
#             logging.warning("Crawling finished, but no final state was captured.")
#             return initial_state.get('visited_urls', []), None

#     if __name__ == "__main__":
#         start_time = datetime.now()
#         crawled_urls, query_response = run_crawler(
#             start_url="https://www.aajtak.in/",
#             max_pages=10,
#             user_query="What are the latest news headlines?"
#         )
#         end_time = datetime.now()
#         logging.info(f"Finished crawling {len(crawled_urls)} URLs in {end_time - start_time}.")
#         logging.info(f"Visited URLs:\n" + "\n".join(crawled_urls))
#         if query_response:
#             logging.info(f"Query response: {query_response}")

# except Exception as e:
#     logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
# finally:
#     if driver:
#         try:
#             driver.quit()
#             logging.info("WebDriver quit successfully.")
#         except Exception as e:
#             logging.error(f"Error quitting WebDriver: {e}", exc_info=True)










# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from pymongo import MongoClient
# from selenium.common.exceptions import TimeoutException, WebDriverException
# import logging
# from urllib.parse import urljoin, urlparse
# from datetime import datetime
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, List, Optional
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from tenacity import retry, stop_after_attempt, wait_exponential
# from googlesearch import search

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class CrawlerState(TypedDict):
#     urls_to_visit: List[str]
#     visited_urls: List[str]
#     content_vectors: List[np.ndarray]
#     priority_queue: List[str]
#     current_content: Optional[str]
#     page_count: int
#     max_pages: int
#     domain: str
#     status: Optional[str]

# def setup_mongodb(db_name="crawler_db", collection_name="pages", query_collection_name="queries", mongo_uri="mongodb://localhost:27017/"):
#     """Connect to MongoDB and return the pages and queries collections."""
#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
#         client.admin.command('ismaster')
#         logging.info("MongoDB connection successful.")
#         return client[db_name][collection_name], client[db_name][query_collection_name]
#     except Exception as e:
#         logging.error(f"MongoDB connection error: {e}", exc_info=True)
#         raise

# def setup_driver():
#     """Configure and return a headless ChromeDriver instance."""
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument('--headless')
#         chrome_options.add_argument('--disable-gpu')
#         chrome_options.add_argument('--no-sandbox')
#         chrome_options.add_argument('--disable-dev-shm-usage')
#         chrome_options.add_argument('--disable-logging')
#         chrome_options.add_argument('--log-level=3')
#         chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
#         chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
#         service = Service(ChromeDriverManager().install())
#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         driver.set_page_load_timeout(30)
#         logging.info("WebDriver setup successful.")
#         return driver
#     except Exception as e:
#         logging.error(f"WebDriver setup error: {e}", exc_info=True)
#         raise

# def setup_azure_openai():
#     """Configure and return Azure OpenAI client."""
#     try:
#         client = AzureChatOpenAI(
#             openai_api_key=os.getenv("API_KEY"),
#             azure_endpoint=os.getenv("ENDPOINT"),
#             deployment_name="gpt-4o-mini",
#             api_version="2024-02-15-preview",
#             temperature=0.7
#         )
#         logging.info(f"Azure OpenAI client setup successful. Client type: {type(client)}")
#         return client
#     except Exception as e:
#         logging.error(f"Azure OpenAI setup error: {e}", exc_info=True)
#         raise

# def setup_embedding_model():
#     """Initialize and return the embedding model."""
#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         logging.info("Embedding model setup successful.")
#         return model
#     except Exception as e:
#         logging.error(f"Embedding model setup error: {e}", exc_info=True)
#         raise

# def validate_url(url: str) -> bool:
#     """Validate if a URL is properly formatted."""
#     try:
#         parsed = urlparse(url)
#         return bool(parsed.scheme in ['http', 'https'] and parsed.netloc)
#     except Exception:
#         return False

# def fetch_urls_for_query(query: str, max_urls: int = 5) -> List[str]:
#     """Fetch relevant URLs for a product or company using Google search."""
#     try:
#         logging.info(f"Fetching URLs for query: {query}")
#         urls = []
#         for url in search(query, num_results=max_urls, lang="en"):
#             if validate_url(url):
#                 urls.append(url)
#         if not urls:
#             logging.warning(f"No URLs found for query: {query}. Using fallback sources.")
#             # Fallback to news sites
#             urls = [
#                 "https://www.bbc.com/news",
#                 "https://www.reuters.com/",
#                 "https://www.theguardian.com/"
#             ]
#         logging.info(f"Fetched {len(urls)} URLs: {urls}")
#         return urls[:max_urls]
#     except Exception as e:
#         logging.error(f"Error fetching URLs for {query}: {e}", exc_info=True)
#         return [
#             "https://www.bbc.com/news",
#             "https://www.reuters.com/",
#             "https://www.theguardian.com/"
#         ]

# def get_user_input() -> tuple[List[str], str, str]:
#     """Prompt user for crawl type and inputs, return URLs, domain, and context."""
#     print("\n=== Crawler Input ===")
#     print("What would you like to crawl?")
#     print("1. Specific website(s) (provide URL(s))")
#     print("2. Data for a product (provide product name)")
#     print("3. Data for a company (provide company name)")
    
#     while True:
#         choice = input("Enter your choice (1, 2, or 3): ").strip()
#         if choice not in ['1', '2', '3']:
#             print("Invalid choice. Please enter 1, 2, or 3.")
#             continue
        
#         if choice == '1':
#             urls_input = input("Enter URLs (comma-separated, e.g., https://example.com,https://example.org): ").strip()
#             urls = [url.strip() for url in urls_input.split(',') if url.strip()]
#             if not urls:
#                 print("No URLs provided. Please enter at least one URL.")
#                 continue
#             valid_urls = []
#             for url in urls:
#                 if validate_url(url):
#                     valid_urls.append(url)
#                 else:
#                     print(f"Invalid URL skipped: {url}")
#             if not valid_urls:
#                 print("No valid URLs provided. Please try again.")
#                 continue
#             domain = urlparse(valid_urls[0]).netloc
#             context = "website"
#             return valid_urls, domain, context
        
#         elif choice == '2':
#             product = input("Enter product name (e.g., Surface Pro): ").strip()
#             if not product:
#                 print("Product name cannot be empty. Please try again.")
#                 continue
#             urls = fetch_urls_for_query(f"{product} news", max_urls=5)
#             domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
#             context = f"product: {product}"
#             return urls, domain, context
        
#         else:  # choice == '3'
#             company = input("Enter company name (e.g., Microsoft): ").strip()
#             if not company:
#                 print("Company name cannot be empty. Please try again.")
#                 continue
#             urls = fetch_urls_for_query(f"{company} news", max_urls=5)
#             domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
#             context = f"company: {company}"
#             return urls, domain, context

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def fetch_url(state: CrawlerState, driver, collection):
#     """Node: Fetch and process a URL using Selenium with retry logic."""
#     logging.debug(f"Entering fetch_url. State: {state}")
#     if state["page_count"] >= state["max_pages"]:
#         state["status"] = "stop"
#         return "stop"

#     url_to_fetch = None
#     if state['priority_queue']:
#         url_to_fetch = state['priority_queue'].pop(0)
#     elif state['urls_to_visit']:
#         potential_url = state['urls_to_visit'].pop(0)
#         while potential_url in state['visited_urls'] and state['urls_to_visit']:
#             potential_url = state['urls_to_visit'].pop(0)
#         if potential_url not in state['visited_urls']:
#             url_to_fetch = potential_url

#     if not url_to_fetch:
#         logging.info("No more URLs to fetch.")
#         return {**state, "status": "queue_empty", "current_content": None}

#     try:
#         logging.info(f"Fetching ({state['page_count'] + 1}/{state['max_pages']}): {url_to_fetch}")
#         driver.get(url_to_fetch)
#         WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
#         title = driver.title or "No Title"
#         body_element = driver.find_elements(By.TAG_NAME, "body")
#         content = body_element[0].text.strip() if body_element else ""
#         collection.update_one(
#             {"url": url_to_fetch},
#             {"$set": {
#                 "url": url_to_fetch,
#                 "title": title,
#                 "content": content[:50000],
#                 "crawled_at": datetime.utcnow().isoformat(),
#                 "domain": state['domain']
#             }},
#             upsert=True
#         )
#         new_urls_found = []
#         try:
#             links = driver.find_elements(By.TAG_NAME, 'a')
#             for link in links:
#                 href = link.get_attribute('href')
#                 if href:
#                     try:
#                         full_url = urljoin(url_to_fetch, href.strip())
#                         parsed_url = urlparse(full_url)
#                         if (parsed_url.netloc == state['domain'] and
#                             parsed_url.scheme in ['http', 'https'] and
#                             full_url not in state['visited_urls'] and
#                             '#' not in full_url):
#                             if full_url not in state['priority_queue'] and full_url not in state['urls_to_visit']:
#                                 new_urls_found.append(full_url)
#                     except Exception as link_e:
#                         logging.warning(f"Could not parse or process link '{href}': {link_e}")
#         except Exception as find_link_e:
#             logging.error(f"Error finding links on {url_to_fetch}: {find_link_e}")
#         updated_state = {
#             **state,
#             "current_content": content,
#             "urls_to_visit": list(set(state['urls_to_visit'] + new_urls_found)),
#             "visited_urls": state['visited_urls'] + [url_to_fetch],
#             "page_count": state['page_count'] + 1,
#             "priority_queue": state['priority_queue'],
#             "status": "fetched"
#         }
#         logging.debug(f"Exiting fetch_url successfully. State: {updated_state}")
#         return updated_state
#     except (TimeoutException, WebDriverException) as e:
#         logging.error(f"Error fetching {url_to_fetch}: {e}", exc_info=True)
#         return {
#             **state,
#             "visited_urls": state['visited_urls'] + [url_to_fetch],
#             "current_content": None,
#             "priority_queue": state['priority_queue'],
#             "status": "fetch_error"
#         }

# def analyze_content(state: CrawlerState, embedding_model, collection):
#     """Node: Analyze content using TF-IDF and generate embeddings."""
#     logging.debug("Entering analyze_content.")
#     if state.get("current_content"):
#         try:
#             # TF-IDF Vectorization
#             vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#             vectors = vectorizer.fit_transform([state['current_content']])
#             new_vector = vectors.toarray()[0]
#             existing_vectors = state.get('content_vectors', [])

#             # Generate Embedding
#             embedding = embedding_model.encode([state['current_content']])[0]
#             # Store embedding in MongoDB
#             collection.update_one(
#                 {"url": state['visited_urls'][-1]},
#                 {"$set": {"embedding": embedding.tolist()}}
#             )

#             updated_state = {
#                 **state,
#                 "content_vectors": existing_vectors + [new_vector]
#             }
#             logging.debug("Content analyzed and embedding stored successfully.")
#             return updated_state
#         except ValueError as e:
#             logging.warning(f"TF-IDF Vectorization failed: {e}. Skipping content analysis.")
#             return {**state}
#         except Exception as e:
#             logging.error(f"Unexpected error during content analysis: {e}", exc_info=True)
#             return {**state}
#     else:
#         logging.debug("No current content to analyze.")
#         return {**state}

# def report_agent(pages_collection, azure_client, domain: str, context: str) -> str:
#     """Generate a detailed report based on crawled data."""
#     logging.debug("Entering report_agent.")
#     try:
#         # Retrieve all crawled documents for the domain
#         documents = list(pages_collection.find({"content": {"$exists": True}, "domain": domain}))
#         if not documents:
#             logging.warning("No content available to generate report.")
#             return f"# Crawl Report\n\nNo content was crawled for {context}. Please check the crawler logs."

#         # Aggregate content (limit to 5000 chars per document to manage token usage)
#         aggregated_content = "\n\n".join([doc['content'][:5000] for doc in documents])
#         if not aggregated_content.strip():
#             logging.warning("Aggregated content is empty.")
#             return f"# Crawl Report\n\nNo valid content was found for {context}."

#         # Extract key terms using TF-IDF
#         vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
#         tfidf_matrix = vectorizer.fit_transform([doc['content'] for doc in documents if doc['content']])
#         feature_names = vectorizer.get_feature_names_out()
#         key_terms = ", ".join(feature_names) if feature_names.size > 0 else "None"

#         # Prepare prompt for report generation
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "You are an analyst generating a detailed report based on crawled web content."),
#             ("user", """
#             Below is aggregated content from crawled webpages related to {context}:

#             {content}

#             Generate a detailed report with the following sections:
#             - **Overview**: Summarize the main focus or purpose of the content (1-2 sentences).
#             - **Key Findings**: List 3-5 key insights or topics found in the content (bullet points).
#             - **Crawl Statistics**: Provide stats like number of pages crawled and key terms identified.
#             - **Recommendations**: Suggest 1-2 ways to explore the content further (e.g., specific queries).

#             Format the report in Markdown for clarity.
#             """)
#         ])

#         chain = prompt_template | azure_client
#         report = chain.invoke({
#             "content": aggregated_content[:20000],  # Limit to manage token usage
#             "context": context
#         }).content.strip()

#         # Add Crawl Statistics
#         stats = f"""
# ## Crawl Statistics
# - **Pages Crawled**: {len(documents)}
# - **Key Terms Identified**: {key_terms}
# - **Crawl Timestamp**: {datetime.utcnow().isoformat()}
# """

#         final_report = f"# Crawl Report for {context}\n\n{report}\n\n{stats}"

#         # Save report to file
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         context_safe = context.replace(":", "_").replace(" ", "_")
#         filename = f"report_{context_safe}_{timestamp}.md"
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 f.write(final_report)
#             logging.info(f"Report saved to {filename}")
#         except Exception as e:
#             logging.error(f"Failed to save report to file: {e}", exc_info=True)

#         logging.info("Report generated successfully.")
#         return final_report
#     except Exception as e:
#         logging.error(f"Error generating report: {e}", exc_info=True)
#         return f"# Crawl Report\n\nError generating report for {context}. Please check the logs."

# def query_agent(query: str, azure_client, query_collection, embedding_model, pages_collection, domain: str) -> str:
#     """Process a single user query using RAG with Azure OpenAI."""
#     logging.debug("Entering query_agent.")
#     try:
#         # Generate query embedding
#         query_embedding = embedding_model.encode([query])[0]

#         # Retrieve top-k relevant documents from MongoDB
#         documents = list(pages_collection.find({"embedding": {"$exists": True}, "domain": domain}))
#         if not documents:
#             context = "No relevant content crawled yet."
#         else:
#             # Compute cosine similarity
#             scored_docs = []
#             for doc in documents:
#                 doc_embedding = np.array(doc['embedding'])
#                 similarity = 1 - cosine(query_embedding, doc_embedding)
#                 scored_docs.append((similarity, doc['content'][:5000]))  # Limit content size
#             # Sort by similarity and take top-k
#             scored_docs.sort(reverse=True)
#             top_k = min(3, len(scored_docs))  # Top 3 documents
#             context = "\n\n".join([doc[1] for doc in scored_docs[:top_k]])
#             if not context:
#                 context = "No relevant content found."

#         # Prepare prompt with retrieved context
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant that answers queries based on crawled web content."),
#             ("user", """
#             Below is relevant content from crawled webpages:

#             {context}

#             User Query: {query}

#             Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
#             """)
#         ])

#         chain = prompt_template | azure_client
#         answer = chain.invoke({"context": context, "query": query}).content.strip()

#         query_collection.insert_one({
#             "query": query,
#             "response": answer,
#             "timestamp": datetime.utcnow().isoformat(),
#             "domain": domain
#         })

#         logging.info(f"Query processed: {query}")
#         logging.debug(f"Query response: {answer}")
#         return answer
#     except Exception as e:
#         logging.error(f"Error processing query: {e}", exc_info=True)
#         return f"Error processing query: {str(e)}"

# def query_loop(azure_client, query_collection, embedding_model, pages_collection, domain: str):
#     """Interactive loop to handle user queries after crawling."""
#     logging.info("Entering query mode. Type 'exit' to quit.")
#     document_count = pages_collection.count_documents({"embedding": {"$exists": True}, "domain": domain})
#     if document_count == 0:
#         logging.warning("No documents with embeddings found in MongoDB. Queries may return limited results.")
#     while True:
#         query = input("Enter your query (or 'exit' to quit): ").strip()
#         if query.lower() == 'exit':
#             break
#         if not query:
#             logging.warning("Empty query provided.")
#             continue
#         start_time = datetime.now()
#         response = query_agent(query, azure_client, query_collection, embedding_model, pages_collection, domain)
#         end_time = datetime.now()
#         print(f"Query: {query}")
#         print(f"Response: {response}")
#         logging.info(f"Query executed in {end_time - start_time}.")

# def should_continue(state: CrawlerState):
#     """Decision node: Continue crawling?"""
#     logging.debug("Entering should_continue.")
#     if state['page_count'] >= state['max_pages']:
#         logging.info("Decision: Stop (max pages reached).")
#         return "stop"
#     if not state['priority_queue'] and not state['urls_to_visit']:
#         logging.info("Decision: Stop (no more URLs in queues).")
#         return "stop"
#     logging.debug("Decision: Continue.")
#     return "continue"

# def prioritize_urls(state: CrawlerState):
#     """Node: Prioritize URLs from urls_to_visit into priority_queue."""
#     logging.debug("Entering prioritize_urls.")
#     urls_to_consider = list(set(state['urls_to_visit']) - set(state['visited_urls']))
#     if not urls_to_consider:
#         logging.debug("No new URLs to prioritize.")
#         return {**state, "urls_to_visit": []}
#     sorted_urls = sorted(
#         urls_to_consider,
#         key=lambda x: len(urlparse(x).path.split('/')),
#         reverse=False
#     )
#     new_priority_queue = sorted_urls + state.get('priority_queue', [])
#     unique_priority_queue = []
#     seen = set()
#     for item in new_priority_queue:
#         if item not in seen:
#             unique_priority_queue.append(item)
#             seen.add(item)
#     MAX_QUEUE_SIZE = 1000
#     final_priority_queue = unique_priority_queue[:MAX_QUEUE_SIZE]
#     logging.info(f"Prioritized {len(urls_to_consider)} URLs. Priority queue size: {len(final_priority_queue)}")
#     updated_state = {
#         **state,
#         "priority_queue": final_priority_queue,
#         "urls_to_visit": []
#     }
#     return updated_state

# def create_workflow(driver, pages_collection, embedding_model):
#     """Create LangGraph workflow for crawling and embedding."""
#     workflow = StateGraph(CrawlerState)
#     workflow.add_node("fetch_url", lambda state: fetch_url(state, driver, pages_collection))
#     workflow.add_node("analyze_content", lambda state: analyze_content(state, embedding_model, pages_collection))
#     workflow.add_node("prioritize_urls", prioritize_urls)
#     workflow.set_entry_point("fetch_url")
#     workflow.add_edge("fetch_url", "analyze_content")
#     workflow.add_edge("analyze_content", "prioritize_urls")
#     workflow.add_conditional_edges(
#         "prioritize_urls",
#         should_continue,
#         {
#             "continue": "fetch_url",
#             "stop": END
#         }
#     )
#     app = workflow.compile()
#     logging.info("Workflow compiled.")
#     return app

# def run_crawler(start_urls: List[str], domain: str, context: str, max_pages: int = 10):
#     """Run the agentic crawler, generate a report, and allow querying."""
#     if not start_urls:
#         logging.error("No start URLs provided.")
#         return []

#     initial_state = CrawlerState(
#         urls_to_visit=[],
#         visited_urls=[],
#         content_vectors=[],
#         priority_queue=start_urls,
#         current_content=None,
#         page_count=0,
#         max_pages=max_pages,
#         domain=domain,
#         status="initial"
#     )
#     driver = setup_driver()
#     pages_collection, query_collection = setup_mongodb()
#     azure_client = setup_azure_openai()
#     embedding_model = setup_embedding_model()
    
#     workflow_app = create_workflow(driver, pages_collection, embedding_model)
#     final_state = None
#     logging.info(f"Starting crawl for {context}, URLs: {start_urls}, max pages: {max_pages}")
#     crawl_start_time = datetime.now()
#     try:
#         for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
#             for node_name, result_state in step_output.items():
#                 if isinstance(result_state, dict) and 'page_count' in result_state:
#                     logging.info(f"Node '{node_name}' finished. Pages crawled: {result_state['page_count']}/{max_pages}. URLs in queue: {len(result_state.get('priority_queue', []))}")
#                     final_state = result_state
#                 else:
#                     logging.debug(f"Node '{node_name}' finished. Output type: {type(result_state)}")
#     except Exception as e:
#         logging.error(f"Crawling interrupted: {e}", exc_info=True)
#     finally:
#         if driver:
#             try:
#                 driver.quit()
#                 logging.info("WebDriver quit successfully.")
#             except Exception as e:
#                 logging.error(f"Error quitting WebDriver: {e}", exc_info=True)
    
#     crawl_end_time = datetime.now()
#     if final_state:
#         logging.info(f"Crawling finished. Total pages visited: {len(final_state['visited_urls'])} in {crawl_end_time - crawl_start_time}.")
#         # Generate and present report
#         print("\n=== Crawl Report ===")
#         report = report_agent(pages_collection, azure_client, final_state['domain'], context)
#         print(report)
#         print("===================\n")
#         # Enter query loop
#         query_loop(azure_client, query_collection, embedding_model, pages_collection, final_state['domain'])
#         return final_state['visited_urls']
#     else:
#         logging.warning("Crawling finished, but no final state was captured.")
#         # Generate report and allow querying even if partial crawl
#         print("\n=== Crawl Report ===")
#         report = report_agent(pages_collection, azure_client, domain, context)
#         print(report)
#         print("===================\n")
#         query_loop(azure_client, query_collection, embedding_model, pages_collection, domain)
#         return initial_state.get('visited_urls', [])

# # --- Main Execution ---
# if __name__ == "__main__":
#     try:
#         start_time = datetime.now()
#         urls, domain, context = get_user_input()
#         crawled_urls = run_crawler(
#             start_urls=urls,
#             domain=domain,
#             context=context,
#             max_pages=10
#         )
#         end_time = datetime.now()
#         logging.info(f"Finished crawling {len(crawled_urls)} URLs in {end_time - start_time}.")
#         logging.info(f"Visited URLs:\n" + "\n".join(crawled_urls))
#     except Exception as e:
#         logging.critical(f"An unrecoverable error occurred: {e}", exc_info=True)






from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import threading
import queue
import os
import markdown2
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pymongo import MongoClient
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging
from urllib.parse import urljoin, urlparse
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from googlesearch import search
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Global variables for crawler state
crawler_thread = None
crawler_queue = queue.Queue()
crawler_status = {"status": "idle", "pages_crawled": 0, "total_pages": 0, "logs": [], "visited_urls": [], "domain": "", "context": ""}

# Global resources
PAGES_COLLECTION = None
QUERY_COLLECTION = None
AZURE_CLIENT = None
EMBEDDING_MODEL = None
LANGFUSE = None

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

# Setup functions
def setup_mongodb(db_name="crawler_db", collection_name="pages", query_collection_name="queries", mongo_uri="mongodb://localhost:27017/"):
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logging.info("MongoDB connection successful.")
        return client[db_name][collection_name], client[db_name][query_collection_name]
    except Exception as e:
        logging.error(f"MongoDB connection error: {e}", exc_info=True)
        raise

def setup_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        logging.info("WebDriver setup successful.")
        return driver
    except Exception as e:
        logging.error(f"WebDriver setup error: {e}", exc_info=True)
        raise

def setup_azure_openai():
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

def setup_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Embedding model setup successful.")
        return model
    except Exception as e:
        logging.error(f"Embedding model setup error: {e}", exc_info=True)
        raise

def setup_langfuse():
    try:
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
        )
        logging.info("Langfuse client setup successful.")
        return langfuse
    except Exception as e:
        logging.error(f"Langfuse setup error: {e}", exc_info=True)
        raise

def setup_global_resources():
    global PAGES_COLLECTION, QUERY_COLLECTION, AZURE_CLIENT, EMBEDDING_MODEL, LANGFUSE
    PAGES_COLLECTION, QUERY_COLLECTION = setup_mongodb()
    AZURE_CLIENT = setup_azure_openai()
    EMBEDDING_MODEL = setup_embedding_model()
    LANGFUSE = setup_langfuse()

def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ['http', 'https'] and parsed.netloc)
    except Exception:
        return False

def fetch_urls_for_query(query: str, max_urls: int = 5) -> List[str]:
    try:
        logging.info(f"Fetching URLs for query: {query}")
        urls = []
        for url in search(query, num_results=max_urls, lang="en"):
            if validate_url(url):
                urls.append(url)
        if not urls:
            logging.warning(f"No URLs found for query: {query}. Using fallback sources.")
            urls = ["https://www.bbc.com/news", "https://www.reuters.com/", "https://www.theguardian.com/"]
        logging.info(f"Fetched {len(urls)} URLs: {urls}")
        return urls[:max_urls]
    except Exception as e:
        logging.error(f"Error fetching URLs for {query}: {e}", exc_info=True)
        return ["https://www.bbc.com/news", "https://www.reuters.com/", "https://www.theguardian.com/"]

# Crawler logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_url(state: CrawlerState, driver, collection):
    global crawler_status
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
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
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
        crawler_status["pages_crawled"] = updated_state["page_count"]
        crawler_status["visited_urls"] = updated_state["visited_urls"]
        crawler_status["logs"].append(f"Fetched: {url_to_fetch}")
        logging.debug(f"Exiting fetch_url successfully. State: {updated_state}")
        return updated_state
    except (TimeoutException, WebDriverException) as e:
        logging.error(f"Error fetching {url_to_fetch}: {e}", exc_info=True)
        crawler_status["logs"].append(f"Error fetching {url_to_fetch}: {str(e)}")
        return {
            **state,
            "visited_urls": state['visited_urls'] + [url_to_fetch],
            "current_content": None,
            "priority_queue": state['priority_queue'],
            "status": "fetch_error"
        }

def analyze_content(state: CrawlerState, embedding_model, collection):
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
            crawler_status["logs"].append("Content analyzed and embedding stored.")
            logging.debug("Content analyzed and embedding stored successfully.")
            return updated_state
        except ValueError as e:
            logging.warning(f"TF-IDF Vectorization failed: {e}. Skipping content analysis.")
            crawler_status["logs"].append(f"TF-IDF Vectorization failed: {e}")
            return {**state}
        except Exception as e:
            logging.error(f"Unexpected error during content analysis: {e}", exc_info=True)
            crawler_status["logs"].append(f"Content analysis error: {str(e)}")
            return {**state}
    else:
        logging.debug("No current content to analyze.")
        return {**state}

def report_agent(domain: str, context: str, langfuse_handler: CallbackHandler) -> str:
    logging.debug("Entering report_agent.")
    try:
        documents = list(PAGES_COLLECTION.find({"content": {"$exists": True}, "domain": domain}))
        if not documents:
            logging.warning("No content available to generate report.")
            raw_report = f"No content was crawled for {context}. Please check the crawler logs."
            return markdown2.markdown(raw_report)

        aggregated_content = "\n\n".join([doc['content'][:5000] for doc in documents])
        if not aggregated_content.strip():
            logging.warning("Aggregated content is empty.")
            raw_report = f"No valid content was found for {context}."
            return markdown2.markdown(raw_report)

        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([doc['content'] for doc in documents if doc['content']])
        feature_names = vectorizer.get_feature_names_out()
        key_terms = ", ".join(feature_names) if feature_names.size > 0 else "None"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
              "You are an expert research analyst. Your task is to generate a deeply analytical, content-rich report based on aggregated data from multiple crawled web pages. Your writing should be highly detailed, data-informed, and structured logically. Your analysis must synthesize, interpret, and draw insight from patterns, discrepancies, and trends found in the source content."),
            ("user", 
              """
              Below is aggregated textual content extracted from a recent web crawl on the topic: {context}

              CONTENT:
              {content}

              Generate a comprehensive report with **detailed sections**:
              
              -Be concise, structured, and eliminate redundant line breaks or whitespace also remove the unnecessary dead spaces(important). Understand and remove the specific spacing problem , there are large gaps between bullet points and list sections. Remove large vertical gaps between sections of the report 
              -Present the content to the user beautifully and in a structured manner . Use Markdown     formatting for clarity and emphasis. Ensure that the report is well-organized, with clear  headings and subheadings for each section. Use bullet points, tables, or other visual aids where appropriate to enhance readability and comprehension.
              -Ensure that the report is not only informative but also engaging and easy to follow. The goal is to provide a comprehensive overview of the content.
              """
            )
        ])

        chain = prompt_template | AZURE_CLIENT
        report = chain.invoke({
            "content": aggregated_content[:20000],
            "context": context
        }, config={"callbacks": [langfuse_handler]}).content.strip()

        stats = f"""
Crawl Statistics
- Pages Crawled: {len(documents)}
- Key Terms Identified: {key_terms}
- Crawl Timestamp: {datetime.utcnow().isoformat()}
"""

        final_report = f"Crawl Report for {context}\n\n{report}\n\n{stats}"
        html_report = markdown2.markdown(final_report)
        cleaned_report = re.sub(r'[#*]+', '', html_report)
        logging.info("Report generated successfully.")
        return cleaned_report
    except Exception as e:
        logging.error(f"Error generating report: {e}", exc_info=True)
        raw_report = f"Error generating report for {context}. Please check the logs."
        return markdown2.markdown(raw_report)

def query_agent(query: str, domain: str, langfuse_handler: CallbackHandler) -> str:
    """
    Processes a user query by retrieving relevant documents, generating a response using a language model,
    and storing the query-response pair.

    Args:
        query (str): The user's query string.
        domain (str): The domain to search for relevant documents.
        langfuse_handler (CallbackHandler): Handler for logging and monitoring.

    Returns:
        str: The generated response or an error message.
    """
    logging.debug("🚀 Entering query_agent.")
    try:
        # Encode the query into an embedding
        query_embedding = EMBEDDING_MODEL.encode([query])[0]
        documents = list(PAGES_COLLECTION.find({"embedding": {"$exists": True}, "domain": domain}))

        # Check if documents exist
        if not documents:
            context = "🔍 No relevant content crawled yet."
        else:
            # Calculate similarity scores for documents
            scored_docs = []
            for doc in documents:
                doc_embedding = np.array(doc['embedding'])
                similarity = 1 - cosine(query_embedding, doc_embedding)
                scored_docs.append((similarity, doc['content'][:5000]))
            scored_docs.sort(reverse=True)

            # Select top-k documents (up to 3)
            top_k = min(3, len(scored_docs))
            context = "\n\n".join([doc[1] for doc in scored_docs[:top_k]])
            if not context:
                context = "🔎 No relevant content found."

        # Define the prompt template for the language model
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "🤖 You are a helpful assistant that answers queries based on crawled web content."),
            ("user", """
            📜 Below is relevant content from crawled webpages:

            {context}

            ❓ User Query: {query}

            Provide a concise and accurate answer based on the content above. If the content doesn't contain enough information, say so and provide a general response if possible.
            """)
        ])

        # Invoke the language model with the prompt and context
        chain = prompt_template | AZURE_CLIENT
        answer = chain.invoke(
            {"context": context, "query": query},
            config={"callbacks": [langfuse_handler]}
        ).content.strip()

        # Store the query and response in the database
        QUERY_COLLECTION.insert_one({
            "query": query,
            "response": answer,
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain
        })

        logging.info(f"✅ Query processed: {query}")
        logging.debug(f"📝 Query response: {answer}")
        return answer

    except Exception as e:
        logging.error(f"❌ Error processing query: {e}", exc_info=True)
        return f"⚠️ Error processing query: {str(e)}"

def should_continue(state: CrawlerState):
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

def create_workflow(driver, pages_collection, embedding_model):
    workflow = StateGraph(CrawlerState)
    workflow.add_node("fetch_url", lambda state: fetch_url(state, driver, pages_collection))
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

def run_crawler(start_urls: List[str], domain: str, context: str, max_pages: int = 10):
    global crawler_status
    if not start_urls:
        logging.error("No start URLs provided.")
        crawler_status["logs"].append("Error: No start URLs provided.")
        return []

    initial_state = CrawlerState(
        urls_to_visit=[],
        visited_urls=[],
        content_vectors=[],
        priority_queue=start_urls,
        current_content=None,
        page_count=0,
        max_pages=max_pages,
        domain=domain,
        status="initial"
    )
    driver = setup_driver()
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )
    
    workflow_app = create_workflow(driver, PAGES_COLLECTION, EMBEDDING_MODEL)
    final_state = None
    logging.info(f"Starting crawl for {context}, URLs: {start_urls}, max pages: {max_pages}")
    crawler_status["logs"].append(f"Starting crawl for {context}")
    crawl_start_time = datetime.now()
    try:
        for step_output in workflow_app.stream(initial_state, {"recursion_limit": max_pages + 50}):
            for node_name, result_state in step_output.items():
                if isinstance(result_state, dict) and 'page_count' in result_state:
                    logging.info(f"Node '{node_name}' finished. Pages crawled: {result_state['page_count']}/{max_pages}. URLs in queue: {len(result_state.get('priority_queue', []))}")
                    final_state = result_state
                    crawler_status["pages_crawled"] = final_state["page_count"]
                    crawler_status["total_pages"] = final_state["max_pages"]
                    crawler_status["visited_urls"] = final_state["visited_urls"]
                else:
                    logging.debug(f"Node '{node_name}' finished. Output type: {type(result_state)}")
    except Exception as e:
        logging.error(f"Crawling interrupted: {e}", exc_info=True)
        crawler_status["logs"].append(f"Crawling interrupted: {str(e)}")
    finally:
        if driver:
            try:
                driver.quit()
                logging.info("WebDriver quit successfully.")
                crawler_status["logs"].append("WebDriver quit successfully.")
            except Exception as e:
                logging.error(f"Error quitting WebDriver: {e}", exc_info=True)
                crawler_status["logs"].append(f"Error quitting WebDriver: {str(e)}")
        LANGFUSE.flush()
    
    crawl_end_time = datetime.now()
    if final_state:
        logging.info(f"Crawling finished. Total pages visited: {len(final_state['visited_urls'])} in {crawl_end_time - crawl_start_time}.")
        crawler_status["logs"].append(f"Crawling finished. Total pages visited: {len(final_state['visited_urls'])}")
        return final_state['visited_urls']
    else:
        logging.warning("Crawling finished, but no final state was captured.")
        crawler_status["logs"].append("Crawling finished, but no final state was captured.")
        return initial_state.get('visited_urls', [])

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_crawler', methods=['POST'])
def start_crawler():
    global crawler_thread, crawler_status
    data = request.json
    crawl_type = data.get('crawl_type')
    max_pages = int(data.get('max_pages', 10))
    
    if crawler_status["status"] in ["running", "starting"]:
        return jsonify({"error": "Crawler is already running."}), 400

    crawler_status = {
        "status": "starting",
        "pages_crawled": 0,
        "total_pages": max_pages,
        "logs": [],
        "visited_urls": [],
        "domain": "",
        "context": ""
    }

    if crawl_type == 'website':
        urls = [url.strip() for url in data.get('urls', '').split(',') if url.strip()]
        if not urls or not all(validate_url(url) for url in urls):
            return jsonify({"error": "Invalid or empty URLs provided."}), 400
        domain = urlparse(urls[0]).netloc
        context = "website"
    elif crawl_type == 'product':
        product = data.get('product', '').strip()
        if not product:
            return jsonify({"error": "Product name is required."}), 400
        urls = fetch_urls_for_query(f"{product} news", max_urls=5)
        domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
        context = f"product: {product}"
    elif crawl_type == 'company':
        company = data.get('company', '').strip()
        if not company:
            return jsonify({"error": "Company name is required."}), 400
        urls = fetch_urls_for_query(f"{company} news", max_urls=5)
        domain = urlparse(urls[0]).netloc if urls else "multiple_domains"
        context = f"company: {company}"
    else:
        return jsonify({"error": "Invalid crawl type."}), 400

    crawler_status["domain"] = domain
    crawler_status["context"] = context
    crawler_status["status"] = "running"

    def run_crawler_thread():
        try:
            run_crawler(urls, domain, context, max_pages)
            crawler_status["status"] = "completed"
            crawler_status["logs"].append("Crawler completed successfully.")
        except Exception as e:
            crawler_status["status"] = "error"
            crawler_status["logs"].append(f"Error: {str(e)}")
        finally:
            crawler_queue.put("done")

    crawler_thread = threading.Thread(target=run_crawler_thread)
    crawler_thread.start()
    return jsonify({"message": "Crawler started.", "status": crawler_status})

@app.route('/crawler_status', methods=['GET'])
def get_crawler_status():
    return jsonify(crawler_status)

@app.route('/stop_crawler', methods=['POST'])
def stop_crawler():
    global crawler_status
    if crawler_status["status"] != "running":
        return jsonify({"error": "No crawler is running."}), 400
    crawler_status["total_pages"] = crawler_status["pages_crawled"]
    crawler_status["status"] = "stopping"
    crawler_status["logs"].append("Crawler stopping.")
    return jsonify({"message": "Crawler stopping."})

@app.route('/get_report', methods=['GET'])
def get_report():
    domain = request.args.get('domain', crawler_status["domain"])
    context = request.args.get('context', crawler_status["context"])
    if not domain or not context:
        return jsonify({"error": "No crawl data available."}), 400
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )
    report = report_agent(domain, context, langfuse_handler)
    return jsonify({"report": report})

@app.route('/download_report', methods=['GET'])
def download_report():
    domain = request.args.get('domain', crawler_status["domain"])
    context = request.args.get('context', crawler_status["context"])
    if not domain or not context:
        return jsonify({"error": "No crawl data available."}), 400
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )
    report = report_agent(domain, context, langfuse_handler)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_safe = context.replace(":", "_").replace(" ", "_")
    filename = f"report_{context_safe}_{timestamp}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    return send_file(filename, as_attachment=True)

@app.route('/query', methods=['POST'])
def query():
    """
    API endpoint to handle query requests.

    Expects JSON payload with 'query' and optional 'domain' fields.
    Returns the query response and metadata in JSON format.
    """
    data = request.json
    query_text = data.get('query', '').strip()
    domain = data.get('domain', crawler_status["domain"])

    # Validate input
    if not query_text:
        return jsonify({"error": "🚫 Query cannot be empty."}), 400
    if not domain:
        return jsonify({"error": "🚫 No crawl data available."}), 400

    # Initialize Langfuse handler for logging
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )

    # Process the query and return the response
    response = query_agent(query_text, domain, langfuse_handler)
    return jsonify({
        "query": query_text,
        "response": response,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/query_history', methods=['GET'])
def query_history():
    domain = request.args.get('domain', crawler_status["domain"])
    if not domain:
        return jsonify({"error": "No crawl data available."}), 400
    queries = list(QUERY_COLLECTION.find({"domain": domain}).sort("timestamp", -1).limit(10))
    return jsonify([{"query": q["query"], "response": q["response"], "timestamp": q["timestamp"]} for q in queries])

if __name__ == "__main__":
    setup_global_resources()
    app.run(debug=True, host='0.0.0.0', port=5000)
