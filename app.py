import re
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import requests
import time
import logging
import wikipedia
import concurrent.futures
from bs4 import BeautifulSoup, Tag
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import streamlit as st

load_dotenv()

# --- Error Handling Helper ---
class RateLimitException(Exception):
    pass

def try_api_call(api_func, query, max_retries=2, backoff_factor=2):
    """Attempts an API call with retries and exponential backoff on rate limit."""
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            result = api_func(query)
            if result:
                return result
        except RateLimitException as e:
            logging.warning(f"{api_func.__name__} rate limited: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= backoff_factor
        except Exception as e:
            logging.error(f"{api_func.__name__} failed: {e}")
            break
    return None

# --- News APIs ---
def fetch_from_newsapi(query):
    #print("[DEBUG] Hitting NewsAPI with query:", query)
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logging.error("NEWSAPI_KEY not set.")
        #print("[DEBUG] NewsAPI: API key not set.")
        return None
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        logging.error(f"NewsAPI request failed: {e}")
        #print("[DEBUG] NewsAPI: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] NewsAPI returned results.")
        return [
            {
                "title": a["title"],
                "url": a["url"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"],
                "content": a.get("content", "")
            }
            for a in data.get("articles", [])
        ]
    elif resp.status_code == 429:
        #print("[DEBUG] NewsAPI: rate limit hit.")
        raise RateLimitException("NewsAPI rate limit hit.")
    else:
        logging.error(f"NewsAPI error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] NewsAPI error: {resp.status_code}")
    return None

def fetch_from_guardian(query):
    #print("[DEBUG] Hitting Guardian API with query:", query)
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        logging.error("GUARDIAN_API_KEY not set.")
        #print("[DEBUG] Guardian API: API key not set.")
        return None
    url = f"https://content.guardianapis.com/search?q={query}&api-key={api_key}&show-fields=all&page-size=5"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        logging.error(f"Guardian API request failed: {e}")
        #print("[DEBUG] Guardian API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] Guardian API returned results.")
        return [
            {
                "title": r['webTitle'],
                "url": r['webUrl'],
                "source": "The Guardian",
                "publishedAt": r['webPublicationDate'],
                "content": r['fields'].get('bodyText', '') if 'fields' in r else ''
            }
            for r in data.get('response', {}).get('results', [])
        ]
    elif resp.status_code == 429:
        #print("[DEBUG] Guardian API: rate limit hit.")
        raise RateLimitException("Guardian API rate limit hit.")
    else:
        logging.error(f"Guardian API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] Guardian API error: {resp.status_code}")
    return None

def fetch_from_serper(query):
    #print("[DEBUG] Hitting Serper API with query:", query)
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logging.error("SERPER_API_KEY not set.")
        #print("[DEBUG] Serper API: API key not set.")
        return None
    url = "https://google.serper.dev/news"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json={"q": query}, timeout=5)
    except Exception as e:
        logging.error(f"Serper API request failed: {e}")
        #print("[DEBUG] Serper API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] Serper API returned results.")
        return [
            {
                "title": n["title"],
                "url": n["link"],
                "source": n.get("source", ""),
                "publishedAt": n.get("date", ""),
                "content": n.get("snippet", "")
            }
            for n in data.get("news", [])
        ]
    elif resp.status_code == 429:
        #print("[DEBUG] Serper API: rate limit hit.")
        raise RateLimitException("Serper API rate limit hit.")
    else:
        logging.error(f"Serper API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] Serper API error: {resp.status_code}")
    return None

def fetch_from_brave(query):
    #print("[DEBUG] Hitting Brave API with query:", query)
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        logging.error("BRAVE_API_KEY not set.")
        #print("[DEBUG] Brave API: API key not set.")
        return None
    url = f"https://api.search.brave.com/res/v1/news/search?q={query}&count=5"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
    except Exception as e:
        logging.error(f"Brave API request failed: {e}")
        #print("[DEBUG] Brave API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] Brave API returned results.")
        return [
            {
                "title": n["title"],
                "url": n["url"],
                "source": n.get("publisher", {}).get("name", ""),
                "publishedAt": n.get("publishedAt", ""),
                "content": n.get("description", "")
            }
            for n in data.get("results", [])
        ]
    elif resp.status_code == 429:
        #print("[DEBUG] Brave API: rate limit hit.")
        raise RateLimitException("Brave API rate limit hit.")
    else:
        logging.error(f"Brave API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] Brave API error: {resp.status_code}")
    return None

def fetch_from_wikipedia(query):
    #print("[DEBUG] Hitting Wikipedia with query:", query)
    try:
        results = wikipedia.search(query)
        if not results:
            #print("[DEBUG] Wikipedia: No results found.")
            return None
        page = wikipedia.page(results[0])
        summary = wikipedia.summary(results[0], sentences=2)
        #print(f"[DEBUG] Wikipedia page found: {page.title}")
        return [{
            "title": page.title,
            "url": page.url,
            "source": "Wikipedia",
            "publishedAt": "",
            "content": summary
        }]
    except Exception as e:
        logging.error(f"Wikipedia API error: {e}")
        #print(f"[DEBUG] Wikipedia API error: {e}")
        return None

def fetch_news_articles(query: str):
    articles = []
    api_funcs = [fetch_from_newsapi, fetch_from_guardian, fetch_from_serper, fetch_from_brave]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(try_api_call, api_func, query) for api_func in api_funcs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                articles.extend(result)
    # Always try Wikipedia as a final fallback if no articles found
    if not articles:
        wiki_result = try_api_call(fetch_from_wikipedia, query)
        if wiki_result:
            articles.extend(wiki_result)
    if not articles:
        logging.error(f"All news APIs failed for query: {query}")
    return articles

# --- Government APIs ---
def fetch_from_congress_gov(query):
    #print("[DEBUG] Hitting Congress.gov API with query:", query)
    api_key = os.getenv("CONGRESS_API_KEY")
    if not api_key:
        logging.error("CONGRESS_API_KEY not set.")
        #print("[DEBUG] Congress.gov API: API key not set.")
        return None
    url = f"https://api.congress.gov/v3/bill?query={query}&api_key={api_key}"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        logging.error(f"Congress.gov API request failed: {e}")
        #print("[DEBUG] Congress.gov API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] Congress.gov API returned results.")
        return data.get("bills", [])
    elif resp.status_code == 429:
        #print("[DEBUG] Congress.gov API: rate limit hit.")
        raise RateLimitException("Congress.gov API rate limit hit.")
    else:
        logging.error(f"Congress.gov API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] Congress.gov API error: {resp.status_code}")
    return None

def fetch_from_govinfo(query):
    #print("[DEBUG] Hitting GovInfo API with query:", query)
    api_key = os.getenv("GOVINFO_API_KEY")
    if not api_key:
        logging.error("GOVINFO_API_KEY not set.")
        #print("[DEBUG] GovInfo API: API key not set.")
        return None
    url = f"https://api.govinfo.gov/collections/BILLS/{query}?api_key={api_key}"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        logging.error(f"GovInfo API request failed: {e}")
        #print("[DEBUG] GovInfo API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] GovInfo API returned results.")
        return data.get("packages", [])
    elif resp.status_code == 429:
        #print("[DEBUG] GovInfo API: rate limit hit.")
        raise RateLimitException("GovInfo API rate limit hit.")
    else:
        logging.error(f"GovInfo API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] GovInfo API error: {resp.status_code}")
    return None

def fetch_from_fec(query):
    #print("[DEBUG] Hitting FEC API with query:", query)
    api_key = os.getenv("FEC_API_KEY")
    if not api_key:
        logging.error("FEC_API_KEY not set.")
        #print("[DEBUG] FEC API: API key not set.")
        return None
    url = f"https://api.open.fec.gov/v1/search/?api_key={api_key}&query={query}"
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:
        logging.error(f"FEC API request failed: {e}")
        #print("[DEBUG] FEC API: request failed.")
        return None
    if resp.status_code == 200:
        data = resp.json()
        #print("[DEBUG] FEC API returned results.")
        return data.get("results", [])
    elif resp.status_code == 429:
        #print("[DEBUG] FEC API: rate limit hit.")
        raise RateLimitException("FEC API rate limit hit.")
    else:
        logging.error(f"FEC API error: {resp.status_code} {resp.text}")
        #print(f"[DEBUG] FEC API error: {resp.status_code}")
    return None

def fetch_legislation_data(query: str):
    for api_func in [fetch_from_congress_gov, fetch_from_govinfo, fetch_from_fec]:
        result = try_api_call(api_func, query)
        if result:
            return result
    logging.error(f"All government APIs failed for query: {query}")
    return []

# --- Classification ---
def is_political_question(question: str) -> bool:
    # Single-word and multi-word keywords
    political_keywords = [
        'election', 'elections', 'congress', 'senate', 'house', 'president', 'presidential',
        'republican', 'democrat', 'policy', 'government', 'law', 'bill', 'bills', 'politics',
        'campaign', 'campaigns', 'vote', 'voting', 'primary', 'primaries', 'issues', 'issue',
        'supreme court', 'governor', 'mayor', 'legislation', 'political', 'partisan',
        'white house', 'administration', 'federal', 'state', 'local government',
        'debt ceiling', 'negotiation', 'shutdown', 'budget', 'appropriations', 'default',
        'fiscal', 'stimulus', 'bipartisan', 'bipartisanship'
    ]
    question_lower = question.lower()
    for keyword in political_keywords:
        if ' ' in keyword:
            if keyword in question_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', question_lower):
                return True
    if re.search(r'\b(act|bill|resolution|amendment)\b', question_lower):
        return True
    return False

def is_partisan_topic(question: str) -> bool:
    """
    Returns True if the question is about a partisan issue or debate, based on keywords.
    """
    partisan_keywords = [
        'immigration', 'abortion', 'gun control', 'tax', 'healthcare', 'climate', 'minimum wage',
        'lgbt', 'transgender', 'border', 'race', 'affirmative action', 'voting rights',
        'police', 'crime', 'education', 'student loan', 'welfare', 'social security',
        'medicare', 'medicaid', 'environment', 'energy', 'foreign policy', 'israel',
        'ukraine', 'china', 'trade', 'tariff', 'military', 'defense', 'war', 'peace',
        'republican', 'democrat', 'bipartisan', 'partisan', 'conservative', 'liberal',
        'progressive', 'right-wing', 'left-wing', 'supreme court', 'court decision',
        'election', 'campaign', 'ballot', 'primary', 'senate race', 'house race',
        'presidential race', 'presidential debate', 'political debate', 'policy debate',
        'controversy', 'scandal', 'investigation', 'indictment', 'impeachment',
    ]
    q = question.lower()
    return any(k in q for k in partisan_keywords)

def is_officeholder_question(question: str) -> bool:
    q = question.lower().strip()
    q = re.sub(r'[?.!]+$', '', q)  # Remove trailing punctuation
    patterns = [
        r"who is (the )?(president|vice president|speaker of the house|senate majority leader|governor|mayor|prime minister|chancellor) of( the)?( .*)?",
        r"current (president|vice president|speaker|leader|governor|mayor|prime minister|chancellor) of( the)?( .*)?"
    ]
    return any(re.match(p, q) for p in patterns)

def filter_legislation_by_year(legislations, year):
    if not year:
        return legislations
    filtered = [
        leg for leg in legislations
        if year in (leg.get("date","") + leg.get("introducedOn",""))
    ]
    return filtered or legislations

# --- Response Synthesis ---
def synthesize_response(topic: str, articles: List[Dict[str, Any]], legislation: List[Dict[str, Any]]) -> str:
    
    # Define the model
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
    )

    model = ChatHuggingFace(llm=llm)

    context = ""
    if articles:
        context += "Relevant news articles:\n"
        for art in articles[:3]:
            context += f"- {art.get('title', '')} ({art.get('source', '')}, {art.get('publishedAt', '')}): {art.get('url', '')}\n"
            context += f"  Summary: {art.get('content', '')}\n"
    if legislation:
        context += "Relevant legislative data:\n"
        for leg in legislation[:3]:
            context += f"- {leg.get('title', '')} ({leg.get('date', '')}): {leg.get('url', '')}\n"
    if context:
        context += "\nUse the above sources for your answer.\n"

    

    try:
        prompt_template1 = PromptTemplate(
            template="Given the context:\n{context}\n\nWrite a detailed report on: {topic}",
            input_variables=["context", "topic"]
        )

        prompt_template2 = PromptTemplate(
            template="write 5 line summary on the following text  /n {text}",
            input_variables=['text']
        )

        parser = StrOutputParser()
        def add_context_to_prompt1(input):
            return {
                "context": "You are a helpful assistant trained to provide in-depth analysis and summaries.",
                "topic": input["topic"]
            }

        def add_context_to_prompt2(intermediate_output):
            return {
                "context": "You are a helpful assistant trained to provide in-depth analysis and summaries.",
                "text": intermediate_output
            }
        chain = (
            RunnableLambda(add_context_to_prompt1)
            | prompt_template1
            | model
            | parser
            | RunnableLambda(add_context_to_prompt2)
            | prompt_template2
            | model
            | parser
        )

        result = chain.invoke({"topic": topic, "context": context})
        # print(result)

        content = result
        if content:
            return content.strip()
        else:
            return "[OpenAI API returned no content in the response.]"
    except Exception as e:
        return f"[OpenAI API error: {e}]"

# --- Post-processing for Neutrality, Citations, and Perspective ---
def detect_partisan_language(response: str) -> bool:
    # Focus on truly partisan or inflammatory language only
    partisan_keywords = [
        r"\b(radical left|far[- ]?left|far[- ]?right|extremist|MAGA|woke|Trumpist|Bidenomics|partisan attack|partisan agenda|leftist|rightist|ultra-conservative|ultra-liberal|socialist agenda|fascist|communist|RINO|snowflake|libtard|fake news|enemy of the people|traitor|un-American|patriot act|witch hunt|deep state|cancel culture|culture war|gaslighting|dog whistle|race-baiting|fearmongering|hate speech|incitement|authoritarian|totalitarian|dictatorship|coup|insurrection|sedition|treason)\b",
        r"\b(slam|attack|vilify|condemn|denounce|smear campaign|scapegoat|demonize|weaponize|fearmonger|race-bait|gaslight|dog-whistle)\b"
    ]
    for pattern in partisan_keywords:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False

def check_citations(response: str) -> bool:
    # Require at least one URL per paragraph or claim
    url_pattern = r"https?://[\w\.-/]+"
    paragraphs = [p for p in response.split('\n') if p.strip()]
    for p in paragraphs:
        if any(word in p.lower() for word in ["according to", "reports", "states", "says", "claims", "announced", "ruled", "decided", "voted", "passed", "signed"]):
            if not re.search(url_pattern, p):
                return False
    return True

def check_perspective_balance(response: str) -> bool:
    # Look for both "Republican" and "Democrat" (or synonyms) in the response
    rep_keywords = ["Republican", "GOP", "conservative", "right-wing"]
    dem_keywords = ["Democrat", "liberal", "left-wing", "progressive"]
    rep_found = any(k.lower() in response.lower() for k in rep_keywords)
    dem_found = any(k.lower() in response.lower() for k in dem_keywords)
    return rep_found and dem_found

def flag_unverified_claims(response: str) -> list:
    # Return sentences that make claims but lack a URL
    url_pattern = r"https?://[\w\.-/]+"
    sentences = re.split(r'[.!?]\s+', response)
    flagged = []
    for s in sentences:
        if any(word in s.lower() for word in ["according to", "reports", "states", "says", "claims", "announced", "ruled", "decided", "voted", "passed", "signed"]):
            if not re.search(url_pattern, s):
                flagged.append(s.strip())
    return flagged

def inline_flag_unverified_claims(response: str) -> str:
    # Highlight sentences that make claims but lack a URL
    url_pattern = r"https?://[\w\.-/]+"
    sentences = re.split(r'([.!?]\s+)', response)  # Keep punctuation as separate tokens
    flagged = []
    rebuilt = ""
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        punct = sentences[i+1] if i+1 < len(sentences) else ""
        if any(word in s.lower() for word in ["according to", "reports", "states", "says", "claims", "announced", "ruled", "decided", "voted", "passed", "signed"]):
            if not re.search(url_pattern, s):
                rebuilt += f"[UNCITED CLAIM: {s.strip()}]{punct}"
                continue
        rebuilt += s + punct
    return rebuilt

# --- Secondary LLM Review ---
def secondary_llm_review(response: str, api_key: str) -> str:
    """
    Use the LLM to review the response for neutrality, perspective balance, and citation compliance.
    Returns the review result or suggested corrections.
    """
    review_prompt = (
        "Review the following answer for strict political neutrality, balanced Republican and Democratic perspectives, and proper citations for every factual claim. "
        "If the answer is neutral, balanced, and fully cited, reply ONLY with 'PASS'. "
        "If not, explain the issues and suggest corrections.\n\n"
        f"Answer to review:\n{response}"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        review = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a political AI quality assurance reviewer."},
                {"role": "user", "content": review_prompt}
            ],
            temperature=0.0,
            max_tokens=400,
        )
        review_content = review.choices[0].message.content
        if review_content is not None:
            return review_content.strip()
        else:
            return "[Review LLM error: No content returned.]"
    except Exception as e:
        return f"[Review LLM error: {e}]"

# Stub for confidence scoring and fact cross-referencing
def score_fact_confidence(articles: list, response: str) -> dict:
    """
    Placeholder for future implementation: Cross-reference facts in the response with articles,
    and assign a confidence score based on agreement and source credibility.
    """
    # To be implemented: parse facts, check agreement, score credibility
    return {}

# --- Main Chatbot Logic ---
def filter_us_articles(articles, user_input):
    # If user explicitly mentions another country, don't filter
    non_us_countries = [
        "canada", "uk", "britain", "england", "australia", "germany", "france", "china", "india", "mexico", "russia", "japan", "brazil", "italy", "spain", "europe", "european union", "eu", "africa", "asia", "middle east", "iran", "iraq", "syria", "turkey", "israel", "palestine", "ukraine", "poland", "sweden", "norway", "finland", "denmark", "netherlands", "switzerland", "austria", "belgium", "ireland", "scotland", "wales", "new zealand", "south africa", "egypt", "saudi arabia", "uae", "argentina", "chile", "colombia", "venezuela", "peru", "cuba", "haiti", "dominican republic", "jamaica", "pakistan", "afghanistan", "north korea", "south korea", "taiwan", "hong kong", "singapore", "malaysia", "indonesia", "philippines", "thailand", "vietnam", "cambodia", "laos", "myanmar", "mongolia", "kazakhstan", "uzbekistan", "turkmenistan", "kyrgyzstan", "tajikistan", "georgia", "armenia", "azerbaijan", "greece", "portugal", "croatia", "serbia", "bosnia", "montenegro", "albania", "macedonia", "slovenia", "slovakia", "czech", "hungary", "romania", "bulgaria", "estonia", "latvia", "lithuania", "belarus", "moldova", "luxembourg", "liechtenstein", "monaco", "andorra", "san marino", "vatican", "malta", "cyprus", "iceland", "greenland", "antarctica", "fiji", "tonga", "samoa", "papua", "guinea", "new caledonia", "solomon islands", "vanuatu", "micronesia", "palau", "marshall islands", "kiribati", "nauru", "tuvalu", "seychelles", "mauritius", "madagascar", "comoros", "cape verde", "sao tome", "principe", "gabon", "congo", "zambia", "zimbabwe", "botswana", "namibia", "angola", "mozambique", "malawi", "tanzania", "kenya", "uganda", "rwanda", "burundi", "sudan", "south sudan", "ethiopia", "somalia", "djibouti", "eritrea", "morocco", "algeria", "tunisia", "libya", "nigeria", "ghana", "ivory coast", "senegal", "mali", "burkina", "niger", "benin", "togo", "sierra leone", "liberia", "guinea", "gambia", "cameroon", "central african republic", "chad", "equatorial guinea", "guinea-bissau", "lesotho", "swaziland", "eswatini", "mauritania", "western sahara", "sudan", "yemen", "oman", "qatar", "bahrain", "kuwait", "jordan", "lebanon", "syria", "iraq", "iran", "afghanistan", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives", "mongolia", "north korea", "south korea", "taiwan", "hong kong", "macau", "china", "japan", "philippines", "vietnam", "thailand", "myanmar", "cambodia", "laos", "malaysia", "singapore", "indonesia", "brunei", "timor-leste", "australia", "new zealand", "fiji", "papua new guinea", "solomon islands", "vanuatu", "samoa", "tonga", "tuvalu", "kiribati", "nauru", "palau", "micronesia", "marshall islands", "palestine"
    ]
    if any(country in user_input.lower() for country in non_us_countries):
        return articles
    us_keywords = [
        "united states", "u.s.", "us ", "america", "american", "congress", "senate", "house", "president", "biden", "trump", "white house", "federal", "washington", "democrat", "republican", "gop"
    ]
    filtered = []
    for art in articles:
        text = (art.get('title', '') + ' ' + art.get('summary', '') + ' ' + art.get('content', '')).lower()
        if any(kw in text for kw in us_keywords):
            filtered.append(art)
    # Soft filter: if nothing matches, fall back to original
    return filtered if filtered else articles

def extract_target_year(user_input):
    import re
    # Look for a 4-digit year in the user input
    match = re.search(r"\b(20\d{2})\b", user_input)
    if match:
        return match.group(1)
    return None

def remove_year_from_query(query):
    import re
    return re.sub(r"\b20\d{2}\b", "", query).strip()

def filter_articles_by_year(articles, year):
    if not year:
        #print("[DEBUG] No year found in query; skipping year filtering.")
        return articles, False
    filtered = []
    fallback_phrases = ["last year", "recent"]
    prev_year = str(int(year) - 1)
    next_year = str(int(year) + 1)
    #print(f"[DEBUG] Filtering articles for year: {year}")
    #print("[DEBUG] All scraped articles before year filtering:")
    for art in articles:
        #print(f"[DEBUG] Article: {art.get('title')} | Summary: {art.get('summary', '')}")
        text = (art.get('title', '') + ' ' + art.get('summary', '') + ' ' + art.get('content', '')).lower()
        if year in text:
            #print(f"[DEBUG] Including article for exact year {year}: {art.get('title')}")
            filtered.append(art)
    if filtered:
        #print(f"[DEBUG] Found {len(filtered)} articles with exact year {year}.")
        return filtered, True
    # Fallback: try previous/next year and phrases
    for art in articles:
        text = (art.get('title', '') + ' ' + art.get('summary', '') + ' ' + art.get('content', '')).lower()
        if prev_year in text or next_year in text or any(phrase in text for phrase in fallback_phrases):
            #print(f"[DEBUG] Including article for fallback year/phrase: {art.get('title')}")
            filtered.append(art)
    if filtered:
        #print(f"[DEBUG] Found {len(filtered)} articles with fallback year/phrase.")
        return filtered, True
    #print("[DEBUG] No articles matched year or fallback; returning all scraped articles as last resort.")
    return articles, False

def chatbot_response(user_input: str) -> str:
    # 1) Officeholder shortcut
    if is_officeholder_question(user_input):
        wiki = fetch_from_wikipedia(user_input)
        if wiki:
            return f"{wiki[0]['content']} (Source: {wiki[0]['url']})"
        return "[No reputable sources found for that officeholder question.]"

    # 2) Scope check
    if not is_political_question(user_input):
        return "I'm sorry, I can only answer questions about political events."

    # 3) Fetch from all APIs
    articles    = fetch_news_articles(user_input)
    legislation = fetch_legislation_data(user_input)

    # 4) Year‐filter both streams
    year = extract_target_year(user_input)            # e.g. "2023"
    if year:
        articles, _    = filter_articles_by_year(articles, year)
        legislation    = filter_legislation_by_year(legislation, year)

    # 5) If zero API articles → scrape news sites
    if not articles:
        scraped = []
        for fn in (
            scrape_reuters_headlines,
            scrape_apnews_headlines,
            scrape_bbc_headlines,
            scrape_nytimes_headlines,
            scrape_aljazeera_headlines
        ):
            try:
                scraped.extend(fn(user_input))
            except Exception:
                pass

        # apply year filter to those scraped headlines
        articles, _ = filter_articles_by_year(scraped, year)

    # ←―――――――――――――――――――――――――――――――――――――――
    # INSERT the Wikipedia fallback for “key issue” questions right here:
    if "key issue" in user_input.lower():
        wiki_ctx = try_api_call(
            fetch_from_wikipedia,
            "2024 United States presidential primary key issues"
        )
        if wiki_ctx:
            # tack on the 2-sentence summary so the LLM has something to cite
            articles.extend(wiki_ctx)
    # ――――――――――――――――――――――――――――――――――――――――→

    # 6) Final bail-out if BOTH streams are still empty
    if not articles and not legislation:
        return "[No reputable sources or citations were found for your query.]"

    # 7) Generate the answer via LLM
    answer = synthesize_response(user_input, articles, legislation)

    # 8) (Optional) Your existing post‐processing: checks for citations,
    #    partisan language, regeneration, LLM review, etc.
    #    ─────────────────────────────────────────────────────────────

    return answer

# --- CLI for Testing (to be replaced by Gradio UI) ---
def main(user_input):
    #print("Welcome to the Political AI Chatbot! Ask me any question about political events.")
    while True:
        if user_input.lower() in ["exit", "quit", "bye"]:
            #print("Goodbye!")
            break
        response = chatbot_response(user_input)
        return response

def scrape_senate_recent_votes() -> List[Dict[str, str]]:
    try:
        url = "https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_118_1.htm"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        votes = []
        for row in soup.select('table.voteList tr')[1:]:  # skip header
            cells = row.find_all('td') if isinstance(row, Tag) else []
            if len(cells) >= 5:
                vote_number = cells[0].get_text(strip=True)
                date = cells[1].get_text(strip=True)
                result = cells[2].get_text(strip=True)
                title = cells[3].get_text(strip=True)
                link_tag = None
                cell3 = cells[3]
                if isinstance(cell3, Tag):
                    link_tag = cell3.find('a')
                url = "https://www.senate.gov" + str(link_tag['href']) if isinstance(link_tag, Tag) and link_tag.has_attr('href') else ""
                votes.append({
                    'vote_number': vote_number,
                    'date': date,
                    'result': result,
                    'title': title,
                    'url': url
                })
        return votes
    except Exception as e:
        #print(f"[DEBUG] Senate.gov scraping error: {e}")
        return []

def scrape_house_recent_votes() -> List[Dict[str, str]]:
    try:
        url = "https://clerk.house.gov/Votes"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        votes = []
        for row in soup.select('table.votestable tbody tr'):
            cells = row.find_all('td') if isinstance(row, Tag) else []
            if len(cells) >= 5:
                vote_number = cells[0].get_text(strip=True)
                date = cells[1].get_text(strip=True)
                result = cells[2].get_text(strip=True)
                title = cells[3].get_text(strip=True)
                link_tag = None
                cell0 = cells[0]
                if isinstance(cell0, Tag):
                    link_tag = cell0.find('a')
                url = "https://clerk.house.gov" + str(link_tag['href']) if isinstance(link_tag, Tag) and link_tag.has_attr('href') else ""
                votes.append({
                    'vote_number': vote_number,
                    'date': date,
                    'result': result,
                    'title': title,
                    'url': url
                })
        return votes
    except Exception as e:
        #print(f"[DEBUG] House.gov scraping error: {e}")
        return []

def scrape_reuters_headlines(query: str) -> List[Dict[str, str]]:
    try:
        url = f"https://www.reuters.com/site-search/?query={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for item in soup.find_all('div', class_='search-result-content'):
            title = item.find('h3') if isinstance(item, Tag) else None
            link = item.find('a') if isinstance(item, Tag) else None
            summary = item.find('p') if isinstance(item, Tag) else None
            if isinstance(title, Tag) and isinstance(link, Tag) and link.has_attr('href'):
                articles.append({
                    'title': title.get_text(strip=True),
                    'url': "https://www.reuters.com" + str(link['href']),
                    'summary': summary.get_text(strip=True) if isinstance(summary, Tag) else ""
                })
        return articles
    except Exception as e:
        #print(f"[DEBUG] Reuters scraping error: {e}")
        return []

def scrape_apnews_headlines(query: str) -> List[Dict[str, str]]:
    try:
        url = f"https://apnews.com/search?q={query.replace(' ', '%20')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for item in soup.find_all('div', class_='SearchResults-item'):
            title_tag = item.find('a', class_='Component-headline') if isinstance(item, Tag) else None
            summary_tag = item.find('div', class_='Component-content') if isinstance(item, Tag) else None
            if isinstance(title_tag, Tag) and title_tag.has_attr('href'):
                href = str(title_tag['href'])
                articles.append({
                    'title': title_tag.get_text(strip=True),
                    'url': "https://apnews.com" + href,
                    'summary': summary_tag.get_text(strip=True) if isinstance(summary_tag, Tag) else ""
                })
        return articles
    except Exception as e:
        #print(f"[DEBUG] AP News scraping error: {e}")
        return []

def scrape_bbc_headlines(query: str) -> List[Dict[str, str]]:
    try:
        url = f"https://www.bbc.co.uk/search?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for item in soup.find_all('article', class_='css-8tq3w8-Stack e1y4nx260'):
            title_tag = item.find('a', class_='css-1aofmbn-PromoLink e1f5wbog0') if isinstance(item, Tag) else None
            summary_tag = item.find('p') if isinstance(item, Tag) else None
            if isinstance(title_tag, Tag) and title_tag.has_attr('href'):
                href = str(title_tag['href'])
                url_full = href if href.startswith('http') else f"https://www.bbc.co.uk{href}"
                articles.append({
                    'title': title_tag.get_text(strip=True),
                    'url': url_full,
                    'summary': summary_tag.get_text(strip=True) if isinstance(summary_tag, Tag) else ""
                })
        return articles
    except Exception as e:
        #print(f"[DEBUG] BBC scraping error: {e}")
        return []

def scrape_nytimes_headlines(query: str) -> List[Dict[str, str]]:
    try:
        url = f"https://www.nytimes.com/search?query={query.replace(' ', '%20')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for item in soup.find_all('li', attrs={'data-testid': 'search-bodega-result'}) or []:
            title_tag = item.find('h4') if isinstance(item, Tag) else None
            link_tag = item.find('a') if isinstance(item, Tag) else None
            summary_tag = item.find('p') if isinstance(item, Tag) else None
            if isinstance(title_tag, Tag) and isinstance(link_tag, Tag) and link_tag.has_attr('href'):
                href = str(link_tag['href'])
                url_full = href if href.startswith('http') else f"https://www.nytimes.com{href}"
                articles.append({
                    'title': title_tag.get_text(strip=True),
                    'url': url_full,
                    'summary': summary_tag.get_text(strip=True) if isinstance(summary_tag, Tag) else ""
                })
        return articles
    except Exception as e:
        #print(f"[DEBUG] NYTimes scraping error: {e}")
        return []

def scrape_aljazeera_headlines(query: str) -> List[Dict[str, str]]:
    try:
        url = f"https://www.aljazeera.com/Search/?q={query.replace(' ', '%20')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for item in soup.find_all('div', class_='gc__content'):
            title_tag = item.find('a', class_='u-clickable-card__link') if isinstance(item, Tag) else None
            summary_tag = item.find('div', class_='gc__excerpt') if isinstance(item, Tag) else None
            if isinstance(title_tag, Tag) and title_tag.has_attr('href'):
                href = str(title_tag['href'])
                url_full = href if href.startswith('http') else f"https://www.aljazeera.com{href}"
                articles.append({
                    'title': title_tag.get_text(strip=True),
                    'url': url_full,
                    'summary': summary_tag.get_text(strip=True) if isinstance(summary_tag, Tag) else ""
                })
        return articles
    except Exception as e:
        #print(f"[DEBUG] Al Jazeera scraping error: {e}")
        return []

if __name__ == "__main__":
    
    # UI elements
    st.header("Chatbot tool")
    user_input = st.text_input("Enter Your Prompt")

    if st.button("Response"):
        start_time = time.time()
        
        with st.spinner("Processing..."):
            response = main(user_input)
        
        end_time = time.time()
        elapsed = int(end_time - start_time)

        # Format time
        if elapsed < 60:
            st.success(f"Done in {elapsed} seconds.")
        else:
            minutes = elapsed // 60
            seconds = elapsed % 60
            st.success(f"Done in {minutes} minute(s) and {seconds} second(s).")
        
        # Show the response
        st.write(response)