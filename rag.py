import os
import sys
import json
import asyncio
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI

from utils.utils import get_ai_docs_urls, send_to_bigquery, chunk_text

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

#Create dataframe to store docs
df_rag_docs = pd.DataFrame(columns=["url", "chunk_number", "content", "metadata", "title", "summary","embedding"]) 

# Create an asyncio lock to ensure thread-safe access to df
lock = asyncio.Lock()

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI designed to extract titles and summaries from sections of documentation. 
                Return a JSON object containing two keys: 'title' and 'summary'. 
                For the title: If this is the beginning of the document, extract the title. If it’s a middle section, generate a descriptive title based on the content. 
                For the summary: Provide a brief and clear summary of the key points in this section. Ensure both the title and summary are short but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model= "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}
    
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> Dict:
    """Process a single chunk of text."""
    # Get title and summary
    extracted_data = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "gorgias_webpage",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    processed_chunk = {
        "url": url,
        "chunk_number": chunk_number,
        "content": chunk,
        "metadata": metadata,
        "title": extracted_data['title'],
        "summary" : extracted_data['summary'],
        "embedding": embedding
    }
    return processed_chunk

async def insert_chunk(chunk: Dict):
    """Insert a processed chunk into Biquery"""
    try:
            # Acquire the lock before appending to the dataframe so the aren't any concurrency problems
        async with lock:
            global df_rag_docs
            df_rag_docs = pd.concat([df_rag_docs, pd.DataFrame([chunk])], ignore_index=True)
            print(f"Inserted chunk {chunk["chunk_number"]} for {chunk["url"]}")
        return True
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


async def main():
    # Get URLs from Gorgias
    urls = get_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

    send_to_bigquery(df_rag_docs)


if __name__ == "__main__":
    asyncio.run(main())



