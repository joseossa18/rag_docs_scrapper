from pandas_gbq import to_gbq
from typing import List, Dict, Any
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
TABLE = os.getenv("TABLE")


def get_ai_docs_urls() -> List[str]:
    """Get URLs from local file"""
    # Open the file and read lines
    with open("rag_urls.txt", "r") as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

def send_to_bigquery(df: pd.DataFrame):
    """Send dataframe to a Bigquery table"""
    # Define your BigQuery destination (e.g., 'project_id.dataset.table')
    destination = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    # Use pandas_gbq to send the dataframe to BigQuery
    to_gbq(df, destination, project_id=PROJECT_ID, if_exists='replace')


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks 