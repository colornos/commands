import os
import sys
import aiohttp
import asyncio
from readability import Document
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
import subprocess
import json

def get_news_urls(query: str, max_results: int = 10) -> list:
    """
    Fetches news URLs based on the search query using DuckDuckGoSearchResults.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to fetch.

    Returns:
        list: A list containing up to `max_results` news URLs.
    """
    tool = DuckDuckGoSearchResults(output_format='json', max_results=max_results)
    results = tool.invoke(query)
    
    # Parse the JSON string into a list of dictionaries
    try:
        results_list = json.loads(results)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    
    # Collect URLs from results
    urls = [result['link'] for result in results_list if 'link' in result]
    return urls

async def fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetches the HTML content of a given URL.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for requests.
        url (str): The URL to fetch.

    Returns:
        str: The HTML content of the page.
    """
    print(f"Fetching {url}")
    async with session.get(url) as response:
        response.raise_for_status()
        html = await response.text()
        return html

def html_to_text(html: str) -> str:
    """
    Converts HTML content to plain text using Readability and BeautifulSoup.

    Args:
        html (str): The HTML content.

    Returns:
        str: The extracted plain text.
    """
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text

async def get_cleaned_texts(session: aiohttp.ClientSession, urls: list) -> list:
    """
    Fetches and cleans text from a list of URLs.

    Args:
        session (aiohttp.ClientSession): The HTTP session to use for requests.
        urls (list): A list of URLs to fetch and clean.

    Returns:
        list: A list of cleaned text strings with sources.
    """
    texts = []
    for url in urls:
        try:
            html = await fetch_html(session, url)
            text = html_to_text(html)
            formatted_text = f"Source: {url}\n{text}\n\n"
            texts.append(formatted_text)
        except Exception as e:
            print(f"Error fetching or processing {url}: {e}")
    return texts

async def answer_query(query: str, texts: list):
    """
    Sends the aggregated texts to the Ollama CLI to generate an answer.

    Args:
        query (str): The user's query.
        texts (list): A list of text strings to use as context.

    Returns:
        None
    """
    prompt = f"{query}. Summarize the information and provide an answer. Use only the information in the following articles to answer the question: {''.join(texts)}"
    
    # Construct the Ollama command
    cmd = ['ollama', 'run', 'llama3.2-vision:11b']
    
    try:
        # Run the Ollama command with the prompt
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate(input=prompt.encode())

        if process.returncode != 0:
            print(f"Ollama error: {stderr.decode().strip()}")
            return

        # Print the response from Ollama
        response = stdout.decode().strip()
        print(response)

    except FileNotFoundError:
        print("Error: 'ollama' command not found. Ensure Ollama is installed and added to your PATH.")
    except Exception as e:
        print(f"An error occurred while running Ollama: {e}")

async def main():
    """
    The main asynchronous function orchestrating the workflow.
    """
    if len(sys.argv) < 2:
        print("Error: No query provided.")
        sys.exit(1)
    
    query = ' '.join(sys.argv[1:])
    print(f"Query: {query}")
    
    urls = get_news_urls(query, max_results=1)  # Adjust max_results as needed
    if not urls:
        print("No URLs found for the given query.")
        sys.exit(1)
    
    async with aiohttp.ClientSession() as session:
        all_texts = await get_cleaned_texts(session, urls)
        if not all_texts:
            print("No texts could be extracted from the fetched URLs.")
            sys.exit(1)
        
        await answer_query(query, all_texts)

if __name__ == "__main__":
    asyncio.run(main())
