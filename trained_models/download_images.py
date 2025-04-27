import os
import random
import asyncio
import aiohttp
import aiofiles
from serpapi import GoogleSearch

# Your SerpAPI key
API_KEY = "2d7d083f365a1756060d5117c6774713afe47c52ca844494a86868af3c77a5f7"

# Base directory to save images
SAVE_DIR = "trained_models/images"

# Ensure the base directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def fetch_image_urls(query, num_images=10):
    """Fetch image URLs from SerpAPI."""
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": API_KEY,
        "num": num_images
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    images_results = results.get("images_results", [])
    return [img["original"] for img in images_results[:num_images]]

async def download_image(session, url, category):
    """Download a single image asynchronously using aiohttp and save it asynchronously."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/103.0.5060.114 Safari/537.36"
        )
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                file_name = f"image_{random.randint(1, 10**24)}.jpg"
                file_path = os.path.join(SAVE_DIR, category, file_name)
                content = await response.read()
                async with aiofiles.open(file_path, "wb") as file:
                    await file.write(content)
                print(f"Downloaded {file_path}")
            else:
                print(f"Failed to download {url} (Status code: {response.status})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

async def download_images(image_urls, category):
    """Download multiple images concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, category) for url in image_urls]
        await asyncio.gather(*tasks)

async def main():
    # Define categories with search queries.
    categories = {
        'guns_knives': ['a man with a gun', 'person shooting with a sniper', 'a man holding a knife'],
        'fire_smoke': ['fire burning', 'smoke explosion'],
        'fights': ['people fighting', 'street fight'],
        'realtime_accidents': ['car accident', 'traffic accident'],
        'climber': ['rock climber on a steep cliff', 'mountain climber at summit', 'climber scaling a rocky wall', 'bouldering challenge']
    }

    for category, queries in categories.items():
        # Ensure the category-specific directory exists.
        category_dir = os.path.join(SAVE_DIR, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            print(f"Created directory: {category_dir}")
        
        # For each query in the category, fetch and download images concurrently.
        if queries:
            for query in queries:
                print(f"\nFetching images for query '{query}' under category '{category}'...")
                image_urls = fetch_image_urls(query, num_images=100)
                await download_images(image_urls, category)
        else:
            print(f"\nNo specific queries for '{category}'. Using category name as query.")
            image_urls = fetch_image_urls(category, num_images=100)
            await download_images(image_urls, category)

if __name__ == "__main__":
    asyncio.run(main())
