import requests
from pathlib import Path

class Gutenberg:

    BOOK_TO_LINK = {
        'crime_and_punishment': 'https://www.gutenberg.org/cache/epub/2554/pg2554.txt',
        'shakespeare': 'https://www.gutenberg.org/cache/epub/100/pg100.txt',
        'pride_and_prejudice': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
        'kant_critique': 'https://www.gutenberg.org/cache/epub/4280/pg4280.txt',
        'zarathustra': 'https://www.gutenberg.org/cache/epub/1998/pg1998.txt',
    }

    CACHE_DIR = Path("./cache")

    @staticmethod
    def fetch_dataset(name: str, force_refresh: bool = False) -> str:
        
        if name not in Gutenberg.BOOK_TO_LINK:
            raise ValueError(f"Unknown dataset: {name}")

        url = Gutenberg.BOOK_TO_LINK[name]
        Gutenberg.CACHE_DIR.mkdir(exist_ok=True)

        # Local cache file path
        cache_path = Gutenberg.CACHE_DIR / f"{name}.txt"

        if cache_path.exists() and not force_refresh:
            return cache_path.read_text(encoding="utf-8")

        # Download if not cached
        response = requests.get(url)
        response.raise_for_status()
        text = response.text

        # Save to cache
        cache_path.write_text(text, encoding="utf-8")
        return text