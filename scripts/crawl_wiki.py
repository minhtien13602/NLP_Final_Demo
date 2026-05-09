# =========================================================
# FILE: scripts/crawl_wiki.py
# =========================================================

import os
import requests

from bs4 import BeautifulSoup


SAVE_DIR = "data/raw"

os.makedirs(SAVE_DIR, exist_ok=True)


URLS = {
    "nha_nguyen.txt":
    "https://vi.wikipedia.org/wiki/Nh%C3%A0_Nguy%E1%BB%85n",

    "nha_ly.txt":
    "https://vi.wikipedia.org/wiki/Nh%C3%A0_L%C3%BD",
}


headers = {
    "User-Agent": "Mozilla/5.0"
}


def crawl_page(url):

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    content = soup.find(
        "div",
        {"id": "mw-content-text"}
    )

    paragraphs = content.find_all("p")

    full_text = ""

    for p in paragraphs:

        text = p.get_text(" ", strip=True)

        if len(text) > 50:
            full_text += text + "\n"

    return full_text


for filename, url in URLS.items():

    print("Crawling:", url)

    text = crawl_page(url)

    save_path = os.path.join(
        SAVE_DIR,
        filename
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    print("Saved:", save_path)

print("DONE")