import requests
from bs4 import BeautifulSoup

url = "https://www.imsdb.com/all-scripts/"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if "/Movie Scripts/" in href:
        links.append(href)

print(f"Found {len(links)} script links.")
print(links[:10])  # print first 10 links
