import requests
import pandas as pd

from bs4 import BeautifulSoup

URL = "https://czystepowietrze.gov.pl/wez-dofinansowanie/pytania-i-odpowiedzi/nowy-program-czyste-powietrze-obowiazujacy-od-31-marca-2025/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(URL, headers=headers, verify=False)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

faq_items = soup.select(".accordion-item")

faq_data = []
for item in faq_items:
    question_tag = item.find("button")
    answer_tag = item.select_one(".card-body .content")
    if question_tag and answer_tag:
        question = question_tag.get_text(strip=True)
        answer = answer_tag.get_text(separator=" ", strip=True)
        faq_data.append({"question": question, "answer": answer})

df = pd.DataFrame(faq_data)

df.to_csv('czyste-powietrze_FAQ.csv', index=False)
