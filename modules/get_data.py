import re
import yaml
import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


def fetch_by_page_no(page_no, data, url):
    # Get
    res = requests.get(url.format(page_no))

    # Parse
    soup = BeautifulSoup(res.content, 'lxml')

    # Find all Property Information Cards
    data_cards = soup.find_all("li", class_="cardholder")

    # Parse the required data
    for data_card in data_cards:
        data.append({
            "BHK": data_card.select_one('a strong span.val').text.strip(),
            "SQFT": data_card.select_one('tr.hcol td.size span.val').text.strip(),
            "Status": data_card.select_one('tr.hcol.w44 td.val').text.strip(),
            "Age": re.match(
                ".+(New|Resale).+",
                data_card.select_one('ul.listing-details').text
            ).group(1),
            "Price": data_card.select_one('span.val[itemprop="offers"]').text.strip(),
            "Unit": data_card.select_one('span.unit').text.strip()
        })


def fetch_data_():

    url = "https://www.makaan.com/kolkata-residential-property/"\
          "buy-property-in-kolkata-city?page={}"

    num_pages = 5

    page_nos = [i for i in range(num_pages)]
    data = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(
            fetch_by_page_no,
            page_nos,
            [data] * num_pages,
            [url] * num_pages
        )

    cols = ["BHK", "Price", "Unit", "SQFT", "Status", "Age"]

    df = DataFrame(data, columns=cols)

    return df