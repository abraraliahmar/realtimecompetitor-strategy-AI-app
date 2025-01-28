from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
from datetime import datetime, timedelta


# Function to generate a random date within the last year
def generate_random_date():
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    random_days = random.randint(0, 365)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")


def get_title(soup):
    try:
        title=soup.find("span",attrs={"id":"productTitle"})
        title_value=title.text
        title_string=title_value.strip()

    except AttributeError:
        title_string = ""

    return title_string

def get_selling_price(soup):

    try:
        price = soup.find("span", attrs={'class':'a-price-whole'}).text.strip()
        price = price.replace(".", "")

    except AttributeError:
        price = ""

    return price

def get_MRP(soup):

    try:
        price = soup.find("span", attrs={'class':'a-size-small aok-offscreen'}).text.strip()
        # Use regular expression to extract only the numeric value
        price = price.replace("M.R.P.: ₹", "")

    except AttributeError:
        price = ""

    return price

def get_discount(soup):

    try:
        discount = soup.find("span", attrs={'class':'a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage'}).text.strip()

        # Remove the minus sign if it exists
        if discount.startswith('-'):
            discount = discount[1:]

    except AttributeError:
        discount = ""

    return discount

def get_rating(soup):
    try:
        rating = soup.find("i", attrs={'class':'a-icon a-icon-star a-star-4-5'}).string.strip()

    except AttributeError:
        try:
            rating = soup.find("span", attrs={'class':'a-icon-alt'}).string.strip()
        except:
            rating = ""

    return rating

def get_reviews(soup):
    try:
        reviews = soup.find("span", attrs={'id':'acrCustomerReviewText'}).string.strip()

    except AttributeError:
        reviews = ""

    return reviews

def get_availability(soup):
    try:
        available = soup.find("div", attrs={'id':'availability'})
        available = available.find("span").string.strip()

    except AttributeError:
        available = "Available"

    return available

def main():
    base_url = "https://www.amazon.in"
    URL = "https://www.amazon.in/s?k=earphones&crid=23H19CC51YB96&sprefix-earphone%2Caps%2C228&ref=nb_sb_noss_2"
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}
    base_responses = requests.get(URL, headers=headers)
    cookies = base_responses.cookies
    responses = requests.get(URL, headers=headers, cookies=cookies)
    soup = BeautifulSoup(responses.content, "html.parser")
    links = soup.find_all("a", attrs={'class': 'a-link-normal s-no-outline'})
    links_list = []
    for link in links:
        links_list.append(link.get('href'))

    d = {"date": [], "title": [], "selling_price": [], "MRP": [], "discount": [], "rating": [], "reviews": [], "availability": []}
    for link in range(min(15, len(links_list))):
        if not links_list[link].startswith("http"):
            new_webpage = requests.get(base_url + links_list[link], headers=headers, cookies=cookies)
        else:
            new_webpage = requests.get(links_list[link], headers=headers, cookies=cookies)

        new_soup = BeautifulSoup(new_webpage.content, "html.parser")
        d['date'].append(generate_random_date())
        d['title'].append(get_title(new_soup))
        d['selling_price'].append(get_selling_price(new_soup))
        d['MRP'].append(get_MRP(new_soup))
        d['discount'].append(get_discount(new_soup))
        d['rating'].append(get_rating(new_soup))
        d['reviews'].append(get_reviews(new_soup))
        d['availability'].append(get_availability(new_soup))

    df = pd.DataFrame.from_dict(d)
    df['title'].replace('', np.nan, inplace=True)
    df = df.dropna(subset=['title'])
    df.to_csv("amazon_scraped_data.csv", header=True, index=False)


# Call the main function
if __name__ == '__main__':
    main()
