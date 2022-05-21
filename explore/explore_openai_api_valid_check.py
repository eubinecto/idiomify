import os

import openai
import requests
from requests import HTTPError


def login(key: str):

    url = "https://api.openai.com/v1/engines"
    headers = {
        'Authorization': f"Bearer {key}"
    }
    r = requests.get(url=url, headers=headers)

    try:
        r.raise_for_status()
    except HTTPError as e:
        raise ValueError("Login failed")
    else:
        print(f"Logged in with (The key is valid): {key}. The key is NOT stored in any databases whatsover")


def main():
    # this is where that is stored
    login(openai.api_key)
    login("dd")


if __name__ == '__main__':
    main()
