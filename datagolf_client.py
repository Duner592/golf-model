import os

import requests
import requests_cache
from dotenv import load_dotenv

load_dotenv()


class DataGolfClient:
    def __init__(self, base_url, cache_name="dg_cache", expire_after=900):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv("DATAGOLF_API_KEY")
        requests_cache.install_cache(cache_name, expire_after=expire_after)
        self.session = requests.Session()

    def get(self, endpoint_path, params=None):
        url = f"{self.base_url}/{endpoint_path.lstrip('/')}"
        params = params.copy() if params else {}
        params["key"] = self.api_key
        r = self.session.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
