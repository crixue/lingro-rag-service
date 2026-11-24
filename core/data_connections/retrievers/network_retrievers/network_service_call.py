import json
import logging
from typing import Dict, List

import requests

from configs.model_config import JAVA_MICRO_SERVICE_URL, LOG_FORMAT, WEB_SEARCH_COUNT

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

class SearchFetchParserServiceCall:
    def __init__(self,):
        pass

    def search_results(self, query: str, search_count: int = WEB_SEARCH_COUNT,) -> List[Dict[str, str]]:
        '''
        :return: {"title": "title",
                    "url": "url",
                    "text": "text"}
        '''
        try:
            resp = requests.post(url=JAVA_MICRO_SERVICE_URL["search-results"] + f"?query={query}&search_count={search_count}",
                                headers={"Content-Type": "application/json"},
                                )
            if resp.status_code != 200 or resp.json().get("code", -1) != 0:
                err_msg = resp.json().get("msg", -1)
                raise Exception(f"Error occurred when search_results by query {query}: {err_msg}")
            return resp.json()["data"]
        except Exception as e:
            logger.exception(f"Error occurred when search_results by query {query}: {e}")
            return [{"title": "Nan",
                    "url": "Nan",
                    "text": "Nan"}]


    def fetch_and_parse(self, urls: List[str]) -> Dict[str, str]:
        try:
            resp = requests.post(url=JAVA_MICRO_SERVICE_URL["fetch-and-parse-html-url"],
                                headers={"Content-Type": "application/json"},
                                data=json.dumps({
                                    urls: urls
                                }, ensure_ascii=False))
            if resp.code != 0:
                raise Exception(f"Error occurred when fetching and parsing {urls}: {resp.msg}")

            return resp.data
        except Exception as e:
            logger.exception(f"Error occurred when fetching and parsing {urls}: {e}")
            return {"Nan": "Nan"}
