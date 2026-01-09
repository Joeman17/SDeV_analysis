import json
import requests
import pandas as pd

class SoSciClient:
    def __init__(self, api_link, apikey=None):
        self.api_link = api_link
        self.apikey = apikey

    def send_json(self, data=None, params=""):
        payload = {"json_payload": json.dumps(data)}
        if self.apikey:
            payload["apikey"] = self.apikey
        return requests.get(self.api_link + params, data=payload)

    def load_variables(self):
        resp = self.send_json(params="&cases=none&infoValues")
        return pd.DataFrame(resp.json()["variables"])

    def load_data(self, vlist):
        params = "&vList=" + ",".join(vlist)
        resp = self.send_json(params=params)
        return resp.json()["data"]
    
    def load_dataframe(self, vlist):
        """
        Load variables and return a normalized DataFrame:
        - rows = cases
        - columns = variables
        - empty / trivial entries removed
        """
        raw = self.load_data(vlist)

        filtered = {
            k: v
            for k, v in raw.items()
            if isinstance(v, dict) and len(v) > 1
        }

        return pd.DataFrame(filtered).T
    
    def load_variable_dataframe(self, variables):
        if isinstance(variables, str):
            variables = [variables]
        return self.load_dataframe(variables)