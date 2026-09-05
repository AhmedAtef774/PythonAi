import pandas as pd
import json
import requests
import urllib3
import io
from config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)





def LoadData() -> pd.DataFrame:
    try:
        
        config = Config()
        headers = {"X-API-Key" : config.ABCSystemApiKey}
        response = requests.get(config.JsonPath , headers = headers , verify = False)
        response.raise_for_status()

        print(f"[Loader] Status: {response.status_code}")
        print(f"[Loader] Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError(
                f"Expected JSON response but got '{content_type}'. "
                f"The server may be returning an error page. "
                f"Response preview: {response.text[:200]}"
            )

        df = pd.read_json(io.StringIO(response.text), encoding="utf-8")
        # Build a combined text field for similarity search without
        # overwriting the original columns (needed for result output).
        df["Name"] = df.astype(str).agg(" ".join, axis=1)
        return df

        

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Error : Unauthrorized Check Api Key")
            return pd.DataFrame(columns=["Name"])
    except Exception as e:
        print("Exception occurred during data loading:")
        print(f"Error : {e}")
        return pd.DataFrame()
