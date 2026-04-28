import requests
import pandas as pd
import io

def test_day(code, day_str):
    url = f"https://www.meteocat.cat/observacions/xema/dades?codi={code}&dia={day_str}T00:00Z"
    print(f"Fetching {url}")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    print("Tables found:", len(tables))
    if len(tables) < 3:
        print("Less than 3 tables")
        return None
    df = pd.DataFrame(tables[2]).copy()
    # quick preview
    print(df.head())
    return df

if __name__ == '__main__':
    df = test_day('XF', '2008-10-24')
    print('Done', type(df))
