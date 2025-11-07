import json

from datagolf_client import DataGolfClient

# Fill these with real documentation values:
BASE_URL = "https://feeds.datagolf.com"
ENDPOINT_SAMPLE = "get-schedule"


def main():
    dg = DataGolfClient(BASE_URL)
    result = dg.get(ENDPOINT_SAMPLE, params={"tour": "pga"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
