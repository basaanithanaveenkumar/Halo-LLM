"""Download WikiText-2 once so training does not hit the hub on every import."""

from datasets import load_dataset


def main():
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    print("wikitext-2 cached")


if __name__ == "__main__":
    main()
