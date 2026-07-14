"""Entry point: initially load the GEFCom2014 load-track data."""

from src.data import load_all_tasks


def main() -> None:
    data = load_all_tasks("data")
    print(data.head())


if __name__ == "__main__":
    main()
