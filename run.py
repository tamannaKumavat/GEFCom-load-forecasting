"""Entry point: initially load the GEFCom2014 load-track data."""

from src.config import load_config
from src.data import load_all_tasks


def main() -> None:
    config = load_config("configs/default.yaml")
    data = load_all_tasks(config["data"]["raw_dir"])
    print(data.head())


if __name__ == "__main__":
    main()
