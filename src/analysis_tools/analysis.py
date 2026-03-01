import pandas as pd


def main():
    df = pd.read_csv("data/news/005380/index.csv")
    df["published_dt"] = df["published_dt"].astype(str)
    min_dt = df["published_dt"].min()
    print(min_dt)


if __name__ == "__main__":
    main()
