"""
This assignment is based on Greg Baker's Data Science course at SFU.

All areas requiring work are marked with a "TODO" label.
"""
import pandas as pd


def city_lowest_precipitation(totals: pd.DataFrame) -> str:
    """
    Given a dataframe where each row represents a city and each column represents a month
    from January to December of a particular year, return the city with the lowest total precipitation.
    """

    # TODO

    return None


def avg_precipitation_month(totals: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the average precipitation at these locations for each month. This will be the total precipitation for
    each month, divided by the total observations for that month.
    """

    # TODO

    return None


def avg_precipitation_city(totals: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """
Do the same for cities: give the average precipitation (average daily precipitation over the month)
for each city.
    """

    # TODO

    return None


# pas de trimestriel car c'est un peu pénible


def main():
    totals = pd.read_csv("data/totals.csv").set_index(keys=["name"])
    counts = pd.read_csv("data/counts.csv").set_index(keys=["name"])

    # You can use this to steer your code
    print(f"Row with the lowest precipitation: {city_lowest_precipitation(totals)}")
    print(f"Average precipitation per month: {avg_precipitation_month(totals, counts)}")
    print(f"Average precipitation per city: {avg_precipitation_city(totals, counts)}")


if __name__ == "__main__":
    main()





