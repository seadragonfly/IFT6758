"""
This assignment is based on Greg Baker's Data Science course at SFU.

All areas requiring work are marked with a "TODO" label.
"""

import numpy as np


def city_lowest_precipitation(totals: np.array) -> int:
    """
    Given a 2D array where each row represents a city, and each column represents a month from January
    to December of a particular year, returns the city with the lowest total precipitation.
    """

    # TODO

    return None


def avg_precipitation_month(totals: np.array, counts: np.array) -> np.array:
    """
    Determine the average precipitation at these locations for each month. This will be the total
    precipitation for each month (axis 0), divided by the total observations for that month.
    """

    # TODO

    return None


def avg_precipitation_city(totals: np.array, counts: np.array) -> np.array:
    """
    Do the same for the cities: give the average precipitation (average daily rainfall over the month)
    for each city.
    """

    # TODO

    return None


def quarterly_precipitation(totals: np.array) -> np.array:
    """
    Calculate the total precipitation for each quarter in each city (i.e., the totals for each station over groups of three months). You can assume that the number of columns will be divisible by 3.

    Tip: Use the reshape function to reshape into a 4n by 3 array, sum, and reshape into n by 4.
    """
    if totals.shape[1] != 12:
        raise NotImplementedError("The entry table does not have 12 months!")

    # TODO

    return None


def main():
    data = np.load("data/monthdata.npz")
    totals = data["totals"]
    counts = data["counts"]

    # You can use this to steer your code
    print(f"Row with lowest precipitation: {city_lowest_precipitation(totals)}")
    print(f"Average precipitation per month: {avg_precipitation_month(totals, counts)}")
    print(f"Average precipitation per city: {avg_precipitation_city(totals, counts)}")
    print(f"Quarterly precipitation: {quarterly_precipitation(totals)}")


if __name__ == "__main__":
    main()
