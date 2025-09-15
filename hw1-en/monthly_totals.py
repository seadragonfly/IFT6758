"""
This assignment is based on Greg Baker's Data Science course at SFU.

By the end of this assignment, you should be convinced that using Pandas' native features is better than doing everything yourself. You should also feel comfortable with DataFrames and know how to pivot these objects to achieve your goals.

All areas requiring work are marked with a "TODO" label.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from geopy import distance

from typing import Tuple


def get_precip_data(fp: str = "data/precipitation.csv") -> pd.DataFrame:
    return pd.read_csv(fp, parse_dates=[2])


def date_to_month(d: pd.Timestamp) -> str:
    """
    You may need to modify this function, depending on your data types (if they don't match
    the expected input types).
    """
    return "%04i-%02i" % (d.year, d.month)


def pivot_months_pandas(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create monthly precipitation totals for each station in the dataset.

    This should use Pandas methods to manipulate the data. Round the precipitation (mm) to the first
    decimal place.
    """
    monthly, counts = None, None

    # TODO

    return monthly, counts


def pivot_months_loops(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create monthly precipitation totals for each station in the dataset.
    The hard way: Use Pandas as a dumb data store and iterate in Python.

    Never do things this way!!!
    """
    # Find all stations and months in the dataset.
    stations = set()
    months = set()
    for i, r in data.iterrows():
        stations.add(r["name"])
        m = date_to_month(r["date"])
        months.add(m)

    # Aggregate into dictionaries so we can look it up later.
    stations = sorted(list(stations))
    row_to_station = dict(enumerate(stations))
    station_to_row = {s: i for i, s in row_to_station.items()}

    months = sorted(list(months))
    col_to_month = dict(enumerate(months))
    month_to_col = {m: i for i, m in col_to_month.items()}

    # Create tables for the data and fill them in.
    precip_total = np.zeros((len(row_to_station), 12), dtype=np.float64)
    obs_count = np.zeros((len(row_to_station), 12), dtype=np.float64)

    for _, row in data.iterrows():
        m = date_to_month(row["date"])
        r = station_to_row[row["name"]]
        c = month_to_col[m]

        precip_total[r, c] += row["precipitation"]
        obs_count[r, c] += 1

    # Build the DataFrames we needed all along (tidying up the index names while we're at it).
    totals = pd.DataFrame(
        data=np.round(precip_total, 1),
        index=stations,
        columns=months,
    )
    totals.index.name = "name"
    totals.columns.name = "month"

    counts = pd.DataFrame(
        data=obs_count.astype(int),
        index=stations,
        columns=months,
    )
    counts.index.name = "name"
    counts.columns.name = "month"

    return totals, counts


def compute_pairwise(df: pd.DataFrame, func: callable) -> pd.DataFrame:
    """
    Complete this function, which takes a dataframe and a function from a pair of columns in the dataframe
    as input and returns a dataframe containing the function applied to

    each pair of rows in the dataframe**.

    To do this, we'll use the `pdist` and `squareform` functions from the `scipy.spatial` library.
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    Tip: Make sure the input dataframe has the station name as the index, not a number! You can do this by rotating
    the dataframe. It should look like this snippet:

    ```
                             column1     column2
    name
    BURNABY SIMON FRASER U   ...        ...
    CALGARY INTL A           ...        ...
    ```
    """
    new_df = None

    # TODO
    # use scipy.spatial.pdist and scipy.spatial.squareform

    return new_df


def geodesic(latlon1, latlon2) -> int:
    """
    Defines a metric between two points; in our case, our two points are latitude/longitude coordinates.
    "We" need to do geometry if we want to obtain the distance between two points on an ellipsoid (Earth),
    but we'll abstract this functionality to another geopy. You can learn more about
    the mathematics here:
    - https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid

    A simplification of this is to consider a sphere instead:
    - https://en.wikipedia.org/wiki/Haversine_formula
    """
    return int(distance.distance(tuple(latlon1), tuple(latlon2)).km)


def compute_pairwise_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the `compute_pairwise()` and `geodesic()` functions defined above,
    compute the distance between each of the stations. The input must be the original raw data frame
    loaded from the CSV.
    """
    new_df = None

    # TODO: rotate the dataframe so that you have lat/lon as columns and names as indexes

    return new_df


def correlation(u, v) -> float:
    """
    Calculate the correlation between two data sets
    - https://en.wikipedia.org/wiki/Correlation

    More precisely, the equation for Pearson's product-moment coefficient is:

        corr = E[(X - x_avg) * (Y - y_avg)] / (x_std * y_std)

    """
    corr = None

    # get proper indices (filter out NaNs; '~' is logical 'not')
    idx_u = ~pd.isna(u)
    idx_v = ~pd.isna(v)
    idx = idx_u & idx_v

    # TODO: calculate the mean and standard deviation of valid entries

    # TODO: calculate the correlation

    return corr


def compute_pairwise_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """

    Given the `compute_pairwise()` and `correlation()` functions completed above, calculate the
    pairwise correlation of daily precipitation between stations. The goal here is to see if there is a
    correlation of precipitation between stations. Ideally, we would expect stations closer
    to each other to have a higher correlation. The input should be the original raw dataframe loaded
    from the CSV.

    Note that you will likely have a diagonal of zeros when it should be ones—this is fine
    for the purposes of this assignment. `pdist` expects the metric function to be a proper metric,
    that is, the distance between an element and itself is zero.

    """
    new_df = None

    # TODO: rotate the dataframe so that you have a column for each date, and the station names are the indices.

    return new_df


def compute_pairwise_correlation_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Surprise! Pandas can actually do the correlation calculation for you in a single function call.

    You'll pivot the table slightly differently, then make a single function call on the dataframe:
    - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html

    You should get the same result as you did for `compute_pairwise_correlation()`, with
    the exception of ones (correctly) along the diagonal.
    """
    new_df = None

    # TODO

    return new_df


def main():
    data = get_precip_data()
    totals, counts = pivot_months_loops(data)

    # Optionally create the data...
    #totals.to_csv("data/totals.csv")
    #counts.to_csv("data/counts.csv")
    #np.savez("data/monthdata.npz", totals=totals.values, counts=counts.values)

    # pivot monthspandas
    totals_pd, counts_pd = pivot_months_pandas(data)
    assert all(abs(totals - totals_pd).max() < 1e-10), "totals != totals_pd"
    assert all(abs(counts - counts_pd).max() < 1e-10), "counts != counts_pd"

    # calculate pairwise
    test_df = pd.DataFrame([[0, 0], [0, 1], [1, 0]], columns=["x", "y"], index=list("abc"))
    euclidean = lambda xy1, xy2: np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)
    expected_output = pd.DataFrame(
        {'a': {'a': 0, 'b': 1, 'c': 1},
        'b': {'a': 1, 'b': 0, 'c': np.sqrt(2)},
        'c': {'a': 1, 'b': np.sqrt(2), 'c': 0}}
    )
    output = compute_pairwise(test_df, euclidean)
    assert np.allclose(output, expected_output)

    # pairwise distances
    print(compute_pairwise_distances(data))

    # pairwise correlation
    print(compute_pairwise_correlation(data))

    # pairwise correlation
    print(compute_pairwise_correlation_pandas(data))

    print("Finished!")


if __name__ == "__main__":
    main()
