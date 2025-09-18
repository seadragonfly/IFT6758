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

    monthly, counts = None, None

    data["month"]=data["date"].apply(date_to_month)
    x=data.groupby(["name", "month"])["precipitation"].sum().reset_index()
    monthly=x.pivot(index="name", columns="month", values="precipitation")
    y=data.groupby(["name", "month"])["precipitation"].count().reset_index()
    counts=y.pivot(index="name", columns="month", values="precipitation")

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

    distances = pdist(df, metric=func)
    distances_square = squareform(distances)                 
    new_df=pd.DataFrame(distances_square, index=df.index, columns=df.index)

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

    columns_to_keep = ["name", "latitude", "longitude"]
    new_df = df[columns_to_keep]
    new_df = new_df.drop_duplicates(subset="name")
    new_df=new_df.set_index("name")
    new_df= compute_pairwise(new_df,geodesic)

    return new_df


def correlation(u, v) -> float:
    corr = None

    idx_u = ~pd.isna(u)
    idx_v = ~pd.isna(v)
    idx = idx_u & idx_v

    u_filter = u[idx]
    v_filter = v[idx]    
    
    u_mean = np.mean(u_filter)
    v_mean = np.mean(v_filter)
    u_std  = np.std(u_filter)
    v_std  = np.std(v_filter)

    cov = np.mean((u_filter - u_mean) * (v_filter- v_mean))
    corr = cov / (u_std * v_std)

    return corr


def compute_pairwise_correlation(df: pd.DataFrame) -> pd.DataFrame:

    daily=df.pivot(index="name", columns="date", values="precipitation")
    new_df = compute_pairwise(daily, correlation)

    return new_df


def compute_pairwise_correlation_pandas(df: pd.DataFrame) -> pd.DataFrame:
    new_df = None
    daily=df.pivot(index="date", columns="name", values="precipitation")
    new_df = daily.corr(method="pearson")
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
