"""
All sections that require work are marked "TODO".
"""
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm.autonotebook import tqdm
from typing import Tuple
import xml.etree.ElementTree as ET


# TODO - complétez cette méthode
def parse_xml(path: str) -> pd.DataFrame:
    path = Path(path)

    # unzip xml data if unavailable
    if not path.is_file():
        if path.with_suffix(".zip").is_file():
            shutil.unpack_archive(path.with_suffix(".zip"), path.parent)
        else:
            raise FileNotFoundError("%s nor an archive exists." % path)

    # define function to extract value, define tags to extract 
    _get_val = lambda entry, tag: entry.find(tag).text
    TARGET_ATTRIBUTES = ["name", "rating", "sex", "birthday", "country", "flag", "title"]

    tree = ET.parse(path)
    root = tree.getroot()
    rows=[]

    for person in root.findall('.//player'):
        row={tag: _get_val(person, tag) for tag in TARGET_ATTRIBUTES}
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=TARGET_ATTRIBUTES)

    return df


# TODO - complete this method
def clean_data(df: pd.DataFrame, year_cutoff: int) -> pd.DataFrame:
    out=df.copy()
    out = out.dropna(subset=["birthday"])
    out["birthday"] = pd.to_numeric(out["birthday"]).astype(int)
    out["rating"] = pd.to_numeric(out["rating"]).astype(int)
    mask = out["birthday"] > year_cutoff
    out = out[~mask]

    return out


# TODO - complétez cette méthode
def bin_counts(df: pd.DataFrame, bins: list, bin_centers: list) -> pd.DataFrame:
    """ 
    Returns a data frame with the `ratings` grouped between the values ​​given in `bins`, and
    with a label given in `bin_centers`. In addition to the raw count, also add a normalized bin.
    Count the column named "count_norm" by dividing the counts by the sum of the counts.

    Ex: Given
    >> x = pd.DataFrame({'rating' : [1, 2, 4, 6, 6, 7, 8, 11]})
    >> bins = [0, 5, 10, 15]
    >> bin_centers = [2.5, 7.5, 12.5]

    bin_counts(x, bins, bin_centers) should return:
    >> rating count count_norm
    >> 0 7.5 4 0.500
    >> 1 2.5 3 0.375
    >> 2 12.5 1 0.125

    Arguments:
    df (pd.DataFrame): Cleaned dataframe with at least the 'rating' column
    bins (list): Defines the bin boundaries; i.e., [0, 5, 10] defines two classes: [0, 5) and [5, 10]
    bin_centers (list): Defines the labels that will be used; i.e., [2.5, 7.5] for the above
    will output values ​​like [0, 5) -> 2.5, and [5, 10) -> 7.5.

    Returns:
    pd.DataFrame: Dataframe with grouped values; must have the columns
    ['rating', 'count', 'count_norm']. Rename them if necessary.
    """
    if 'rating' not in df.keys():
        raise ValueError("Incorrect input format; 'rating' must be a column in the input dataframe")

    df["rating_bin"] = pd.cut(df["rating"], bins=bins, right=False)
    bin_counts = df["rating_bin"].value_counts().sort_index()
    total = bin_counts.sum()

    hist = pd.DataFrame({
        "rating": bin_centers,           
        "count": bin_counts.values,     
    })

    hist["count_norm"] = hist["count"] / hist["count"].sum()

    
    hist = hist.sort_values(by="count", ascending=False).reset_index(drop=True)

    return hist


class PermutationTest:
    def __init__(self, df: pd.DataFrame, n_overrep: int, n_underrep: int):
        """ 
        Implements the permutation test experiment.

        Arguments:
        df(pd.DataFrame): Full data frame to partition (i.e., include both groups)
        n_overrep(integer): Number of elements in the overrepresented group
        n_underrep(integer): Number of elements in the underrepresented group

        n_overrep + n_underrep should be == len(df)! Technically < len(df) is also correct...
         """
        if len(df) < n_overrep + n_underrep:
            raise ValueError(f"Sum of n_overrep + n_underrep must be <= len(df)")
        self.df = df
        self.n_overrep = n_overrep
        self.n_underrep = n_underrep
    
    def job(self, seed: int = None) -> Tuple[int, int]:

        if seed is not None:
            np.random.seed(seed)

        permuted_scores = np.random.permutation(self.df["rating"])

        overrep_group = permuted_scores[:self.n_overrep]
        underrep_group = permuted_scores[self.n_overrep:self.n_overrep + self.n_underrep]
        
        over=int(np.max(overrep_group))
        under=int(np.max(underrep_group))
        return (over,under)


def sample_two_groups(
    df: pd.DataFrame, n_overrep: int, n_underrep: int, n_iter: int=1000
) -> Tuple[np.array, np.array]:

    best_over = []
    best_under = []
    pt = PermutationTest(df, n_overrep, n_underrep)

    for _ in tqdm(range(n_iter)):
        over, under = pt.job()
        best_over.append(over)
        best_under.append(under)

        
    return np.array(best_over), np.array(best_under)


