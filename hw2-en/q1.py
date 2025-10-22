import json
import pandas as pd


def count_labels(labels: str) -> int:
    output=len(labels.split(','))
    
    return output


def convert_id(ID: str) -> str:
    with open("data/ontology.json", "r") as f:
        ontology = json.load(f)
    ontology_dict={}
    for i in ontology:
        if "id" in i and "name" in i:
            ontology_dict[i["id"]] = i["name"]

    output=str(ontology_dict[ID])
    # TODO
    return output


def convert_ids(labels: str) -> str:
    parts=[]
    for i in labels.split(','):
        name=convert_id(i)
        parts.append(name)  
    output = "|".join(parts)
    
    return output


def contains_label(labels: pd.Series, label: str) -> pd.Series:
    valid_idx = []

    for i in range(len(labels)):
        val = labels.iloc[i]                 
        if pd.isna(val):
            continue                         
        tokens = [t.strip() for t in str(val).split("|")] 
        if label in tokens:
            valid_idx.append(labels.index[i])  

    output = labels.loc[valid_idx]       
    return output


def get_correlation(labels: pd.Series, label_1: str, label_2: str) -> float:
    """
    Create a function that, given a Series as described above, returns the proportion of rows
    with label_1 that also have label_2. Make use of the function you created above.

    For example, suppose the Series has 1000 values, of which 120 have label_1. If 30 of the 120
    have label_2, your function should return 0.25.
    """
    s_1=contains_label(labels, label_1)
    count_1=len(s_1)
    if count_1==0:
        return 0
    s_2=contains_label(s_1, label_2)
    count_2=len(s_2)

    output= count_2/count_1
    
    return output


if __name__ == "__main__":
    print(count_labels("/m/04rlf,/m/06_fw,/m/09x0r"))
    print(convert_id("/m/04rlf"))
    print(convert_ids("/m/04rlf,/m/06_fw,/m/09x0r"))

    series = pd.Series([
        "Music|Skateboard|Speech",
        "Voice|Speech",
        "Music|Piano"
    ])
    print(contains_label(series, "Music"))
    print(get_correlation(series, "Music", "Piano"))
