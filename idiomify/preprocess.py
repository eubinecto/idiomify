from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def upsample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    # TODO: implement upsampling later
    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return:
    """
    # TODO: implement cleansing
    return df


def annotate(df: pd.DataFrame, boi_token: str, eoi_token: str) -> pd.DataFrame:
    """
    e.g.
    given a row like this:
    Idiom                                                 keep an eye on
    Sense                   keep a watch on something or someone closely
    Idiomatic_Sent     He had put on a lot of weight lately , so he started keeping an eye on what he ate .
    Literal_Sent       He had put on a lot of weight lately , so he started to watch what he ate .
    Idiomatic_Label            O O O O O O O O O O O O O B I I O O O O O
    Literal_Label                  O O O O O O O O O O O O O B I O O O O

    use Idiomatic_Label to replace Idiomatic_Sent with:
    He had put on a lot of weight lately , so he started <idiom> keeping an eye on </idiom> what he ate .
    """
    for idx, row in df.iterrows():
        tokens = row['Idiomatic_Sent'].split(" ")
        labels = row["Idiomatic_Label"].split(" ")
        if "B" in labels:
            boi_idx = labels.index("B")
            if "I" in labels:
                eoi_idx = -1 * (list(reversed(labels)).index("I") + 1)
                tokens[boi_idx] = f"{boi_token} {tokens[boi_idx]}"
                tokens[eoi_idx] = f"{tokens[eoi_idx]} {eoi_token}"
            else:
                tokens[boi_idx] = f"{boi_token} {tokens[boi_idx]} {eoi_token}"
            row['Idiomatic_Sent'] = " ".join(tokens)

    return df


def stratified_split(df: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    stratified-split the given df into two df's.
    """
    total = len(df)
    ratio_size = int(total * ratio)
    other_size = total - ratio_size
    ratio_df, other_df = train_test_split(df, train_size=ratio_size,
                                          stratify=df['Idiom'],
                                          test_size=other_size, random_state=seed,
                                          shuffle=True)
    return ratio_df, other_df
