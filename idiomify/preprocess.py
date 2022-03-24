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


def replace_labels(df: pd.DataFrame) -> pd.DataFrame:
    for idx, row in df.iterrows():
        idiom = row['Idiom']
        row['Literal_Label'] = "|".join([
            f"B/{idiom}" if label == "B" else
            f"I/{idiom}" if label == "I" else
            "O"
            for label in row['Literal_Label'].split(" ")
        ])
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
