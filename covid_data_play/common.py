import pandas as pd
from datetime import datetime


OMICRON_VARIANT_NAME = "B.1.1.529"

OUT_DIR = "../../data_output"

FEATURES_PLK_PATH = f"{OUT_DIR}/features.pkl"
FEATURES_CSV_PATH = f"{OUT_DIR}/features.csv"

def data_file_path(name):
    return f'../../data/{name}.csv'


def read_file(name):
    return pd.read_csv(data_file_path(name))


def str_date_to_year_week(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    w = parsed_date.isocalendar()[1]
    return f"{parsed_date.isocalendar()[0]}-{'0' if w <10 else ''}{w}"


def str_date_to_month(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    return parsed_date.month


def week_day(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    return parsed_date.weekday()
