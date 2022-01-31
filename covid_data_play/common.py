import pandas as pd
from datetime import datetime

OMICRON_VARIANT_NAME = "B.1.1.529"

OUT_DIR = "../../data_output"

FEATURES_PLK_PATH = f"{OUT_DIR}/features.pkl"
FEATURES_CSV_PATH = f"{OUT_DIR}/features.csv"

COUNTRIES_WITH_GOOD_ENOUGH_DATA = [
    'Albania', 'Australia', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Canada', 'Chile',
    'Colombia', 'Costa Rica', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Mexico',
    'Moldova', 'Montenegro', 'Netherlands', 'New Zealand', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania',
    'Russia', 'Serbia', 'Slovakia', 'Slovenia', 'South Korea', 'South Africa', 'Spain', 'Sweden', 'Switzerland',
    'Turkey', 'Ukraine', 'United Kingdom', 'United States']

COUNTRIES_WITH_OECD_AVOIDABLE_DEATHS_STATS = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Czechia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Latvia',
    'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovakia',
    'Slovenia', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States']


def data_file_path(name):
    return f'../../data/{name}.csv'


def read_file(name):
    return pd.read_csv(data_file_path(name))


def str_date_to_year_week(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    w = parsed_date.isocalendar()[1]
    return f"{parsed_date.isocalendar()[0]}-{'0' if w < 10 else ''}{w}"


def str_date_to_month(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    return parsed_date.month


def week_day(dt):
    parsed_date = datetime.strptime(dt, '%Y-%m-%d')
    return parsed_date.weekday()
