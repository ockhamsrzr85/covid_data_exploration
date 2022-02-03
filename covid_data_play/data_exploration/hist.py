import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from covid_data_play.common import FEATURES_PLK_PATH, OUT_DIR, COUNTRIES_WITH_GOOD_ENOUGH_DATA
import os
from dataclasses import dataclass


def run(min_date, max_date):
    full_data = pd.read_pickle(FEATURES_PLK_PATH)

    full_data = full_data[(full_data['date'] >= min_date) & (full_data['date'] <= max_date) &
                          (full_data['location'].isin(COUNTRIES_WITH_GOOD_ENOUGH_DATA))]

    full_data.fillna(0.0, inplace=True)

    # plt.show()

    df_num = full_data[['prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory',
                        'pop_over_65_per_100_2019',
                        'total_vaccinations_per_hundred']]  # .select_dtypes(include=['float64', 'int64'])

    aggr = full_data.groupby(['location'], as_index=False).agg(
        prob_of_dying_30_70_from_any_of=pd.NamedAgg(
            column='prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory',
            aggfunc='max'),
        pop_over_65_per_100_2019=pd.NamedAgg(column='pop_over_65_per_100_2019',
                                             aggfunc='max'),
        total_vax=pd.NamedAgg(column='total_vaccinations_per_hundred',
                              aggfunc='max'),
        daily_cases_per_100k_biweekly_avg=pd.NamedAgg(column='daily_cases_per_100k_biweekly_avg',
                                                      aggfunc='sum'),
        daily_deaths_per_100k_biweekly_avg=pd.NamedAgg(column='daily_deaths_per_100k_biweekly_avg',
                                                       aggfunc='sum'),
        stroke_deaths_per_100k=pd.NamedAgg(column='stroke_deaths_per_100k',
                                           aggfunc='max'),
        diabetes_prevalence_per_100_ages_20to79=pd.NamedAgg(column='diabetes_prevalence_per_100_ages_20to79',
                                                            aggfunc='max'),
        prevalence_of_obesity_both_sexes=pd.NamedAgg(column='prevalence_of_obesity_both_sexes',
                                                     aggfunc='max'),
        prevalence_of_overweight_adults_both_sexes=pd.NamedAgg(column='prevalence_of_overweight_adults_both_sexes',
                                                               aggfunc='max'),
        dementia_death_rate_per_100k=pd.NamedAgg(column='dementia_death_rate_per_100k',
                                                 aggfunc='max'),
        hypertension_prevalence=pd.NamedAgg(column='hypertension_prevalence', aggfunc='max')
    )
    print(aggr.head())

    aggr.hist(figsize=(16, 20), bins=30, xlabelsize=8, ylabelsize=8)

    plt.show()


if __name__ == '__main__':
    run('2020-01-01', '2022-01-30')
    # run('2020-01-01', '2021-01-01')
    # run('2021-01-01', '2022-01-30')

##  todo pozerat vakcinovanost >60
