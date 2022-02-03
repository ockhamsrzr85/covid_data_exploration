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

    @dataclass
    class ColToAgg:
        src_column: str
        agg_fn: str
        normalize: bool = True

        def normalize_col_name(self):
            if self.normalize:
                return f'norm_{self.src_column}'
            else:
                return self.src_column

        def agg_col_name(self):
            return f'{self.agg_fn}({self.normalize_col_name()})'

    # response_col = ColToAgg('daily_deaths_per_100k_biweekly_avg', 'sum', normalize=False)
    response_col = ColToAgg('daily_deaths_per_100k_biweekly_avg', 'sum', normalize=False)

    columns_to_examine = [
        ColToAgg('mental_health_and_substance_use_deaths', 'max'),
        ColToAgg('prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory', 'max', normalize=False),
        ColToAgg('dementia_death_rate_per_100k', 'max'),
        ColToAgg('stroke_deaths_per_100k', 'max'),
        ColToAgg('diabetes_prevalence_per_100_ages_20to79', 'max'),
        ColToAgg('median_age', 'max'),
        ColToAgg('pop_over_65_per_100_2019', 'max'),
        ColToAgg('prevalence_of_obesity_both_sexes', 'max'),
        ColToAgg('prevalence_of_overweight_adults_both_sexes', 'max'),
        ColToAgg('total_vaccinations_per_hundred', 'sum', normalize=False),
        ColToAgg('daily_cases_per_100k_biweekly_avg', 'sum', normalize=False),
        ColToAgg('daily_deaths_per_100k_biweekly_avg', 'sum', normalize=False),
        ColToAgg('hypertension_prevalence', 'max'),
    ]

    for col_settings in columns_to_examine + [response_col]:
        if col_settings.normalize:
            mx = full_data[col_settings.src_column].max()
            mn = full_data[col_settings.src_column].min()
            full_data[col_settings.normalize_col_name()] = (full_data[col_settings.src_column] - mn) / float(mx - mn)

    to_agg = dict([
        (
        col_settings.agg_col_name(), pd.NamedAgg(column=col_settings.normalize_col_name(), aggfunc=col_settings.agg_fn))
        for col_settings in columns_to_examine + [response_col]])

    aggr = full_data.groupby(['location'], as_index=False).agg(**to_agg)

    colors = aggr['sum(daily_deaths_per_100k_biweekly_avg)'] / aggr['sum(daily_cases_per_100k_biweekly_avg)']
    normalized_colors = (colors - colors.min()) / (colors.max() - colors.min()) * 100.0

    # xc = 'norm_prevalence_of_obesity_both_sexes'
    # yc = 'norm_prevalence_of_overweight_adults_both_sexes'

    # xc = 'norm_prevalence_of_overweight_adults_both_sexes'
    # yc = 'norm_prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory'

    xc = 'norm_diabetes_prevalence_per_100_ages_20to79'
    yc = 'norm_hypertension_prevalence'

    X = aggr[f'max({xc})']
    y = aggr[f'max({yc})']

    plt.scatter(X, y, c=normalized_colors, cmap='Reds', s=400)

    plt.title("...")
    plt.xlabel(xc)
    plt.ylabel(yc)
    for i, txt in enumerate(aggr['location']):
        plt.annotate(txt, (X[i], y[i]))

    ax = aggr.plot.hexbin(
        x=f'max(norm_diabetes_prevalence_per_100_ages_20to79)',
        y=f'max(norm_hypertension_prevalence)', gridsize=20)

    plt.show()


if __name__ == '__main__':
    run('2020-01-01', '2022-01-30')
    # run('2020-01-01', '2021-01-01')
    # run('2021-01-01', '2022-01-30')
