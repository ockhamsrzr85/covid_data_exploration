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
        ColToAgg('prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory', 'max'),
        ColToAgg('dementia_death_rate_per_100k', 'max'),
        ColToAgg('stroke_deaths_per_100k', 'max'),
        ColToAgg('diabetes_prevalence_per_100_ages_20to79', 'max'),
        ColToAgg('median_age', 'max'),
        ColToAgg('pop_over_65_per_100_2019', 'max'),
        ColToAgg('prevalence_of_obesity_both_sexes', 'max'),
        ColToAgg('prevalence_of_overweight_adults_both_sexes', 'max'),
        ColToAgg('total_vaccinations_per_hundred', 'sum'),
        ColToAgg('daily_cases_per_100k_biweekly_avg', 'sum')
    ]

    for col_settings in columns_to_examine + [response_col]:
        if col_settings.normalize:
            mx = full_data[col_settings.src_column].max()
            mn = full_data[col_settings.src_column].min()
            full_data[col_settings.normalize_col_name()] = (full_data[col_settings.src_column] - mn) / float(mx - mn)

    to_agg = dict([
        (col_settings.agg_col_name(),  pd.NamedAgg(column=col_settings.normalize_col_name(), aggfunc=col_settings.agg_fn))
        for col_settings in columns_to_examine + [response_col]])

    aggr = full_data.groupby(['location'], as_index=False).agg(**to_agg)

    fig, axes = plt.subplots(nrows=len(columns_to_examine), ncols=1)

    sublpot = 0
    for colidx in range(len(columns_to_examine)):
        col_settings = columns_to_examine[colidx]
        X = aggr[col_settings.agg_col_name()]
        y = aggr[response_col.agg_col_name()]

        colors = aggr['max(norm_pop_over_65_per_100_2019)'] * 100

        axes[sublpot].scatter(X, y, c=colors, cmap='GnBu', s=400)

        axes[sublpot].update(dict(title=f"{col_settings.src_column} vs All-time COVID-19 Deaths per 100k",
                                  xlabel=col_settings.agg_col_name(),
                                  ylabel=response_col.agg_col_name()))

        for i, txt in enumerate(aggr['location']):
            axes[sublpot].annotate(txt, (X[i], y[i]))

        X_ = sm.add_constant(X)
        model = sm.OLS(y, X_).fit()
        trend = model.predict(X_)
        axes[sublpot].plot(X, trend)

        sublpot = sublpot + 1

    plt.gcf().set_size_inches(15, sublpot * 6)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    print(aggr.dtypes)

    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}_{min_date}_to_{max_date}.svg")

    aggr.to_pickle(f"{OUT_DIR}/aggfeatures.pkl")


if __name__ == '__main__':
    run('2020-01-01', '2022-01-30')
    #run('2020-01-01', '2021-01-01')
    #run('2021-01-01', '2022-01-30')

