import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from covid_data_play.common import FEATURES_PLK_PATH, OUT_DIR, COUNTRIES_WITH_OECD_AVOIDABLE_DEATHS_STATS
import os


def run(min_date, max_date):
    full_data = pd.read_pickle(FEATURES_PLK_PATH)

    full_data = full_data[(full_data['date'] >= min_date) & (full_data['date'] <= max_date) &
                          (full_data['location'].isin(COUNTRIES_WITH_OECD_AVOIDABLE_DEATHS_STATS))]
    to_check = [
        ('mental_health_and_substance_use_deaths', 'max'),
        ('prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory', 'max'),
        ('dementia_death_rate_per_100k', 'max'),
        ('stroke_deaths_per_100k', 'max'),
        ('diabetes_prevalence_per_100_ages_20to79', 'max'),
        ('median_age', 'max'),
        ('preventable_causes_mortality_per_100_k', 'max'),
        ('treatable_causes_mortality_per_100_k', 'max'),
        ('prevalence_of_obesity_both_sexes', 'max'),
        ('prevalence_of_overweight_adults_both_sexes', 'max'),
        ('age_standardized_death_rate_cardiovascular', 'max'),
        ('monthly_sunshine_hours', 'sum'),
        ('monthly_average_temperature', 'sum'),
        ('monthly_sunshine_hours', 'median'),
        ('monthly_average_temperature', 'median'),
        ('total_vaccinations_per_hundred', 'sum')
    ]

    for c, f in to_check:
        mx = full_data[c].max()
        mn = full_data[c].min()
        full_data[f'norm_{c}'] = (full_data[c] - mn) / float(mx - mn)

    fig, axes = plt.subplots(nrows=len(to_check), ncols=1)

    y_col = 'daily_deaths_per_100k_biweekly_avg'
    y_col_agg_fn = 'sum'

    sublpot = 0
    for prop_to_examine, prop_to_examine_agg in to_check:
        prop_to_examine = 'norm_' + prop_to_examine
        prop_to_examine_agg_name = prop_to_examine_agg + '(' + prop_to_examine + ')'
        aggr = full_data.groupby(['location'], as_index=False).agg(
            x_label=pd.NamedAgg(column=prop_to_examine, aggfunc=prop_to_examine_agg),
            y=pd.NamedAgg(column=y_col, aggfunc=y_col_agg_fn))

        X = aggr['x_label']
        y = aggr['y']
        axes[sublpot].scatter(X, y)

        axes[sublpot].update(dict(title=f"{prop_to_examine} vs All-time COVID-19 Deaths per 100k",
                                  xlabel=prop_to_examine_agg_name,
                                  ylabel=f'{y_col_agg_fn}({y_col})'))

        for i, txt in enumerate(aggr['location']):
            axes[sublpot].annotate(txt, (X[i], y[i]))

        X_ = sm.add_constant(X)
        model = sm.OLS(y, X_).fit()
        trend = model.predict(X_)
        axes[sublpot].plot(X, trend)

        print(f"========= Examining {prop_to_examine}")
        print(model.params)
        print(f"# {prop_to_examine} OLS params ([const, x_label]),  {list(model.params[['const', 'x_label']])}")

        aggr.rename({'x_label': prop_to_examine_agg_name, 'y': f"{y_col_agg_fn}({y_col})"}, axis='columns', inplace=True)
        print(aggr.to_csv())

        print("\n=====================\n")

        sublpot = sublpot + 1

    plt.gcf().set_size_inches(10, sublpot * 6)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}_{min_date}_to_{max_date}.svg")
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}_{min_date}_to_{max_date}.png")


if __name__ == '__main__':
    run('2020-01-01', '2022-01-30')
    run('2020-01-01', '2021-01-01')
    run('2021-01-01', '2022-01-30')

