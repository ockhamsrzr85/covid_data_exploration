import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from covid_data_play.common import FEATURES_PLK_PATH, OUT_DIR
import os

if __name__ == '__main__':
    full_data = pd.read_pickle(FEATURES_PLK_PATH)

    # full_data = full_data[full_data['date'] >= '2020-04-01']

    aggr = full_data.groupby(['location'], as_index=False).agg(
        person_days_with_vax_per_100=pd.NamedAgg(column="total_vaccinations_per_hundred", aggfunc="sum"),
        median_owd_cfr_over_100_cases_only=pd.NamedAgg(column="owd_cfr_over_100_cases_only", aggfunc="median"))

    print(aggr['location'].to_string())

    fig, ax = plt.subplots()

    X = aggr['person_days_with_vax_per_100']
    y = aggr['median_owd_cfr_over_100_cases_only']
    ax.scatter(X, y)

    plt.gca().update(dict(title='Ordinary Least Squares regression: person_days_with_vax_per_100 vs median_owd_cfr_over_100_cases_only',
                          xlabel='person_days_with_vax_per_100',
                          ylabel='median_owd_cfr_over_100_cases_only'))

    for i, txt in enumerate(aggr['location']):
        ax.annotate(txt, (X[i], y[i]))

    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()  # Ordinary Least Squares regression
    trend = model.predict(X_)
    ax.plot(X, trend)

    plt.gcf().set_size_inches(15, 7)

    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.svg")
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.png")
    plt.show()

