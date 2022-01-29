# evaluate random forest algorithm for classification
import matplotlib.pyplot as plt

from common import *

if __name__ == '__main__':
    full_data = pd.read_pickle("../features.pkl")

    # full_data = full_data[full_data['date'] >= '2020-04-01']

    countries_to_plot = ['Slovakia', 'Czechia', 'Poland', 'Germany', 'Portugal']

    def plot_smoothed_cfr(c):
        data = full_data[full_data['location'] == c]
        data['cfr'] = data['cfr'].rolling(20).mean()
        data.plot(x='date', y=[  # 'total_vaccinations_per_hundred',
            # 'biweekly_deaths_per_100k', 'biweekly_cases_per_100k',
            # total_boosters_per_hundred',
            'cfr'], title=c)


    for c in countries_to_plot:
        plot_smoothed_cfr(c)

    plt.show()
