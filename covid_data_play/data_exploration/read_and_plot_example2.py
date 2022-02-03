# evaluate random forest algorithm for classification
import os

import matplotlib.pyplot as plt

from covid_data_play.common import *

from covid_data_play.common import FEATURES_PLK_PATH, OUT_DIR, COUNTRIES_WITH_GOOD_ENOUGH_DATA

if __name__ == '__main__':
    full_data = pd.read_pickle(FEATURES_PLK_PATH)

    # full_data = full_data[full_data['date'] >= '2020-04-01']

    countries_to_plot = COUNTRIES_WITH_GOOD_ENOUGH_DATA

    fig, axes = plt.subplots(nrows=len(countries_to_plot), ncols=4)

    def plot_smoothed_cfr(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data['cfr'] = data['owd_cfr_over_100_cases_only']  # data['cfr'].rolling(20).mean()
        data.plot(x='date', y=['cfr'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 10))


    def plot_biweekly_cases(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data.plot(x='date', y=['biweekly_cases_per_100k'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 3000))


    def plot_biweekly_deaths(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data.plot(x='date', y=['biweekly_deaths_per_100k'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 40))


    def plot_total_vax(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data.plot(x='date', y=['total_vaccinations_per_hundred'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 200))
        data.plot(x='date', y=['total_boosters_per_hundred'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 200))


    for cntr in countries_to_plot:
        plot_smoothed_cfr(cntr, cntr, countries_to_plot.index(cntr), 0)
        plot_biweekly_cases(cntr, '', countries_to_plot.index(cntr), 1)
        plot_biweekly_deaths(cntr, '', countries_to_plot.index(cntr), 2)
        plot_total_vax(cntr, '', countries_to_plot.index(cntr), 3)

    # plt.show()
    figure = plt.gcf()
    figure.set_size_inches(30, len(countries_to_plot)*4)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.svg")
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.png")
    #plt.show()
