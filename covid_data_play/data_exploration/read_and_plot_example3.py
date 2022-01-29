# evaluate random forest algorithm for classification
import os

import matplotlib.pyplot as plt

from covid_data_play.common import *

if __name__ == '__main__':
    full_data = pd.read_pickle(FEATURES_PLK_PATH)

    # full_data = full_data[full_data['date'] >= '2020-04-01']

    countries_to_plot = [
        'United States', 'United Kingdom', 'Turkey', 'Switzerland', 'Sweden', 'Spain', 'Slovenia', 'Slovakia',
        'Portugal', 'Poland'
        # , 'Norway', 'New Zealand', 'Netherlands', 'Mexico', 'Luxembourg', 'Lithuania', 'Latvia', 'South Korea', 'Japan', 'Italy', 'Israel', 'Ireland', 'Iceland', 'Hungary', 'Greece', 'Germany', 'France', 'Finland', 'Estonia', 'Denmark', 'Czechia', 'Costa Rica', 'Colombia', 'Chile', 'Canada', 'Belgium', 'Austria', 'Australia'
    ]
    # ['Slovakia', 'Czechia', 'Portugal']

    fig, axes = plt.subplots(nrows=5, ncols=len(countries_to_plot))

    def plot_smoothed_cfr(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data['cfr'] = data['cfr'].rolling(20).mean()
        data.plot(x='date', y=['cfr'], title=plot_title, ax=axes[subplot_row, subplot_col],
                  ylim=(0, 10))

    def plot_owd_cfr(c, plot_title, subplot_row, subplot_col):
        data = full_data[full_data['location'] == c]
        data.plot(x='date', y=['owd_cfr_over_100_cases_only'], title=plot_title, ax=axes[subplot_row, subplot_col],
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
        plot_smoothed_cfr(cntr, cntr, 0, countries_to_plot.index(cntr))
        plot_owd_cfr(cntr, '', 1, countries_to_plot.index(cntr))
        plot_biweekly_cases(cntr, '', 2, countries_to_plot.index(cntr))
        plot_biweekly_deaths(cntr, '', 3, countries_to_plot.index(cntr))
        plot_total_vax(cntr, '', 4, countries_to_plot.index(cntr))

    # plt.show()
    figure = plt.gcf()
    figure.set_size_inches(len(countries_to_plot) * 5, 8)
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.svg")
    plt.savefig(f"{OUT_DIR}/{os.path.basename(__file__).replace('.py', '')}.pdf")

