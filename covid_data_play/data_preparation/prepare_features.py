from covid_data_play.common import *

if __name__ == '__main__':
    preventable_deaths = read_file('preventable_deaths')

    locations_to_process = COUNTRIES_WITH_GOOD_ENOUGH_DATA

    print(locations_to_process)

    locations = read_file('locations')[['location', 'population']]
    preventable_deaths = pd.read_csv(data_file_path('preventable_deaths'))

    biweekly_cases = read_file('biweekly_cases')
    biweekly_cases_flattened = biweekly_cases \
        .melt(id_vars=['date'], var_name='location', value_name='biweekly_cases',
              value_vars=locations_to_process)

    biweekly_deaths = read_file('biweekly_deaths')
    biweekly_deaths_flattened = biweekly_deaths \
        .melt(id_vars=['date'], var_name='location', value_name='biweekly_deaths',
              value_vars=locations_to_process)

    # Full Data (full_data.csv) from https://github.com/owid/covid-19-data/tree/master/public/data
    # (public/data/archived/ecdc and public/data/archived/who) ends on 2020-11-29, no other full data file found.
    # The full time range can be found in weekly and bi-weekly cases and deaths data sets, so we combine these:
    # full_data = read_file('full_data')
    full_data = pd.merge(biweekly_cases_flattened, biweekly_deaths_flattened, how='inner', on=['location', 'date'])

    full_data = full_data[full_data['location'].isin(locations_to_process)]
    full_data.fillna(value=0.0, inplace=True)

    full_data['year_week'] = full_data['date'].apply(str_date_to_year_week)
    full_data['month'] = full_data['date'].apply(str_date_to_month)
    full_data['week_day'] = full_data['date'].apply(week_day)

    full_data = pd.merge(full_data, locations, how='left', on=['location'])
    full_data = pd.merge(full_data, preventable_deaths, how='left', on=['location'])

    obese_adults_2016 = read_file('share-of-adults-defined-as-obese_2016')[
        ['location', 'prevalence_of_obesity_both_sexes']]
    overweight_adults_2016 = read_file('share-of-adults-who-are-overweight_2016')[
        ['location', 'prevalence_of_overweight_adults_both_sexes']]
    full_data = pd.merge(full_data, obese_adults_2016, how='left', on=['location'])
    full_data = pd.merge(full_data, overweight_adults_2016, how='left', on=['location'])

    # print(f"full_data['location'].unique().size {full_data['location'].unique().size}")

    vac_to_concat = []
    vaccinations = read_file("vaccinations")[
        ['location', 'date', 'total_vaccinations_per_hundred', 'total_boosters_per_hundred']]
    for location in locations_to_process:
        vac_for_loc = vaccinations[vaccinations['location'] == location].loc[:]
        vac_for_loc['total_vaccinations_per_hundred'].interpolate(method='linear', limit_direction='both', inplace=True)
        vac_for_loc['total_boosters_per_hundred'].interpolate(method='linear', limit_direction='both', inplace=True)
        vac_to_concat.append(vac_for_loc)
    vaccinations = pd.concat(vac_to_concat)

    full_data = pd.merge(full_data, vaccinations, how='left', on=['date', 'location'])

    largest_cities = read_file('largest_cities')
    sunshine_hours = read_file('sunshine_hours') \
        .melt(id_vars=['location', 'city'], var_name='month', value_name='monthly_sunshine_hours',
              value_vars=[str(n + 1) for n in range(12)])
    sunshine_hours = sunshine_hours[sunshine_hours['city'].isin(largest_cities['largest_city'])][
        ['location', 'month', 'monthly_sunshine_hours']]

    average_temperature = read_file('average_temperature') \
        .melt(id_vars=['location', 'city'], var_name='month', value_name='monthly_average_temperature',
              value_vars=[str(n + 1) for n in range(12)])
    average_temperature = average_temperature[average_temperature['city'].isin(largest_cities['largest_city'])][
        ['location', 'month', 'monthly_average_temperature']]

    average_temperature["monthly_average_temperature"] = pd.to_numeric(
        average_temperature["monthly_average_temperature"])
    sunshine_hours["monthly_sunshine_hours"] = pd.to_numeric(sunshine_hours["monthly_sunshine_hours"])
    average_temperature["month"] = pd.to_numeric(average_temperature["month"])
    sunshine_hours["month"] = pd.to_numeric(sunshine_hours["month"])

    full_data = pd.merge(full_data, largest_cities, how='left', on=['location'])
    full_data = pd.merge(full_data, average_temperature, how='left', on=['location', 'month'])
    full_data = pd.merge(full_data, sunshine_hours, how='left', on=['location', 'month'])

    # Diabetes
    diabetes_prevalence = read_file('diabetes-prevalence-2019')[['location', 'diabetes_prevalence_per_100_ages_20to79']]
    full_data = pd.merge(full_data, diabetes_prevalence, how='left', on=['location'])

    # Median Age
    median_age = read_file('median-age-2020-prediction')[['location', 'median_age']]
    full_data = pd.merge(full_data, median_age, how='left', on=['location'])

    # omicron %
    v = read_file('variants')
    variants = v[v['variant'] == 'B.1.1.529'][['location', 'year_week', 'percent_variant']]
    variants[['percent_variant']] = \
        variants[['percent_variant']].fillna(0.0)
    full_data = pd.merge(full_data, variants, how='left', on=['year_week', 'location'])
    full_data.rename({'percent_variant': 'percent_omicron'}, axis='columns', inplace=True)

    # age-standardized-death-rate-cardiovascular-disease-annual-per_100k
    cardiovascular_disease = \
        read_file('age-standardized-death-rate-cardiovascular-disease-annual-per_100k')[['location', 'age_standardized_death_rate_cardiovascular']]
    full_data = pd.merge(full_data, cardiovascular_disease, how='left', on=['location'])

    # Demential
    dementia_death_rates = read_file('dementia-death-rates')[['location', 'dementia_death_rate_per_100k']]
    full_data = pd.merge(full_data, dementia_death_rates, how='left', on=['location'])

    # mental health & substance use
    mental = read_file('death-rates-from-mental-health-and-substance-use-disorders')[['location', 'mental_health_and_substance_use_deaths']]
    full_data = pd.merge(full_data, mental, how='left', on=['location'])

    # strokes
    strokes = read_file('stroke-death-rates')[['location', 'stroke_deaths_per_100k']]
    full_data = pd.merge(full_data, strokes, how='left', on=['location'])

    # Population ages 65 and above (% of total population)
    popover65 = read_file('worlda-bank-pop-over-65-percentage')[['location', 'pop_over_65_per_100_2019']]
    full_data = pd.merge(full_data, popover65, how='left', on=['location'])

    # Hypertension
    hypertension = read_file('hypertension')[['location', 'hypertension_prevalence_men', 'hypertension_prevalence_women']]
    full_data = pd.merge(full_data, hypertension, how='left', on=['location'])
    full_data['hypertension_prevalence'] = (full_data['hypertension_prevalence_men'] + full_data['hypertension_prevalence_women']) * 0.5

    # all top causes
    all_top_causes = read_file('probability-of-death-from-any-of-the-top-global-causes-of-death')[['location',
                                                      'prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory_female',
                                                      'prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory_male',
                                                      'prob_of_dying_30_70_from_any_of_cardiovascular_cancer_diabetes_chronic_respiratory']]
    full_data = pd.merge(full_data, all_top_causes, how='left', on=['location'])

    # official Our World In Data CFR
    owd_cfr = read_file('covid-cfr-exemplars')[['date', 'location', 'owd_cfr_over_100_cases_only']]
    full_data = pd.merge(full_data, owd_cfr, how='left', on=['date', 'location'])

    full_data['date'] = pd.to_datetime(full_data['date'], format='%Y-%m-%d')
    full_data.sort_values(by=['location', 'date'], inplace=True)

    full_data['total_vaccinations_per_hundred'] = full_data['total_vaccinations_per_hundred'].fillna(0)
    full_data['total_boosters_per_hundred'] = full_data['total_boosters_per_hundred'].fillna(0)

    full_data['biweekly_deaths_per_100k'] = full_data['biweekly_deaths'] / full_data['population'] * 100000.0
    full_data['biweekly_cases_per_100k'] = full_data['biweekly_cases'] / full_data['population'] * 100000.0

    full_data['cfr'] = full_data['biweekly_deaths'] / full_data['biweekly_cases'].shift(15) * 100

    pd.set_option('use_inf_as_na', True)
    full_data['cfr'] = full_data['cfr'].fillna(0)

    full_data['daily_deaths_per_100k_biweekly_avg'] = full_data['biweekly_deaths_per_100k'] / 14.0
    full_data['daily_cases_per_100k_biweekly_avg'] = full_data['biweekly_cases_per_100k'] / 14.0

    full_data.to_csv(FEATURES_CSV_PATH)
    full_data.to_pickle(FEATURES_PLK_PATH)

    mondays_only = full_data[full_data['week_day'] == 0]
    mondays_only.to_csv(f"{OUT_DIR}/features_mondays.csv")

