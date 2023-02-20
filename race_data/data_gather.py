import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import dateutil.relativedelta as rd

# Get all of the data
def get_all_data(file_or_online):
    pass

# Gets data from a certain year and race index
def get_data_from_idx(races, year, idx, drop_quals=True):
    races = races[races['season'] == year]
    races = races[races['round'] == idx]

    if drop_quals and 'qualifying_time' in races.columns:
        races.drop(labels=['qualifying_time', 'grid'], axis=1, inplace=True)

    return races

# Gets data from before a certain year and race index
def get_data_before_idx(races, year, idx, drop_quals=True):
    if drop_quals and 'qualifying_time' in races.columns:
        races.drop(labels=['qualifying_time', 'grid'], axis=1, inplace=True)

    races_prior_years = races[races['season'] < year]
    races_this_year = races[races['season'] == year]
    races_this_year = races_this_year[races_this_year['round'] < idx]

    return pd.concat([races_prior_years, races_this_year])

# Splits a dataset into inputs and outputs
def split_input_output(dataset):
    pass

# Not entirely sure what this function does
def lookup (df, team, points):
    df['lookup1'] = df.season.astype(str) + df[team] + df['round'].astype(str)
    df['lookup2'] = df.season.astype(str) + df[team] + (df['round']-1).astype(str)
    new_df = df.merge(df[['lookup1', points]], how = 'left', left_on='lookup2',right_on='lookup1')
    new_df.drop(['lookup1_x', 'lookup2', 'lookup1_y'], axis = 1, inplace = True)
    new_df.rename(columns = {points+'_x': points+'_after_race', points+'_y': points}, inplace = True)
    new_df[points].fillna(0, inplace = True)
    return new_df

# Calculate difference in qualifying times
def convert_time(time):
    if str(time) == '00.000':
        return 0
    elif time == 0:
        return 0
    else:
        split_time = str(time).split(':')
        if len(split_time) == 2:
            return float(split_time[1]) + (60 * float(split_time[0]))
        else:
            return float(split_time[0])

# Creates an embedded list of lists with the rounds of each year's season
def compose_race_rounds(races):
    rounds = {}
    for year in np.array(races.season.unique()):
        rounds[year] = list(races[races.season == year]['round'].unique())
    return rounds

# Gets the race data
def get_race_data(scrape_or_file, start_year=1950, end_year=2022):

    if scrape_or_file == "file":
        races = pd.read_csv('races.csv')
        return races[races['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":
        races = {'season': [],
                'round': [],
                'circuit_id': [],
                'lat': [],
                'long': [],
                'country': [],
                'date': [],
                'url': []}

        for year in list(range(start_year, end_year+1)):
            url = 'https://ergast.com/api/f1/{}.json'
            r = requests.get(url.format(year))
            json = r.json()

            for item in json['MRData']['RaceTable']['Races']:
                try:
                    races['season'].append(int(item['season']))
                except:
                    races['season'].append(None)

                try:
                    races['round'].append(int(item['round']))
                except:
                    races['round'].append(None)

                try:
                    races['circuit_id'].append(item['Circuit']['circuitId'])
                except:
                    races['circuit_id'].append(None)

                try:
                    races['lat'].append(float(item['Circuit']['Location']['lat']))
                except:
                    races['lat'].append(None)

                try:
                    races['long'].append(float(item['Circuit']['Location']['long']))
                except:
                    races['long'].append(None)

                try:
                    races['country'].append(item['Circuit']['Location']['country'])
                except:
                    races['country'].append(None)

                try:
                    races['date'].append(item['date'])
                except:
                    races['date'].append(None)

                try:
                    races['url'].append(item['url'])
                except:
                    races['url'].append(None)
        return pd.DataFrame(races)

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_race_results(scrape_or_file, races):

    if scrape_or_file == "file":
        results = pd.read_csv('results.csv')

        start_year = races['season'].min()
        end_year = races['season'].max()

        return results[results['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":

        results = {'season': [],
          'round':[],
           'circuit_id':[],
          'driver': [],
           'date_of_birth': [],
           'nationality': [],
          'constructor': [],
          'grid': [],
          'time': [],
          'status': [],
          'points': [],
          'podium': [],
          'url': []}

        rounds = compose_race_rounds(races)

        for n in list(range(len(rounds.keys()))):
            for i in rounds[n][1]:
        
                url = 'http://ergast.com/api/f1/{}/{}/results.json'
                r = requests.get(url.format(rounds[n][0], i))
                json = r.json()

                for item in json['MRData']['RaceTable']['Races'][0]['Results']:
                    try:
                        results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
                    except:
                        results['season'].append(None)

                    try:
                        results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
                    except:
                        results['round'].append(None)

                    try:
                        results['circuit_id'].append(json['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId'])
                    except:
                        results['circuit_id'].append(None)

                    try:
                        results['driver'].append(item['Driver']['driverId'])
                    except:
                        results['driver'].append(None)
                    
                    try:
                        results['date_of_birth'].append(item['Driver']['dateOfBirth'])
                    except:
                        results['date_of_birth'].append(None)
                        
                    try:
                        results['nationality'].append(item['Driver']['nationality'])
                    except:
                        results['nationality'].append(None)

                    try:
                        results['constructor'].append(item['Constructor']['constructorId'])
                    except:
                        results['constructor'].append(None)

                    try:
                        results['grid'].append(int(item['grid']))
                    except:
                        results['grid'].append(None)

                    try:
                        results['time'].append(int(item['Time']['millis']))
                    except:
                        results['time'].append(None)

                    try:
                        results['status'].append(item['status'])
                    except:
                        results['status'].append(None)

                    try:
                        results['points'].append(int(item['points']))
                    except:
                        results['points'].append(None)

                    try:
                        results['podium'].append(int(item['position']))
                    except:
                        results['podium'].append(None)

                    try:
                        results['url'].append(json['MRData']['RaceTable']['Races'][0]['url'])
                    except:
                        results['url'].append(None)

        return pd.DataFrame(results)

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_driver_standings(scrape_or_file, races):

    if scrape_or_file == "file":
        results = pd.read_csv('driver_standings.csv')

        start_year = races['season'].min()
        end_year = races['season'].max()

        return results[results['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":

        driver_standings = {'season': [],
                        'round':[],
                        'driver': [],
                        'driver_points': [],
                        'driver_wins': [],
                    'driver_standings_pos': []}

        rounds = compose_race_rounds(races)

        for n in list(range(len(rounds.keys()))):
            for i in rounds[n][1]:
            
                url = 'https://ergast.com/api/f1/{}/{}/driverStandings.json'
                r = requests.get(url.format(rounds[n][0], i))
                json = r.json()

                for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
                    try:
                        driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
                    except:
                        driver_standings['season'].append(None)

                    try:
                        driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
                    except:
                        driver_standings['round'].append(None)
                                                
                    try:
                        driver_standings['driver'].append(item['Driver']['driverId'])
                    except:
                        driver_standings['driver'].append(None)
                    
                    try:
                        driver_standings['driver_points'].append(int(item['points']))
                    except:
                        driver_standings['driver_points'].append(None)
                    
                    try:
                        driver_standings['driver_wins'].append(int(item['wins']))
                    except:
                        driver_standings['driver_wins'].append(None)
                        
                    try:
                        driver_standings['driver_standings_pos'].append(int(item['position']))
                    except:
                        driver_standings['driver_standings_pos'].append(None)
                    
        return pd.DataFrame(driver_standings)
    
    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_constructor_standings(scrape_or_file, races):

    if scrape_or_file == "file":
        results = pd.read_csv('constructor_standings.csv')

        start_year = races['season'].min()
        end_year = races['season'].max()

        return results[results['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":

        constructor_standings = {'season': [],
                            'round':[],
                            'constructor': [],
                            'constructor_points': [],
                            'constructor_wins': [],
                        'constructor_standings_pos': []}

        constructor_rounds = compose_race_rounds(races)

        # Delete indices for which there is no constructor data?
        for i in range(1950, 1958):
            del constructor_rounds[i]

        for n in list(range(len(constructor_rounds))):
            for i in constructor_rounds[n][1]:
            
                url = 'https://ergast.com/api/f1/{}/{}/constructorStandings.json'
                r = requests.get(url.format(constructor_rounds[n][0], i))
                json = r.json()

                for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
                    try:
                        constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
                    except:
                        constructor_standings['season'].append(None)

                    try:
                        constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
                    except:
                        constructor_standings['round'].append(None)
                                                
                    try:
                        constructor_standings['constructor'].append(item['Constructor']['constructorId'])
                    except:
                        constructor_standings['constructor'].append(None)
                    
                    try:
                        constructor_standings['constructor_points'].append(int(item['points']))
                    except:
                        constructor_standings['constructor_points'].append(None)

                    try:
                        constructor_standings['constructor_wins'].append(int(item['wins']))
                    except:
                        constructor_standings['constructor_wins'].append(None)
                        
                    try:
                        constructor_standings['constructor_standings_pos'].append(int(item['position']))
                    except:
                        constructor_standings['constructor_standings_pos'].append(None)
                    
        return pd.DataFrame(constructor_standings)

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_qualifying_results(scrape_or_file, races):

    start_year = races['season'].min()
    end_year = races['season'].max()

    if scrape_or_file == "file":
        results = pd.read_csv('qualifying.csv')
        return results[results['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":

        qualifying_results = pd.DataFrame()
        for year in list(range(start_year,end_year+1)):
            url = 'https://www.formula1.com/en/results.html/{}/races.html'
            r = requests.get(url.format(year))
            soup = BeautifulSoup(r.text, 'html.parser')
            
            year_links = []
            for page in soup.find_all('a', attrs = {'class':"resultsarchive-filter-item-link FilterTrigger"}):
                link = page.get('href')
                if f'/en/results.html/{year}/races/' in link: 
                    year_links.append(link)

            year_df = pd.DataFrame()
            new_url = 'https://www.formula1.com{}'
            for n, link in list(enumerate(year_links)):
                link = link.replace('race-result.html', 'starting-grid.html')
                try:
                    this_link = new_url.format(link)
                    df = pd.read_html(new_url.format(link))
                    df = df[0]
                    df['season'] = year
                    df['round'] = n+1
                    for col in df:
                        if 'Unnamed' in col:
                            df.drop(col, axis = 1, inplace = True)

                    year_df = pd.concat([year_df, df])
                except:
                    print(link)

            qualifying_results = pd.concat([qualifying_results, year_df])

        qualifying_results.rename(columns = {'Pos': 'grid_position', 'Driver': 'driver_name', 'Car': 'car',
                                     'Time': 'qualifying_time'}, inplace = True)

        qualifying_results.drop('No', axis = 1, inplace = True)
            
        return qualifying_results

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_weather(scrape_or_file, races):

    start_year = races['season'].min()
    end_year = races['season'].max()

    if scrape_or_file == "file":
        results = pd.read_csv('weather.csv')
        return results[results['season'].between(start_year, end_year)]

    elif scrape_or_file == "scrape":

        weather = races.iloc[:,[0,1,2]]

        info = []

        for link in races.url:
            try:
                df = pd.read_html(link)[0]
                if 'Weather' in list(df.iloc[:,0]):
                    n = list(df.iloc[:,0]).index('Weather')
                    info.append(df.iloc[n,1])
                else:
                    df = pd.read_html(link)[1]
                    if 'Weather' in list(df.iloc[:,0]):
                        n = list(df.iloc[:,0]).index('Weather')
                        info.append(df.iloc[n,1])
                    else:
                        df = pd.read_html(link)[2]
                        if 'Weather' in list(df.iloc[:,0]):
                            n = list(df.iloc[:,0]).index('Weather')
                            info.append(df.iloc[n,1])
                        else:
                            df = pd.read_html(link)[3]
                            if 'Weather' in list(df.iloc[:,0]):
                                n = list(df.iloc[:,0]).index('Weather')
                                info.append(df.iloc[n,1])
                            else:
                                driver = webdriver.Chrome()
                                driver.get(link)

                                # click language button
                                button = driver.find_element_by_link_text('Italiano')
                                button.click()
                                
                                clima = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr[9]/td').text
                                info.append(clima) 
                                        
            except:
                info.append('not found')

        weather['weather'] = info

        weather_dict = {'weather_warm': ['soleggiato', 'clear', 'warm', 'hot', 'sunny', 'fine', 'mild', 'sereno'],
               'weather_cold': ['cold', 'fresh', 'chilly', 'cool'],
               'weather_dry': ['dry', 'asciutto'],
               'weather_wet': ['showers', 'wet', 'rain', 'pioggia', 'damp', 'thunderstorms', 'rainy'],
               'weather_cloudy': ['overcast', 'nuvoloso', 'clouds', 'cloudy', 'grey', 'coperto']}

        weather_df = pd.DataFrame(columns = weather_dict.keys())

        for col in weather_df:
            weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)

        return pd.concat([weather, weather_df], axis = 1)

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")

def get_merged_data(scrape_or_file, races=None, results=None, qualifying=None, driver_standings=None, constructor_standings=None, weather=None):

    if scrape_or_file == "file":
        df = pd.read_csv('final_df.csv')
        return df.drop(labels='Unnamed: 0', axis=1)

    elif scrape_or_file == "scrape":
        qualifying.rename(columns = {'grid_position': 'grid'}, inplace = True)
        driver_standings.drop(['driver_points_after_race', 'driver_wins_after_race', 'driver_standings_pos_after_race'] ,axis = 1, inplace = True)
        constructor_standings.drop(['constructor_points_after_race', 'constructor_wins_after_race','constructor_standings_pos_after_race' ],axis = 1, inplace = True)

        df1 = pd.merge(races, weather, how='inner', on=['season', 'round', 'circuit_id']).drop(['lat', 'long','country','weather'], axis = 1)
        df2 = pd.merge(df1, results, how='inner', on=['season', 'round', 'circuit_id', 'url']).drop(['url','points', 'status', 'time'], axis = 1)

        df3 = pd.merge(df2, driver_standings, how='left', on=['season', 'round', 'driver']) 
        df4 = pd.merge(df3, constructor_standings, how='left', on=['season', 'round', 'constructor']) #from 1958

        final_df = pd.merge(df4, qualifying, how='inner', on=['season', 'round', 'grid']).drop(['driver_name', 'car'], axis = 1) #from 1983

        # Calculate age of drivers
        final_df['date'] = pd.to_datetime(final_df.date)
        final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
        final_df['driver_age'] = final_df.apply(lambda x: rd.relativedelta(x['date'], x['date_of_birth']).years, axis=1)
        final_df.drop(['date', 'date_of_birth'], axis = 1, inplace = True)

        for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 
            'constructor_wins' , 'constructor_standings_pos']:
            final_df[col].fillna(0, inplace = True)
            final_df[col] = final_df[col].map(lambda x: int(x))
            
        final_df.dropna(inplace = True )

        for col in ['weather_warm', 'weather_cold','weather_dry', 'weather_wet', 'weather_cloudy']:
            final_df[col] = final_df[col].map(lambda x: bool(x))

        final_df['qualifying_time'] = final_df.qualifying_time.map(convert_time)
        final_df = final_df[final_df['qualifying_time'] != 0]
        final_df.sort_values(['season', 'round', 'grid'], inplace = True)
        final_df['qualifying_time_diff'] = final_df.groupby(['season', 'round']).qualifying_time.diff()
        final_df['qualifying_time'] = final_df.groupby(['season', 'round']).qualifying_time_diff.cumsum().fillna(0)
        final_df.drop('qualifying_time_diff', axis = 1, inplace = True)

        to_onehot = ['circuit_id', 'nationality', 'constructor', 'driver']
        df_dum = pd.get_dummies(final_df, columns=to_onehot)

        '''
        for col in df_dum.columns:
            if 'nationality' in col and df_dum[col].sum() < 140:
                df_dum.drop(col, axis = 1, inplace = True)
                
            elif 'constructor' in col and df_dum[col].sum() < 140:
                df_dum.drop(col, axis = 1, inplace = True)
                
            elif 'circuit_id' in col and df_dum[col].sum() < 70:
                df_dum.drop(col, axis = 1, inplace = True)
            else:
                pass
        '''
                
        final_df = pd.merge(final_df, df_dum)

        final_df.drop(labels=to_onehot, inplace=True, axis=1)

        return final_df

    else:
        raise Exception("Input scrape_or_file must be either 'scrape' or 'file'")