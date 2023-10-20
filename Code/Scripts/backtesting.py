import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
pd.set_option('display.max_columns', None)

#raceId_dict
raceId_dict = {}

raceId_dict['2022'] = {'Singapore': 1091,
              'UnitedStates': 1093,
              'Canadian': 1082,
              'Brazilian': 1095,
              'Miami': 1078,
              'Azerbaijan': 1081,
              'British': 1083,
              'Spanish': 1079,
              'Australian': 1076,
              'Hungarian': 1086,
              'Bahrain': 1074,
              'Italian': 1089,
              'Dutch': 1088,
              'Japanese': 1092,
              'SaudiArabian': 1075,
              'Austrian': 1084,
              'Monaco': 1080,
              'AbuDhabi': 1096,
              'Belgian': 1087,
              'MexicoCity': 1094,
              'EmiliaRomagna': 1077,
              'French': 1085}


raceId_dict['2021'] = {'Portuguese': 1054,
              'Styrian': 1058,
              'Austrian': 1060,
              'Brazilian': 1071,
              'Azerbaijan': 1057,
              'British': 1061,
              'Spanish': 1055,
              'Hungarian': 1062,
              'Bahrain': 1052,
              'Italian': 1065,
              'Dutch': 1064,
              'SaudiArabian': 1072,
              'AbuDhabi': 1073,
              'Monaco': 1056,
              'MexicoCity': 1070,
              'Belgian': 1063,
              'MexicoCity': 1070,
              'EmiliaRomagna': 1053,
              'French': 1059,
              'Russian': 1066,
              'Turkish': 1067,
              'UnitedStates': 1069,
              'Qatar': 1038}


raceId_dict['2020'] = {'Portuguese': 1042,
              #'Styrian': 1032, We don't have odds for this one
              'Austrian': 1031,
              'British': 1034,
              'Spanish': 1036,
              'Hungarian': 1033,
              'Bahrain': 1045,
              'Italian': 1038,
              'AbuDhabi': 1047,
              'Belgian': 1037,
              'EmiliaRomagna': 1043,
              'Russian': 1040,
              'Eifel': 1041,
              #'Tuscan': 1039, We don't have odds for this one
              'Turkish': 1044,
              'Sakhir': 1046,
              '70thAnniversary': 1035}

driverId_dict = {
      'Lewis Hamilton': 1,
     'Fernando Alonso': 4,
    'Sebastian Vettel': 20,
        'Pierre Gasly': 842,
    'Daniel Ricciardo': 817,
     'Valtteri Bottas': 822,
     'Kevin Magnussen': 825,
      'Max Verstappen': 830,
        'Carlos Sainz': 832,
        'Esteban Ocon': 839,
        'Lance Stroll': 840,
     'Charles Leclerc': 844,
        'Lando Norris': 846,
      'George Russell': 847,
     'Nicholas Latifi': 849,
        'Yuki Tsunoda': 852,
     'Mick Schumacher': 854,
         'Zhou Guanyu': 855,
     'Alexander Albon': 848,
        'Sergio Perez': 815,
     'Nico Hulkenberg': 807,
  'Antonio Giovinazzi': 841,
      'Kimi Raikkonen': 8,
      'Nikita Mazepin': 853,
       'Robert Kubica': 9,
        'Daniil Kvyat': 826,
     'Romain Grosjean': 154,
   'Pietro Fittipaldi': 850,
         'Jack Aitken': 851

}

#StrategyDict
StrategyDict = {

'SingleUnit':{
    'StrategyName':'SingleUnit',
    'UseKellyCriterion':0,
    'KellyCriterionWeighting':1
},

'Kelly1Percent':{
    'StrategyName':'Kelly1Percent',
    'UseKellyCriterion':1,
    'KellyCriterionWeighting':.01
},

'Kelly5Percent':{
    'StrategyName':'Kelly5Percent',
    'UseKellyCriterion':1,
    'KellyCriterionWeighting':.05
},

}


class Backtest(object):
    def __init__(self, RunName, RunFolder, StartingBankroll):
        self.RunName = RunName
        self.RunFolder = RunFolder
        self.StartingBankroll = StartingBankroll
        self.year_list, self.odds_df_dict, self.races, self.converted_predictions_df_dict = self.DictionaryCreation()

        self.backtesting_run()

    def odds_conversion(self,x):
        if x < 0:
            return (-x) / ((-x) + 100)
        else:
            return (100 / (x + 100))

    def amount_wagered_calc(self,UseKellyCriterion,KellyCriterionWeighting,Bankroll,EstimatedOdds,ImpliedOdds):
        if Bankroll <= 0:
            return 0
        elif UseKellyCriterion == 1:
            ProportionGained = 1/ImpliedOdds
            BankrollPercentage = KellyCriterionWeighting * (EstimatedOdds - (1 - EstimatedOdds)/ProportionGained)
            AmountToBet = BankrollPercentage * Bankroll
            return round(AmountToBet, 0)
        else:
            return 1

    # Function for summarizing the results
    def backtesting_summary(self,strategy):

        df = pd.read_csv('../../Processed Data/Backtesting Results/'+self.RunName+'/'+strategy+'_BackTestingLog.csv',header = 0,sep = ',')

        df1 = df.groupby('Year')['Cumulative bankroll'].last().reset_index()
        df2 = df.groupby('Year')['Cumulative bankroll'].count().reset_index()
        df3 = df.groupby('Year')['Bet outcome'].sum().reset_index()
        df4 = df.groupby('Year')['Amount wagered'].sum().reset_index()
        df5 = df.groupby('Year')['Amount wagered'].mean().reset_index()
        df6 = df.groupby('Year')['Net units won'].sum().reset_index()
        df7 = df.groupby('Year')['Expected value'].mean().reset_index()
        df8 = df.groupby('Year')['Expected value'].min().reset_index()
        df9 = df.groupby('Year')['Expected value'].median().reset_index()
        df10 = df.groupby('Year')['Expected value'].max().reset_index()

        df1.rename(columns = {'Cumulative bankroll':'Ending bankroll'}, inplace = True)
        df2.rename(columns = {'Cumulative bankroll':'Bets placed'}, inplace = True)
        df3.rename(columns = {'Bet outcome':'Bets won'}, inplace = True)
        df5.rename(columns = {'Amount wagered':'Average wager'}, inplace = True)
        df7.rename(columns = {'Expected value':'Mean expected value'}, inplace = True)
        df8.rename(columns = {'Expected value':'Min expected value'}, inplace = True)
        df9.rename(columns = {'Expected value':'Median expected value'}, inplace = True)
        df10.rename(columns = {'Expected value':'Max expected value'}, inplace = True)

        concatenated = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10], axis = "columns")
        concatenated2 = concatenated.T.drop_duplicates().T
        concatenated2["Year"] = concatenated2["Year"].astype(int)

        concatenated2.to_csv('../../Processed Data/Backtesting Results/'+self.RunName+'/Summary/'+strategy+'_Summary.csv', index=False)

        pd.set_option('display.expand_frame_repr', False)
        print(strategy)
        print(concatenated2.head())
        print("\n")

    #Creates a number of dictionaries needed specifically for the backtesting run
    def DictionaryCreation(self):


        # Create a list to loop over for importing the historical odds

        import os

        year_list = ['2020','2021','2022']

        table_dictionary = {}

        for year in year_list:

            path = '../../Raw Data/Odds Data/Historical Odds/'+year
            table_list = []

            for filename in os.listdir(path):

                if filename.endswith('.csv'):
                    table_list.append(filename[:-4])

            table_dictionary[year] = table_list


        # Loops over years to import the historical odds

        odds_df_dict = {}

        for year in table_dictionary:

            odds_df_dict[year] = {}

            for race in table_dictionary[year]:
                df = pd.read_csv('../../Raw Data/Odds Data/Historical Odds/'+year+'/'+race+'.csv',header = 0,sep = '|')
                odds_df_dict[year][race] = df



        # Converting the odds from American odds format to implied probabilities
        # There is also some data cleaning for driver names in this loop
        # Loops over years

        for year in odds_df_dict:

            for race in odds_df_dict[year]:
                # The purpose of the if statements here is that there are several races where the historical odds were just missing a column
                if 'Odds to Win' in odds_df_dict[year][race]:
                    odds_df_dict[year][race]['Odds to Win'] = odds_df_dict[year][race]['Odds to Win'].apply(self.odds_conversion)
                if 'Odds to Finish Top Three' in odds_df_dict[year][race]:
                    odds_df_dict[year][race]['Odds to Finish Top Three'] = odds_df_dict[year][race]['Odds to Finish Top Three'].apply(self.odds_conversion)
                if 'Odds to Finish Top Six' in odds_df_dict[year][race]:
                    odds_df_dict[year][race]['Odds to Finish Top Six'] = odds_df_dict[year][race]['Odds to Finish Top Six'].apply(self.odds_conversion)
                if 'Odds to Finish Top Ten' in odds_df_dict[year][race]:
                    odds_df_dict[year][race]['Odds to Finish Top Ten'] = odds_df_dict[year][race]['Odds to Finish Top Ten'].apply(self.odds_conversion)


                # Below here is data cleaning - making sure the driver name is consistent across files
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Alex Albon','Alexander Albon',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Carlos Sainz Jr.','Carlos Sainz',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Guanyu Zhou','Zhou Guanyu',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Nick Latifi','Nicholas Latifi',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Estaban Ocon','Esteban Ocon',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Daniil Kyvat','Daniil Kvyat',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Kimi Raikonen','Kimi Raikkonen',regex = True)
                odds_df_dict[year][race]['Driver'] = odds_df_dict[year][race]['Driver'].str.replace('Danil Kvyat','Daniil Kvyat',regex = True)



        # Importing race results, the race information, and driver information

        results = pd.read_csv('../../Raw Data/Historical Race Data/1950_to_2022_CSVs/races.csv',header = 0,sep = ',')
        races = pd.read_csv('../../Raw Data/Historical Race Data/1950_to_2022_CSVs/results.csv',header = 0,sep = ',')
        drivers = pd.read_csv('../../Raw Data/Historical Race Data/1950_to_2022_CSVs/drivers.csv',header = 0,sep = ',')

        results_dict = {}

        for year in year_list:
            results_dict[year] = results.loc[results['year'] == int(year)]



        # Creating a dictionary for the driver Ids and racer names

        drivers['combined name'] = drivers['forename'] + ' ' + drivers['surname']
        drivers.head()


        # Reading the probabilities into a dictionary of dataframes

        predictions_df_dict = {}

        for year in year_list:
            predictions_df_dict[year] = {}
            for race in raceId_dict[year]:
                #Warning! Try except is to deal with missing races and should eventually be resolved
                try:
                    df = pd.read_csv('../../Processed Data/Probability Outputs/'+self.RunFolder+'/'+year+'/'+race+'.csv',header = 0,sep = ',')
                    predictions_df_dict[year][race] = df
                except:
                    continue


        # Creating a dictionary of converted predictions
        # Transforming the even weighting dummy file so that it contains odds for 'Odds to Win', 'Odds to Finish Top Three',
        # 'Odds to Finish Top Six', and 'Odds to Finish Top Ten'

        converted_predictions_df_dict = {}

        for year in year_list:

            converted_predictions_df_dict[year] = {}

            for race in raceId_dict[year]:

                converted_predictions_df_dict[year][race] = pd.DataFrame(columns=['Driver','Probability of Winning',
                                                                'Probability of Finishing Top Three',
                                                                'Probability of Finishing Top Six',
                                                                'Probability of Finishing Top Ten'])

                #Warning! Try except is to deal with missing races and should eventually be resolved

                try:
                    converted_predictions_df_dict[year][race]['Driver'] = predictions_df_dict[year][race]['Driver']
                    converted_predictions_df_dict[year][race]['Probability of Winning'] = predictions_df_dict[year][race]['1']
                    converted_predictions_df_dict[year][race]['Probability of Finishing Top Three'] = predictions_df_dict[year][race]['1'] + predictions_df_dict[year][race]['2'] + predictions_df_dict[year][race]['3']
                    converted_predictions_df_dict[year][race]['Probability of Finishing Top Six'] = predictions_df_dict[year][race]['1'] + predictions_df_dict[year][race]['2'] + predictions_df_dict[year][race]['3'] + predictions_df_dict[year][race]['4'] + predictions_df_dict[year][race]['5'] + predictions_df_dict[year][race]['6']
                    converted_predictions_df_dict[year][race]['Probability of Finishing Top Ten'] = predictions_df_dict[year][race]['1'] + predictions_df_dict[year][race]['2'] + predictions_df_dict[year][race]['3'] + predictions_df_dict[year][race]['4'] + predictions_df_dict[year][race]['5'] + predictions_df_dict[year][race]['6']+ predictions_df_dict[year][race]['7'] + predictions_df_dict[year][race]['8'] + predictions_df_dict[year][race]['9'] + predictions_df_dict[year][race]['10']
                except:
                    continue


        return year_list, odds_df_dict, races, converted_predictions_df_dict

    def BacktestingFunction(self, StartingBankroll, StrategyName, UseKellyCriterion, KellyCriterionWeighting, RunName):

        Bankroll = StartingBankroll

        BacktestingLog = pd.DataFrame(columns=['Year'
                                           ,'Race'
                                           ,'Driver'
                                           , 'Bet placed'
                                           , 'Driver race outcome'
                                           , 'Implied probability'
                                           , 'Estimated probability'
                                           , 'Expected value'
                                           , 'Bet outcome'
                                           , 'Amount wagered'
                                           , 'Units won'
                                           , 'Net units won'
                                           , 'Cumulative bankroll'])


        # Creating a triple loop over the year, race, and driver using the implied probability dataframes
        # This will perform the backtesting and log the results into a new dataframe


        for year in self.year_list:

            temp = []

            for race in raceId_dict[year]:

                for driver in self.odds_df_dict[year][race]['Driver']:

                    # NOTE: This if statement is for handling two situations where a driver was subbed out last minute
                    # for another driver. Because this is a rare scenario, I thought it was better to handle these manually
                    # rather than trying to program something dynamic
                    if (race == 'Italian' and driver == 'Alexander Albon') or (race == 'SaudiArabian' and driver == 'Sebastian Vettel'):
                        continue


                    DriverOutcome = self.races.loc[((self.races['driverId'] == driverId_dict[driver]) & (self.races['raceId'] == raceId_dict[year][race])),'position']

                    # NOTE: It is likely possible to replace the four 'comparison' sections with a loop but this was not deemed a priority

                    #First comparison - odds to win
                    ImpliedOdds = self.odds_df_dict[year][race].loc[self.odds_df_dict[year][race]['Driver'] == driver,'Odds to Win']
                    EstimatedOdds = self.converted_predictions_df_dict[year][race].loc[self.converted_predictions_df_dict[year][race]['Driver'] == driver,'Probability of Winning']

                    #WARNING: This try except is to handle bugs that should be addressed
                    try:


                        if EstimatedOdds.iloc[0] > ImpliedOdds.iloc[0]:


                            DriverOutcome = self.races.loc[((self.races['driverId'] == driverId_dict[driver])
                                                       & (self.races['raceId'] == raceId_dict[year][race])),'position']

                            BetOutcome = 0
                            UnitsWon = 0

                            AmountWagered = self.amount_wagered_calc(UseKellyCriterion,KellyCriterionWeighting,Bankroll,EstimatedOdds.iloc[0],ImpliedOdds.iloc[0])

                            if DriverOutcome.iloc[0] == '1':
                                BetOutcome = 1
                                UnitsWon = AmountWagered / ImpliedOdds.iloc[0]

                            NetUnitsWon = UnitsWon - AmountWagered

                            Bankroll = Bankroll + NetUnitsWon

                            BacktestingLog = pd.concat([BacktestingLog, pd.DataFrame.from_records([{
                                'Year': year,
                                'Race': race,
                                'Driver': driver,
                                'Bet placed': 'Odds to Win',
                                'Driver race outcome': DriverOutcome.iloc[0],
                                'Implied probability': ImpliedOdds.iloc[0],
                                'Estimated probability': EstimatedOdds.iloc[0],
                                'Expected value': (EstimatedOdds.iloc[0] / ImpliedOdds.iloc[0]) - 1,
                                'Bet outcome': BetOutcome,
                                'Amount wagered': AmountWagered,
                                'Units won': UnitsWon,
                                'Net units won': NetUnitsWon,
                                'Cumulative bankroll': Bankroll
                            }])])

                    except:
                        continue

                    #Second comparison - Odds to Finish Top Three
                    if 'Odds to Finish Top Three' in self.odds_df_dict[year][race].columns:
                        ImpliedOdds = self.odds_df_dict[year][race].loc[self.odds_df_dict[year][race]['Driver'] == driver,'Odds to Finish Top Three']
                        EstimatedOdds = self.converted_predictions_df_dict[year][race].loc[self.converted_predictions_df_dict[year][race]['Driver'] == driver,'Probability of Finishing Top Three']


                        if EstimatedOdds.iloc[0] > ImpliedOdds.iloc[0]:


                            DriverOutcome = self.races.loc[((self.races['driverId'] == driverId_dict[driver])
                                                       & (self.races['raceId'] == raceId_dict[year][race])),'position']

                            BetOutcome = 0
                            UnitsWon = 0

                            AmountWagered = self.amount_wagered_calc(UseKellyCriterion,KellyCriterionWeighting,Bankroll,EstimatedOdds.iloc[0],ImpliedOdds.iloc[0])

                            if DriverOutcome.iloc[0] in ['1',  '2', '3']:
                                BetOutcome = 1
                                UnitsWon = AmountWagered / ImpliedOdds.iloc[0]

                            NetUnitsWon = UnitsWon - AmountWagered

                            Bankroll = Bankroll + NetUnitsWon

                            BacktestingLog = pd.concat([BacktestingLog, pd.DataFrame.from_records([{
                                'Year': year,
                                'Race': race,
                                'Driver': driver,
                                'Bet placed': 'Odds to Finish Top Three',
                                'Driver race outcome': DriverOutcome.iloc[0],
                                'Implied probability': ImpliedOdds.iloc[0],
                                'Estimated probability': EstimatedOdds.iloc[0],
                                'Expected value': (EstimatedOdds.iloc[0] / ImpliedOdds.iloc[0]) - 1,
                                'Bet outcome': BetOutcome,
                                'Amount wagered': AmountWagered,
                                'Units won': UnitsWon,
                                'Net units won': NetUnitsWon,
                                'Cumulative bankroll': Bankroll
                            }])])

                    #Third comparison - Odds to Finish Top Six
                    if 'Odds to Finish Top Six' in self.odds_df_dict[year][race].columns:
                        ImpliedOdds = self.odds_df_dict[year][race].loc[self.odds_df_dict[year][race]['Driver'] == driver,'Odds to Finish Top Six']
                        EstimatedOdds = self.converted_predictions_df_dict[year][race].loc[self.converted_predictions_df_dict[year][race]['Driver'] == driver,'Probability of Finishing Top Six']


                        if EstimatedOdds.iloc[0] > ImpliedOdds.iloc[0]:


                            DriverOutcome = self.races.loc[((self.races['driverId'] == driverId_dict[driver])
                                                       & (self.races['raceId'] == raceId_dict[year][race])),'position']

                            BetOutcome = 0
                            UnitsWon = 0

                            AmountWagered = self.amount_wagered_calc(UseKellyCriterion,KellyCriterionWeighting,Bankroll,EstimatedOdds.iloc[0],ImpliedOdds.iloc[0])

                            if DriverOutcome.iloc[0] in ['1','2','3','4','5','6']:
                                BetOutcome = 1
                                UnitsWon = AmountWagered / ImpliedOdds.iloc[0]

                            NetUnitsWon = UnitsWon - AmountWagered

                            Bankroll = Bankroll + NetUnitsWon

                            BacktestingLog = pd.concat([BacktestingLog, pd.DataFrame.from_records([{
                                'Year': year,
                                'Race': race,
                                'Driver': driver,
                                'Bet placed': 'Odds to Finish Top Six',
                                'Driver race outcome': DriverOutcome.iloc[0],
                                'Implied probability': ImpliedOdds.iloc[0],
                                'Estimated probability': EstimatedOdds.iloc[0],
                                'Expected value': (EstimatedOdds.iloc[0] / ImpliedOdds.iloc[0]) - 1,
                                'Bet outcome': BetOutcome,
                                'Amount wagered': AmountWagered,
                                'Units won': UnitsWon,
                                'Net units won': NetUnitsWon,
                                'Cumulative bankroll': Bankroll
                            }])])


                    #Fourth comparison - Odds to Finish Top Ten


                    if 'Odds to Finish Top Ten' in self.odds_df_dict[year][race].columns:
                        ImpliedOdds = self.odds_df_dict[year][race].loc[self.odds_df_dict[year][race]['Driver'] == driver,'Odds to Finish Top Ten']
                        EstimatedOdds = self.converted_predictions_df_dict[year][race].loc[self.converted_predictions_df_dict[year][race]['Driver'] == driver,'Probability of Finishing Top Ten']


                        if EstimatedOdds.iloc[0] > ImpliedOdds.iloc[0]:


                            DriverOutcome = self.races.loc[((self.races['driverId'] == driverId_dict[driver])
                                                       & (self.races['raceId'] == raceId_dict[year][race])),'position']

                            BetOutcome = 0
                            UnitsWon = 0

                            AmountWagered = self.amount_wagered_calc(UseKellyCriterion,KellyCriterionWeighting,Bankroll,EstimatedOdds.iloc[0],ImpliedOdds.iloc[0])

                            if DriverOutcome.iloc[0] in ['1','2','3','4','5','6','7','8','9','10']:
                                BetOutcome = 1
                                UnitsWon = AmountWagered / ImpliedOdds.iloc[0]

                            NetUnitsWon = UnitsWon - AmountWagered

                            Bankroll = Bankroll + NetUnitsWon

                            BacktestingLog = pd.concat([BacktestingLog, pd.DataFrame.from_records([{
                                'Year': year,
                                'Race': race,
                                'Driver': driver,
                                'Bet placed': 'Odds to Finish Top Ten',
                                'Driver race outcome': DriverOutcome.iloc[0],
                                'Implied probability': ImpliedOdds.iloc[0],
                                'Estimated probability': EstimatedOdds.iloc[0],
                                'Expected value': (EstimatedOdds.iloc[0] / ImpliedOdds.iloc[0]) - 1,
                                'Bet outcome': BetOutcome,
                                'Amount wagered': AmountWagered,
                                'Units won': UnitsWon,
                                'Net units won': NetUnitsWon,
                                'Cumulative bankroll': Bankroll
                            }])])

        BacktestingLog.to_csv('../../Processed Data/Backtesting Results/'+RunName+'/'+StrategyName+'_BackTestingLog.csv', index=False)

    def backtesting_run(self):

        if not os.path.exists('../../Processed Data/Backtesting Results/'+self.RunName):
            os.makedirs('../../Processed Data/Backtesting Results/'+self.RunName)

        if not os.path.exists('../../Processed Data/Backtesting Results/'+self.RunName+'/Summary'):
            os.makedirs('../../Processed Data/Backtesting Results/'+self.RunName+'/Summary')

        for strategy in StrategyDict:
            self.BacktestingFunction(self.StartingBankroll, StrategyDict[strategy]['StrategyName']
                                , StrategyDict[strategy]['UseKellyCriterion']
                                , StrategyDict[strategy]['KellyCriterionWeighting']
                                , self.RunName)
            self.backtesting_summary(StrategyDict[strategy]['StrategyName'])

