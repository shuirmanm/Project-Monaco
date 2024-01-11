

import data_gather
import numpy as np
import math
import torch
import pandas as pd
import os

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
              'Styrian': 1032,
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
              'Tuscan': 1039,
              'Turkish': 1044,
              'Sakhir': 1046,
              '70thAnniversary': 1035}


# Steps through each race starting at start_year and ending with
# end_year, testing on that race and training on all prior races
def test_year_range(model, start_year, end_year, model_name, initial_epochs):

    races = data_gather.get_merged_data('file')

    if 'driver' in races.columns:
      races.drop(['circuit_id', 'driver', 'nationality', 'constructor'], axis = 1, inplace = True)

    if 'Unnamed: 0' in races.columns:
      races.drop('Unnamed: 0', axis = 1, inplace = True)

    rounds = data_gather.compose_race_rounds(races)

    num_epochs = initial_epochs

    for i in range(start_year, end_year+1):
        for j in rounds[i]:
            train_data = data_gather.get_data_before_idx(races, i, j)
            test_data = data_gather.get_data_from_idx(races, i, j)

            drivers = []
            for col in test_data.columns:
                if 'driver' in col and (not 'wins' in col) and (not 'points' in col) and (not 'standings' in col):
                    drivers.append(col)

            test_data_subset = test_data[drivers]
            drivers_ordered = []
            for _, row in test_data.iterrows():
                for driver_name in drivers:
                    if row[driver_name] == 1:
                        drivers_ordered.append(driver_name)
                        break

            model.train(train_data, num_epochs=num_epochs)

            (accuracy, preds) = model.test(test_data)


            #We are pulling in the raw races table to pull certain values for naming output files
            #There is probably a way to adjust the code upstream so that this table doesn't need to be brought in again
            #But doing it this way was quicker
            races_raw = pd.read_csv('../../Raw Data/Historical Race Data/1950_to_2022_CSVs/races.csv',header = 0,sep = ',')

            ID = races_raw.loc[(races_raw['year'] == int(i)) & (races_raw['round'] == int(j)),'raceId']

            for key, value in raceId_dict[str(i)].items():
                if value == ID.iloc[0]:
                    filename = '../../Processed Data/Probability Outputs/' + model_name + '/' + str(i) + '/' +key+ '.csv'
                    #These two lines of code should go in a loop that is just over years
                    if not os.path.exists('../../Processed Data/Probability Outputs/' + model_name + '/' + str(i)):
                      os.makedirs('../../Processed Data/Probability Outputs/' + model_name + '/' + str(i))


            save_420_pred(drivers_ordered, preds, filename)

            print("Year = %d | Race Idx = %d | Loss = %.2f"%(i, j, accuracy))

            model.reset()

            num_epochs = 5


def save_420_pred(drivers, preds, filename):
    data = pd.DataFrame()
    drivernames = []
    for key in drivers:
        if key in driverKeys.keys():
            ans = driverKeys[key]
        else:
            ans = key
        drivernames.append(ans)
    data['Driver'] = drivernames
    # normalizing_values = np.sum(preds, axis=1)
    # preds = preds / normalizing_values[:,None]
    for i in range(20):
      data[i+1] = preds[:,i]


    data.to_csv(filename)

def prob_within_one(actual, pred):
    return prob_within_range(1, actual, pred)

def prob_within_two(actual, pred):
    return prob_within_range(2, actual, pred)

# Finds probability that the predicted value was within a certain range of the correct value
def prob_within_range(val, actual, pred):
    num = len(actual)
    max_val = pred.shape[1]
    avg_prob = 0
    for i in range(num):
        preds_one = pred[i]
        indices = list(range(actual[i]-val, actual[i]+val + 1))
        indices = [idx for idx in indices if (idx >= 0 and idx < max_val)]
        avg_prob += sum(preds_one[indices])

    return avg_prob / num

def split_df_into_minibatches(df, batch_size):
    num_minibatches = math.floor(len(df) / batch_size)
    num_samples = num_minibatches * batch_size
    indices = np.arange(num_samples)
    #np.random.shuffle(indices)
    indices = indices.reshape((-1,batch_size))

    minibatches = []
    for i in range(num_minibatches):
        minibatch_df = df.iloc[indices[i,:]]
        x = minibatch_df.drop('podium', axis = 1)
        y = minibatch_df['podium'].to_numpy()
        for i in range(len(y)):
            if y[i] > 20:
                y[i] = 20
        x_tensor = torch.tensor(x.values.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.float32))
        minibatches.append([x_tensor, y_tensor])

    return minibatches

driverKeys = {
    'driver_hamilton' : 'Lewis Hamilton',
    'driver_alonso' : 'Fernando Alonso',
    'driver_vettel' : 'Sebastian Vettel',
    'driver_gasly' : 'Pierre Gasly',
    'driver_ricciardo' : 'Daniel Ricciardo',
    'driver_bottas' : 'Valtteri Bottas',
    'driver_magnussen' : 'Kevin Magnussen',
    'driver_max_verstappen' : 'Max Verstappen',
    'driver_sainz' : 'Carlos Sainz',
    'driver_ocon' : 'Esteban Ocon',
    'driver_stroll' : 'Lance Stroll',
    'driver_leclerc' : 'Charles Leclerc',
    'driver_norris' : 'Lando Norris',
    'driver_russell' : 'George Russell',
    'driver_latifi' : 'Nicholas Latifi',
    'driver_tsunoda' : 'Yuki Tsunoda',
    'driver_mick_schumacher' : 'Mick Schumacher',
    'driver_zhou' : 'Zhou Guanyu',
    'driver_albon' : 'Alexander Albon',
    'driver_perez' : 'Sergio Perez',
    'driver_hulkenberg' : 'Nico Hulkenberg',
    'driver_giovinazzi' : 'Antonio Giovinazzi',
    'driver_raikkonen' : 'Kimi Raikkonen',
    'driver_mazepin' : 'Nikita Mazepin',
    'driver_kvyat' : 'Daniil Kvyat',
    'driver_kevin_magnussen' : 'Kevin Magnussen',
    'driver_grosjean' : 'Romain Grosjean',
    'driver_pietro_fittipaldi' : 'Pietro Fittipaldi',

}
