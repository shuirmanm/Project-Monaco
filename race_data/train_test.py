import data_gather
import numpy as np
import math
import torch
import pandas as pd

# Steps through each race starting at start_year and ending with
# end_year, testing on that race and training on all prior races
def test_year_range(model, start_year, end_year):

    races = data_gather.get_merged_data('file')
    rounds = data_gather.compose_race_rounds(races)

    num_epochs = 150

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
            filename = 'preds2/year_' + str(i) + '_race_' + str(j) + '.csv'
            save_420_pred(drivers_ordered, preds, filename)

            print("Year = %d | Race Idx = %d | Accuracy = %.2f"%(i, j, accuracy))

            model.reset()

            num_epochs = 10


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
    normalizing_values = np.sum(preds, axis=1)
    preds = preds / normalizing_values[:,None]
    for i in range(20):
        data[i+1] = preds[:,i]
    
    data.to_csv(filename)

def prob_within_one(actual, pred):
    return prob_within_range(1, actual, pred)

def prob_within_two(actual, pred):
    return prob_within_range(2, actual, pred)

# Finds probability that the predicted value was within a certain
# range of the correct value
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
    'driver_mazepin' : 'Nikita Mazepin'
}