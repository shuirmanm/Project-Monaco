import data_gather

# Steps through each race starting at start_year and ending with
# end_year, testing on that race and training on all prior races
def test_year_range(model, start_year, end_year):

    races = data_gather.get_merged_data('file')
    rounds = data_gather.compose_race_rounds(races)

    for i in range(start_year, end_year+1):
        for j in rounds[i]:
            train_data = data_gather.get_data_before_idx(races, i, j)
            test_data = data_gather.get_data_from_idx(races, i, j)

            model.train(train_data)
            accuracy = model.test(test_data)

            print("Year = %d | Race Idx = %d | Accuracy = %.2f"%(i, j, accuracy))

            model.reset()

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
