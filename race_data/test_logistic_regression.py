import model
import train_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


base_model = LogisticRegression(max_iter=1000)
test_model = model.Model(base_model, train_test.prob_within_two)

train_test.test_year_range(test_model, 2010, 2020)