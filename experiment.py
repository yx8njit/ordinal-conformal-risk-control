import datetime
import pandas as pd
import numpy as np
import random
from ordinal_regression_conformal_risk_predictors import WeightedCRPredictor
from ordinal_regression_conformal_risk_predictors import DivergenceCRPredictor

def load_raw_fyxs(filename, num_classes):
    df = pd.read_csv(filename)  
    true_labels = df.iloc[:, 2].to_numpy().astype(int)
    predicted_labels = df.iloc[:, 3].to_numpy().astype(int)
    raw_fyxs = df.iloc[:, 4: 4 + num_classes].to_numpy()
    return true_labels, predicted_labels, raw_fyxs


def convert2softmax(raw_fyxs):
    (data_num, num_classes) = raw_fyxs.shape
    raw_fyxs_min = raw_fyxs.min(axis =1).reshape(data_num,1)
    raw_fyxs_min_tile = np.tile(raw_fyxs_min, [1, num_classes])
    raw_fyxs_zeroed = raw_fyxs - raw_fyxs_min_tile
    raw_fyxs_sum = np.sum(raw_fyxs_zeroed, axis = 1).reshape(data_num,1)
    raw_fyxs_sum_tile = np.tile(raw_fyxs_sum, [1, num_classes])
    raw_fyxs_sum_inv = np.reciprocal(raw_fyxs_sum_tile)
    fyxs_normalized = raw_fyxs_zeroed * raw_fyxs_sum_inv
    return fyxs_normalized


def random_split_data(fyxs, labels, val_ratio = 0.5):
    data_num = len(labels)
    val_data_num = int(data_num * val_ratio)
    
    rand_indices = list(range(data_num))
    random.shuffle(rand_indices)
    
    val_indices = rand_indices[:val_data_num]
    val_data = fyxs[val_indices, :]
    y_val = labels[val_indices]
    
    test_indices = rand_indices[val_data_num:]
    test_data = fyxs[test_indices, :]
    y_test = labels[test_indices]
    
    return val_data, y_val, test_data, y_test

if __name__ == "__main__":
    true_labels, predicted_labels, raw_fyxs = load_raw_fyxs('output.csv', 10)
    print(true_labels[:100])
    print(predicted_labels)

    fyxs_normalized = convert2softmax(raw_fyxs)
    (data_num, num_classes) = fyxs_normalized.shape
    print(sum(fyxs_normalized[3110,:]))
    print(data_num)
    print(num_classes)

    val_data, y_val, test_data, y_test = random_split_data(fyxs_normalized, true_labels, 0.5)
    print(val_data[0])
    print(y_val)
    print(test_data[0])
    print(y_test)
    print(datetime.datetime.now())

    num_classes = 10
    hy = np.ones(num_classes)
    # hy[16] = 2
    # hy[17] = 2
    # hy[18] = 2
    # hy[19] = 2
    predictor = WeightedCRPredictor(hy)
    # predictor = DivergenceCRPredictor()

    num_runs = 100
    loss_mean = {}
    loss_variance = {}
    setsize_mean = {}
    setsize_variance = {}
    alpha_vals = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    # alpha_vals = np.arange(0.02, 0.22, 0.02)
    # alpha_vals = [0.02]
    for alpha in alpha_vals:
        all_losses = []
        all_setsizes = []
        for i in range(num_runs):
            val_data, y_val, test_data, y_test = random_split_data(fyxs_normalized, true_labels, 0.5)
            lambda_val = predictor.find_lambda(val_data, y_val, alpha)
            prediction_sets, cur_losses = predictor.run_predictions(test_data, y_test, lambda_val)
            all_losses.append(np.average(cur_losses))
            cur_setsizes = [p[1] - p[0] + 1 for p in prediction_sets]
            all_setsizes.append(cur_setsizes)
        loss_mean[alpha] = np.average(all_losses)
        loss_variance[alpha] = np.var(all_losses)
        setsize_mean[alpha] = np.average(all_setsizes)
        setsize_variance[alpha] = np.var(all_setsizes)
        print('{}, {}, {}, {}, {}'
              .format(alpha, loss_mean[alpha], loss_variance[alpha], setsize_mean[alpha], setsize_variance[alpha]))

    print(datetime.datetime.now())
