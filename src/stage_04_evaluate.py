from src.utils.all_utils import create_directory, read_yaml, save_reports
import argparse, os
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

def evalute_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def adjusted_r2(x, y, model):
     r2 = model.score(x, y)
     n = x.shape[0]
     p = x.shape[1]
     adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
     return adj_r2

def evaluate(config_path, params_path):
    params = read_yaml(params_path)
    config = read_yaml(config_path)

    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']

    test_data_filename = config['artifacts']['test']
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)
    
    test_data = pd.read_csv(test_data_path)

    test_x = test_data.drop('quality', axis=1)
    test_y = test_data['quality']

    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_file']
    model_path = os.path.join(artifacts_dir, model_dir, model_filename)

    lr = joblib.load(model_path)

    predicted_values = lr.predict(test_x)
    rmse, mae, r2 = evalute_metrics(test_y, predicted_values)
    adj_r2 = adjusted_r2(test_x, test_y, lr)

    scores_dir = config['artifacts']['reports_dir']
    scores_filename = config['artifacts']['scores']

    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_directory([scores_dir_path])
    
    scores_filepath = os.path.join(scores_dir_path, scores_filename)
    scores = {
        'rmse':rmse,
        'mae':mae,
        'r2':r2,
        'adj_r2':adj_r2
    }
    save_reports(scores, scores_filepath)

if __name__=="__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    evaluate(config_path=parsed_args.config, params_path=parsed_args.params)