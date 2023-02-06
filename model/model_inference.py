import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import torch

from model.utils import load_model

MODEL_INFERENCE_BEF0RE_PATH = "data/input_data.png"
MODEL_INFERENCE_RESULT_SAVE_PATH = "results/output_result.png"
MODEL_PATH = "model/output/model_output.pth"


class ModelInference():

    def __init__(self, input_data_path):
        self.input_data = input_data_path
        self.model = load_model(MODEL_PATH)

    def model_result(self):
        input_data = pd.read_csv(self.input_data)
        input_data['# Date'] = pd.to_datetime(input_data['# Date'])
        base = max(input_data['# Date']) + timedelta(days=1)
        date_list = [base + timedelta(days=x) for x in range(365)]
        d = {'# Date': date_list, 'year': [i.year for i in date_list], 'month': [i.month for i in date_list],
             'day': [i.day for i in date_list]}
        nxt_yr_df = pd.DataFrame(data=d)
        X_test = nxt_yr_df.iloc[:, -3:].values
        inputs = torch.from_numpy(X_test)

        # Perform forward pass
        inputs = inputs.to(torch.float32)
        outputs = self.model(inputs)
        dates = list(input_data['# Date'])
        counts = list(input_data['Receipt_Count'])
        self.plot_fig(dates, counts, path=MODEL_INFERENCE_BEF0RE_PATH)
        y_future = [float(i[0].data) + (np.min(counts) / 9) for i in outputs]
        dates = list(input_data['# Date'].append(nxt_yr_df['# Date']))
        counts.extend(y_future)
        self.plot_fig(dates, counts, path=MODEL_INFERENCE_RESULT_SAVE_PATH)

    def plot_fig(self, dates, counts, path):
        # Plot and show the time series on axis ax1
        plt.style.use('bmh')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.scatter(dates, counts)
        plt.title('Daily change in receipts count')
        plt.xlabel('Date')
        plt.ylabel('Number of receipts')
        plt.savefig(path)


if __name__ == "__main__":
    CONFIG_PATH = "Config.yml"
    ModelInference(input_data_path="../data/data_daily.csv").model_result()
