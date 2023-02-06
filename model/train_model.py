import json
import traceback
import yaml
import os
from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from data_preprocessing import TrainTestHelper
from model_definition import MLP

CONFIG_PATH = "Config.yml"


class Forecast:

    def __init__(self):
        with open(CONFIG_PATH) as file:
            self.config = yaml.safe_load(file)

        # Split dataset into train and validation

    def generate_train_test_split(self):
        torch.manual_seed(42)
        data = TrainTestHelper.data_preprocessing(self.config["dataset"]["path"])
        train_size = int(len(data) * self.config["dataset"]["split_ratio"])
        train_dataset = TrainTestHelper(data.iloc[:train_size, :])
        val_dataset = TrainTestHelper(data.iloc[train_size:, :])
        batch_size = self.config["model"]["batch_size"]
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        return trainloader, valloader

    # implementing ridge regression
    def train_model(self, visualize=False):
        trainloader, valloader = self.generate_train_test_split()
        mlp = MLP()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.config["model"]["initial_lr"])
        overall_loss = []
        # Run the training loop
        for epoch in range(0, self.config["model"]["epochs"]):  # 5 epochs at maximum
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            loss_arr = []
            # Set current loss value
            current_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                # optimizer.zero_grad()

                # Perform forward pass
                outputs = mlp(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                loss_arr.append(loss.item())
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / (i + 1) * self.config["model"]["batch_size"]))
                    current_loss = 0.0
            overall_loss.append(np.mean(loss_arr))
        # Process is complete.
        print('Training process has finished.')
        val_loss = self.prediction(valloader, mlp)
        self.save_model(mlp, self.config["model"]["model_output_path"])
        print("validation loss", val_loss)
        plt.plot(np.arange(1, self.config["model"]["epochs"] + 1), overall_loss)
        plt.savefig(self.config["model"]["mode_forecast_image_path"])
        if visualize:
            plt.show()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def prediction(self, data_loader, model):
        losses = []
        loss_function = nn.MSELoss()
        for i, data in enumerate(data_loader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = model(inputs)

            loss = loss_function(outputs, targets)

            losses.append(loss.data.item())
        return np.mean(losses) / len(list(data_loader))


# observe error decrease and converge across iterations

if __name__ == "__main__":
    forecast_obj = Forecast()
    result = forecast_obj.train_model(visualize=True)
    print(result)
