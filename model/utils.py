import json
import yaml
from model.model_definition import MLP
import torch


CONFIG_PATH = "Config.yml"

def write_to_json(dictionary, file_save_path):
    with open(file_save_path, "w+") as f:
        json.dump(dictionary, f)


def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    return model.eval()