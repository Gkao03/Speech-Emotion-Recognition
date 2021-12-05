import torch


def save_model(save_path, save_dict):
    torch.save(save_dict, save_path)
