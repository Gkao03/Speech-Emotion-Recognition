import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def save_model(save_path, save_dict):
    torch.save(save_dict, save_path)
