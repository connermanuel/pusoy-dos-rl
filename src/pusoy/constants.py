import torch

OUTPUT_SIZES = [52, 5, 5]
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
