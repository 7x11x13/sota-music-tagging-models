import argparse
import time
from pathlib import Path

import model as Model
import numpy as np
import torch
import torch.nn as nn
import tqdm
from datasets import DATASETS, SplitType, get_dataset
from eval import Predict
from matplotlib import pyplot as plt
from sklearn import metrics
from torch_utils import build_model, to_device, to_var


def eval(model_name: str, dataset: str, data_path: str, n_stems: int):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mtat",
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        choices=Model.MODEL_NAMES,
    )
    parser.add_argument("--n_stems", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_load_path", type=str, default="./models")
    parser.add_argument("--data_path", type=str, default="./data")

    config = parser.parse_args()
    config.data_path = data_path
    config.dataset = dataset
    config.model_type = model_name
    config.n_stems = n_stems

    model_load_dir: Path = (
        Path(config.model_load_path) / config.dataset / config.model_type
    )
    model_load_dir.mkdir(parents=True, exist_ok=True)

    config.model_load_path = (
        model_load_dir
        / f"best_model{'' if config.n_stems == 1 else f'_{config.n_stems}_stems'}.pth"
    )

    p = Predict(config)
    start = time.time()
    p.test()
    return time.time() - start


if __name__ == "__main__":
    models = [
        "short",
        "short_res",
        "short_multi_64",
        "short_multi_32",
        "short_res_multi_64",
        "short_res_multi_32",
    ]
    with open("eval_all_log.txt", "w") as f:
        for model in models:
            t = eval(model, "mtat", "F:/datasets/mtat_npy/npy", 1)
            f.write(f"{model} (mtat, 1): {t}\n")
            t = eval(model, "mtat", "F:/datasets/mtat_npy/npy_split", 4)
            f.write(f"{model} (mtat, 4): {t}\n")
            t = eval(model, "gtzan", "E:/datasets/GTZAN/npy", 1)
            f.write(f"{model} (gtzan, 1): {t}\n")
            t = eval(model, "gtzan", "E:/datasets/GTZAN/npy_split", 4)
            f.write(f"{model} (gtzan, 4): {t}\n")
