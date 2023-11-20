import argparse
import time
from pathlib import Path

from datasets import DATASETS
from model import MODEL_NAMES
from solver import Solver


def train(model_name: str, dataset: str, data_path: str, n_stems: int):
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
        choices=MODEL_NAMES,
    )
    parser.add_argument("--n_stems", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--model_load_path", type=str, default="./models")
    parser.add_argument("--load_model", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_step", type=int, default=20)

    config = parser.parse_args()
    config.load_model = 0
    config.data_path = data_path
    config.dataset = dataset
    config.model_type = model_name
    config.n_stems = n_stems

    config.model_load_path = None

    model_save_dir: Path = (
        Path(config.model_save_path) / config.dataset / config.model_type
    )
    model_save_dir.mkdir(parents=True, exist_ok=True)
    config.model_save_path = (
        model_save_dir
        / f"best_model{'' if config.n_stems == 1 else f'_{config.n_stems}_stems'}.pth"
    )
    solver = Solver(config)
    start = time.time()
    solver.train()
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
    with open("train_all_log.txt", "w") as f:
        for model in models:
            t = train(model, "mtat", "F:/datasets/mtat_npy/npy", 1)
            f.write(f"{model} (mtat, 1): {t}\n")
            t = train(model, "mtat", "F:/datasets/mtat_npy/npy_split", 4)
            f.write(f"{model} (mtat, 4): {t}\n")
            t = train(model, "gtzan", "E:/datasets/GTZAN/npy", 1)
            f.write(f"{model} (gtzan, 1): {t}\n")
            t = train(model, "gtzan", "E:/datasets/GTZAN/npy_split", 4)
            f.write(f"{model} (gtzan, 4): {t}\n")
