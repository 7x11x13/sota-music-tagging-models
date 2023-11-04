from pathlib import Path

import model as Model
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    model_name: str, dataset_name: str, n_stems: int, model_load_path: Path | None
) -> tuple[torch.nn.Module, int]:
    model, input_length = Model.get_model(model_name, dataset_name, n_stems)

    # load pretrained model
    if model_load_path is not None:
        if model_load_path.exists():
            # load model
            S = torch.load(model_load_path)
            if "spec.mel_scale.fb" in S.keys():
                model.spec.mel_scale.fb = S["spec.mel_scale.fb"]
            model.load_state_dict(S)
        else:
            print(f"Could not load model from '{model_load_path}'")

    # cuda
    model.to(device)

    return model, input_length


def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(device)


def to_var(x: torch.Tensor) -> Variable:
    return Variable(x.to(device))
