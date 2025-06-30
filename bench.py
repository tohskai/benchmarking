import argparse
import configparser
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch.profiler import record_function


def read_config(path: str) -> configparser.ConfigParser:
    """Read an INI-style configuration file."""
    cfg = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        cfg.read_file(f)
    return cfg


def str_to_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(mapping)}.")
    return mapping[key]


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def build_model(cfg: configparser.ConfigParser) -> nn.Module:
    d_in = cfg.getint("model", "d_in")
    d_hidden = cfg.getint("model", "d_hidden")
    compile_flag = cfg.getboolean("run", "compile")
    device = cfg.get("run", "device")
    dtype = str_to_dtype(cfg.get("run", "dtype"))

    model = MLP(d_in, d_hidden).to(device=device, dtype=dtype)
    if compile_flag:
        model = torch.compile(model)
    return model


def build_dataloader(
    cfg: configparser.ConfigParser, d_in: int, dtype: torch.dtype
) -> DataLoader:
    batch_size = cfg.getint("data", "batch_size")

    # Dummy dataset (10k samples). Replace with real dataset as needed.
    X = torch.randn(10_000, d_in, dtype=dtype)
    y = torch.randn(10_000, d_in, dtype=dtype)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    x, y = batch
    x = x.to(device=device, dtype=next(model.parameters()).dtype)
    y = y.to(device=device, dtype=next(model.parameters()).dtype)

    optimizer.zero_grad(set_to_none=True)

    with record_function("forward"):
      pred = model(x)
    with record_function("loss"):
      loss = criterion(pred, y)
    with record_function("backward"):
      loss.backward()
    with record_function("optimizer"):
      optimizer.step()

    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLP with configurable runtime options."
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to config.ini file"
    )
    args = parser.parse_args()

    cfg = read_config(args.config)

    device = torch.device(cfg.get("run", "device"))
    dtype = str_to_dtype(cfg.get("run", "dtype"))
    d_in = cfg.getint("model", "d_in")
    iterations = cfg.getint("profiling", "iterations")
    warmup_iters = cfg.getint("profiling", "warmup")
    profile_flag = cfg.getboolean("profiling", "profile")

    model = build_model(cfg)
    dataloader = build_dataloader(cfg, d_in, dtype)

    criterion = torch.compile(nn.MSELoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def run_loop():
        for step, batch in enumerate(dataloader):
            torch.cuda.cudart().cudaProfilerStart()
            loss_val = train_step(model, batch, criterion, optimizer, device)
            if step % 10 == 0:
                print(f"step={step:04d} | loss={loss_val:.5f}")
            if step >= warmup_iters + iterations - 1:
                torch.cuda.cudart().cudaProfilerStop()
                break

    if profile_flag:
        # PyTorch Profiler context
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=warmup_iters, warmup=warmup_iters, active=iterations, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for step, batch in enumerate(dataloader):
                loss_val = train_step(model, batch, criterion, optimizer, device)
                prof.step()
                if step % 10 == 0:
                    print(f"[PROF] step={step:04d} | loss={loss_val:.5f}")
                if step >= warmup_iters + iterations - 1:
                    break
    else:
        run_loop()


if __name__ == "__main__":
    main()
