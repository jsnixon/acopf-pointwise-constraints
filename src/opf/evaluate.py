import pandas as pd
import torch
from opf.test import test_run

torch.set_float32_matmul_precision("high")

RUN_ID = "j6x5931r"  # from W&B
CASE_NAME = "IEEE 30"

df = test_run(
    RUN_ID,
    load_existing=True,
    project=True,
    clamp=False,
    output_root_path="data/out",
    data_dir="data",
    best=True,
)

df = df.assign(
    id=RUN_ID,
    case_name=CASE_NAME,
    model_name="Dual-P",
    optimality_gap=lambda df: df["test/cost"] / df["acopf/cost"] - 1,
)

print(df[["optimality_gap", "test_normal/inequality/error_mean", "test_normal/inequality/error_max"]].describe())