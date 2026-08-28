import pandas as pd
import torch
from opf.test import load_run, test_run, data_to_device

torch.set_float32_matmul_precision("high")

RUN_ID = "b3n0m1f6"  # from W&B
CASE_NAME = "IEEE 30"

dm, opfdual = load_run(RUN_ID, batch_size=500, data_dir="data", best=True)

# extract pointwise multipliers — shape (n_train, n_multipliers)
mp = opfdual.model_dual.multipliers_pointwise.detach().cpu().numpy()
print(f"multipliers_pointwise shape: {mp.shape}")

# save to csv
mp_df = pd.DataFrame(mp)
mp_df.to_csv(f"multipliers_pointwise_{RUN_ID}.csv", index=False)
print(f"Saved to multipliers_pointwise_{RUN_ID}.csv")

# per constraint stats (column = one constraint across all training samples)
stats = pd.DataFrame({
    'mean': mp_df.mean(),
    'std': mp_df.std(),
    'min': mp_df.min(),
    'max': mp_df.max(),
    'median': mp_df.median(),
})
print("\nPer constraint statistics:")
print(stats.describe())
stats.to_csv(f"multiplier_stats_{RUN_ID}.csv")
print(f"Saved to multiplier_stats_{RUN_ID}.csv")

'''
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

print(df[['optimality_gap', 
           'test_normal/inequality/error_mean', 
           'test_normal/inequality/error_max',
           'test_normal/equality/error_mean',
           'test_normal/equality/error_max',
           'test/cost',
           'acopf/cost',
           'test_normal/inequality/rate',
           'test_normal/equality/rate']].describe())

'''