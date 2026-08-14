"""
Demand optimization via backprop through the primal-dual model.
Finds a load profile that achieves a target dual value for
inequality/active_power/node2/b1 (upper active power limit of generator 2).

Usage:
    poetry run python scripts/dual_optimization.py --target_dual 0.4 --n_steps 500 --lr 1e-2
"""

import sys
import os
sys.path.insert(0, '/home/jonathan/opf/src')
os.chdir('/home/jonathan/opf')

import argparse
import copy
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
import wandb

from opf.test import load_run, data_to_device
from opf.powerflow import GenParameters
from opf.dataset import PowerflowData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="uomqic9a")
    parser.add_argument("--target_dual", type=float, default=0.4)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--output_dir", type=str, default="data/out/demand_opt")
    return parser.parse_args()


def dual_optimization():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    api = wandb.Api()
    run = api.run(f"alelab/opf_param/{args.run_id}")
    artifacts = list(run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    best_artifact = min(model_artifacts, key=lambda a: a.metadata.get("score", float("inf")))
    print(f"Using: {best_artifact.name} score={best_artifact.metadata.get('score')}")

    with TemporaryDirectory() as tmpdir:
        checkpoint_path = best_artifact.download(root=tmpdir)
        ckpt_file = str(Path(checkpoint_path) / "model.ckpt")

        dm, opfdual = load_run(
            args.run_id, batch_size=500, data_dir="data", best=True,
            local_checkpoint_path=ckpt_file,
        )
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        dual_state = {
            k.replace("model_dual.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model_dual.")
        }
        opfdual.model_dual.load_state_dict(dual_state, strict=False)
        opfdual = opfdual.to(DEVICE)
        opfdual.eval()

        wandb.init(
            project="opf_param",
            entity="alelab",
            name=f"dual_opt_{args.target_dual:.3f}",
            config=vars(args)
        )

        # ── get starting sample ───────────────────────────────────────────────
        for data in dm.test_dataloader():
            data = data_to_device(data, DEVICE)
            break

        graph_single = data.graph[0].to(DEVICE)
        load_bus_ids = dm.powerflow_parameters.load_bus_ids.to(DEVICE)
        n_bus = dm.powerflow_parameters.n_bus

        init_load = graph_single["bus"].load[load_bus_ids].clone().detach()
        print(f"Initial load shape: {init_load.shape}")
        print(f"Initial dual: checking...")

        x_opt = init_load.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_opt], lr=args.lr)

        load_min = init_load * 0.5
        load_max = init_load * 1.5

        history = {
            'step': [], 'loss': [], 'violation': [], 'dual': [],
        }

        print(f"\nOptimizing to achieve target dual = {args.target_dual:.4f}")
        print(f"on inequality/active_power/node2/b1\n")

        for step in range(args.n_steps):
            optimizer.zero_grad()

            #with torch.no_grad():
                #x_opt.clamp_(min=load_min, max=load_max)

            # inject x_opt into graph
            graph_opt = copy.copy(graph_single)
            load_full = torch.zeros(n_bus, 2, device=DEVICE)
            load_full[load_bus_ids] = x_opt
            graph_opt["bus"].load = load_full

            dummy_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
            pf_data = PowerflowData(graph=graph_opt, index=dummy_index)

            # primal forward pass
            variables, _, _ = opfdual(pf_data)

            # constraint violation
            gen_params = GenParameters.from_tensor(graph_opt["gen"]["params"])
            sg_real = variables.Sg.real
            sg_max = gen_params.Sg_max.real
            violation = sg_real[2] - sg_max[2]

            # dual prediction — no torch.no_grad() so gradients flow to x_opt
            multipliers = opfdual.model_dual.get_multipliers(pf_data)
            dual = multipliers["inequality/active_power"][2, 1]

            # loss: MSE between predicted dual and target
            loss = (dual - args.target_dual) ** 2

            loss.backward()
            optimizer.step()

            history['step'].append(step)
            history['loss'].append(loss.item())
            history['violation'].append(violation.item())
            history['dual'].append(dual.item())

            if step % 50 == 0:
                print(f"Step {step:4d} | loss={loss.item():.6f} | "
                      f"violation={violation.item():.6f} | dual={dual.item():.6f}")

            wandb.log({
                'step': step,
                'loss': loss.item(),
                'violation': violation.item(),
                'dual': dual.item(),
            })

        # ── save results ──────────────────────────────────────────────────────
        df = pd.DataFrame(history)
        out_path = os.path.join(args.output_dir, f"dual_opt_{args.target_dual:.3f}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

        np.save(
            os.path.join(args.output_dir, f"x_opt_dual_{args.target_dual:.3f}.npy"),
            x_opt.detach().cpu().numpy()
        )
        print(f"Final violation: {history['violation'][-1]:.6f}")
        print(f"Final dual: {history['dual'][-1]:.6f}")
        print(f"Target dual: {args.target_dual:.6f}")

        wandb.finish()


if __name__ == "__main__":
    dual_optimization()