"""
Demand optimization via backprop through the primal-dual model.
Finds a load profile that achieves a target violation level on
inequality/active_power/node2/b1 (upper active power limit of generator 2).

Usage:
    python scripts/analyze/demand_optimization.py --target_violation 0.01 --n_steps 500 --lr 1e-3
"""

import sys
import os
sys.path.insert(0, '/home/jonathan/opf/src')
os.chdir('/home/jonathan/opf')

import argparse
import torch
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import wandb
import copy

from opf.dataset import PowerflowData
from opf.test import load_run, data_to_device
from opf.powerflow import BranchParameters, GenParameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="uomqic9a")
    parser.add_argument("--target_violation", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="data/out/demand_opt")
    return parser.parse_args()


def get_gen2_upper_violation(variables, graph, opfdual):
    """Compute violation of upper active power limit on generator 2."""
    multipliers = opfdual.model_dual.get_multipliers(
        type('obj', (object,), {'graph': graph})()
    )
    
    # get raw violation: Sg.real - Sg_max
    gen_params = GenParameters.from_tensor(graph["gen"]["params"])
    sg_real = variables.Sg.real  # (n_gen,)
    sg_max = gen_params.Sg_max.real  # (n_gen,)
    
    # upper bound violation for generator 2 (node index 2)
    violation = sg_real[2] - sg_max[2]
    
    # predicted dual for this constraint
    dual = multipliers["inequality/active_power"][2, 1]  # node 2, upper bound
    
    return violation, dual


def main():
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
            name=f"demand_opt_target{args.target_violation:.3f}",
            config=vars(args)
        )       
        # ── get a starting sample from test set ───────────────────────────────
        for data in dm.test_dataloader():
            data = data_to_device(data, DEVICE)
            break  # just take first batch, use first sample

        # extract single sample graph
        graph_single = data.graph[0]  # first graph in batch
        graph_single = graph_single.to(DEVICE)

        # get load_bus_ids and initial load
        load_bus_ids = dm.powerflow_parameters.load_bus_ids.to(DEVICE)
        n_bus = dm.powerflow_parameters.n_bus
        n_load = load_bus_ids.shape[0]

        # initial load from the sample
        init_load = graph_single["bus"].load[load_bus_ids].clone().detach()
        print(f"Initial load shape: {init_load.shape}")  # (21, 2)

        # ── setup x_opt ───────────────────────────────────────────────────────
        x_opt = init_load.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_opt], lr=args.lr)

        # load bounds from training data (0.8 to 1.2 of base)
        load_min = init_load * 0.8 / 1.0  # scale relative to initial
        load_max = init_load * 1.2 / 1.0

        # ── optimization loop ─────────────────────────────────────────────────
        history = {
            'step': [],
            'loss': [],
            'violation': [],
            'dual': [],
        }

        print(f"\nOptimizing to achieve target violation = {args.target_violation:.4f}")
        print(f"on inequality/active_power/node2/b1\n")

        for step in range(args.n_steps):
            optimizer.zero_grad()

            # clamp x_opt to physically realistic range
            #with torch.no_grad():
                #x_opt.clamp_(min=load_min, max=load_max)

            # inject x_opt into graph
            graph_opt = copy.copy(graph_single)
            load_full = torch.zeros(n_bus, 2, device=DEVICE)
            load_full[load_bus_ids] = x_opt
            graph_opt["bus"].load = load_full

            # forward pass through primal model
            from opf import powerflow as pf
            powerflow_params = dm.powerflow_parameters
            
            # run primal GNN
            dummy_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
            pf_data = PowerflowData(graph=graph_opt, index=dummy_index)
            variables, _, _ = opfdual(pf_data)

            # get violation and dual
            gen_params = GenParameters.from_tensor(graph_opt["gen"]["params"])
            sg_real = variables.Sg.real
            sg_max = gen_params.Sg_max.real
            violation = sg_real[2] - sg_max[2]

            multipliers = opfdual.model_dual.get_multipliers(pf_data)
            dual = multipliers["inequality/active_power"][2, 1]

            # loss: MSE between actual violation and target
            loss = (violation - args.target_violation) ** 2

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
        import pandas as pd
        df = pd.DataFrame(history)
        out_path = os.path.join(args.output_dir, f"opt_target{args.target_violation:.3f}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

        # final optimized load
        np.save(
            os.path.join(args.output_dir, f"x_opt_target{args.target_violation:.3f}.npy"),
            x_opt.detach().cpu().numpy()
        )
        print(f"Final violation: {history['violation'][-1]:.6f}")
        print(f"Final dual: {history['dual'][-1]:.6f}")
        print(f"Target violation: {args.target_violation:.6f}")

        wandb.finish()

if __name__ == "__main__":
    main()