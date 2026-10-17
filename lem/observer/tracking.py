# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Optional live W&B tracking; local checkpoints remain the experiment record."""
import json
from pathlib import Path


def metrics(row, batch, training_views):
    values = {"training_step": row["step"], "epoch_equivalent": row["step"] * batch / training_views,
              "train_batch_loss": row["train_batch_nll"]}
    if "train_probe" in row:
        values.update(train_probe_loss=row["train_probe"]["all"]["nll"],
                      train_probe_accuracy=row["train_probe"]["all"]["top1"])
    for group, stats in row["validation"].items():
        for name in ("nll", "top1", "known_impossible_mass"):
            values[f"validation_{group}_{name}"] = stats[name]
    return values


class Tracking:
    def __init__(self, args, config, about, history, training_views):
        self.args, self.run, self.training_views = args, None, training_views
        self.out = Path(args.out)
        if not args.wandb_project:
            return
        import wandb
        self.sdk = wandb
        receipt = self.out / "wandb-run.json"
        identity = json.loads(receipt.read_text()) if args.resume and receipt.exists() else None
        if identity and (identity["project"] != args.wandb_project or
                         (args.wandb_entity and identity["entity"] != args.wandb_entity)):
            raise ValueError("resume must retain the W&B project and entity")
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
            id=identity["id"] if identity else None, resume="must" if identity else "never",
            dir=str(self.out), mode="online", save_code=False,
            config={**vars(args), **about, "model": config, "training_views": training_views,
                    "epoch_definition": "training_step * batch / training_views; sampling with replacement",
                    "learning_rate_schedule": "constant", "early_stopping": False},
            tags=["tiny_gpt", "observer", "formation", "deliberate_overtraining"],
        )
        receipt.write_text(json.dumps({"id": self.run.id, "url": self.run.url,
                                       "entity": self.run.entity, "project": self.run.project}, indent=2) + "\n")
        self.run.define_metric("epoch_equivalent")
        self.run.define_metric("*", step_metric="epoch_equivalent", step_sync=True)
        self.run.summary["status_note"] = "training; best and final checkpoints retained separately"
        print(f"LIVE_RUN {self.run.url}", flush=True)
        if not identity:
            for row in history:
                self.log(row)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.run:
            self.run.finish(exit_code=1 if kind else 0)

    def log(self, row):
        if self.run:
            self.run.log(metrics(row, self.args.batch, self.training_views))

    def checkpoint(self, step):
        if not self.run:
            return
        artifact = self.sdk.Artifact(f"observer_{self.run.id}", type="model",
                                     metadata={"training_step": step})
        artifact.add_file(str(self.out / f"step-{step:05d}.pt"), name="checkpoint.pt")
        self.run.log_artifact(artifact, aliases=["latest", f"step_{step}"])

    def finish(self, report):
        if not self.run:
            return
        self.run.summary.update({"best_step": report["best_step"], "final_step": report["steps"],
                                 "best_test_nll": report["test"]["all"]["nll"],
                                 "final_test_nll": report["final_test"]["all"]["nll"],
                                 "status_note": "completed; test is exploratory after the first run"})
        artifact = self.sdk.Artifact(f"observer_evidence_{self.run.id}", type="research")
        for name in ("initial.pt", "best.pt", "latest.pt", "report.json", "data.json", "deals.json", "initial.json"):
            artifact.add_file(str(self.out / name), name=name)
        self.run.log_artifact(artifact, aliases=["complete"])
