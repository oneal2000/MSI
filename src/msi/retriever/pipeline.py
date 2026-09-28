"""Public retriever prepare/train/retrieve orchestration."""
from msi.retriever.ownership import claim
import subprocess
import sys

from msi.runtime.pipeline import path, datasets


def run(config, *, dry_run=False):
    action = config["_selection"]["retriever_action"]
    root = path(config, "work_dir") / config["run_id"]
    gpu_count = len(config["_selection"]["gpus"])
    if dry_run:
        print(f"retriever {action}: root={root}; GPUs={gpu_count}; global batch={config['defaults']['batch_size']}")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with claim(root / f"{action}.claim"):
        from msi.config import write_resolved
        write_resolved(config, config["_resolved_config_file"])
        if action == "prepare":
            from msi.retriever.data import prepare
            prepare(config, root, {key: path(config, key) for key in
                    ("task_pool", "anchor_retrieval", "corpus", "instances_dir")})
        elif action == "train":
            if gpu_count not in (1, 2, 4):
                raise ValueError("retriever training supports 1, 2, or 4 GPUs")
            command = [config.get("_python") or sys.executable, "-m", "torch.distributed.run",
                       "--standalone", f"--nproc-per-node={gpu_count}", "-m", "msi.retriever.train",
                       "--config", config["_resolved_config_file"]]
            subprocess.run(command, check=True)
        else:
            from msi.retriever.retrieve import retrieve
            selected = path(config, "selected_checkpoint") if config["paths"].get("selected_checkpoint") else root / "selected.json"
            for dataset in datasets(config):
                output = path(config, "output_dir", f"results/retrieval/{config['run_id']}") / f"{dataset}.json"
                retrieve(selected_file=selected, corpus_file=path(config, "corpus"),
                         instances_file=path(config, "instances_dir") / f"{dataset}.json",
                         output=output, batch_size=config["defaults"]["eval_batch_size"])
    return 0
