#!/usr/bin/env python3
"""Roda o EXP 46 (Optuna sobre modelos, com LOEO e placebo) num worker do ClearML. Reaproveita a montagem de insumos do worker do EXP 44.

Teste local (sem ClearML):  PACOTE_DIR=... INSUMOS_DIR=... python scripts/worker_exp46.py --teste
Enfileirar:  PYTHONPATH=src python scripts/worker_exp46.py --enqueue --queue default
"""
import argparse, importlib.util, os, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "scripts"))
import worker_exp44 as w44


def carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "experiments" / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; sys.path.insert(0, str(RAIZ / "experiments")); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim"); ap.add_argument("--teste", action="store_true"); a = ap.parse_args()
    nome = "nova-visao | EXP 46 Optuna sobre modelos (LOEO e placebo)"
    if a.enqueue:
        from clearml import Task
        if not a.queue:
            raise SystemExit("--queue é obrigatório com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name=nome); t.set_base_docker(a.image, docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(["torch @ https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl", "numpy==1.26.4", "pandas==3.0.2", "pyarrow==23.0.1", "scikit-learn==1.9.0", "scipy==1.17.1", "statsmodels==0.15.0", "lightgbm==4.7.0", "optuna==5.0.0", "clearml==2.1.6"]); t.execute_remotely(queue_name=a.queue)
    if not a.enqueue and os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        Task.init(project_name="Cabiunas", task_name=nome); os.environ.setdefault("EXP46_PROCS", str(min(60, os.cpu_count() or 4)))
    if a.teste:
        os.environ["EXP46_TESTE"] = "1"
    w44.monta(os.environ.get("INSUMOS_ID", w44.DS_INSUMOS)); os.environ.setdefault("EXP46_PROCS", str(min(60, os.cpu_count() or 4)))
    print("EXP 46 no worker: CPUs", os.cpu_count(), "| PROCS", os.environ["EXP46_PROCS"], "| teste", a.teste, flush=True)
    m = carrega("exp46", "46_optuna_modelos_loeo_placebo.py"); import pickle
    for fn in (m.pontua, m.estudo):
        pickle.dumps(fn)
    print("pickle das funcoes do Pool: ok", flush=True); m.main()
    if os.environ.get("CLEARML_TASK_ID") and not a.teste:
        from clearml import Task
        tk = Task.current_task()
        if tk is not None:
            tk.upload_artifact("exp46_estudos", artifact_object=str(m.ROOT / "exp46_estudos.csv"))


if __name__ == "__main__":
    main()
