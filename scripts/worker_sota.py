#!/usr/bin/env python3
"""Worker genérico da frente "estado da arte" (BRIEF_estado_da_arte_cabiunas.md): roda `experiments/sota/<exp>.py` no ClearML com as tags do brief.

Insumos (sem dataset novo): pacote de produção (b71d7c5f), insumos do EXP 60 (86e3990f: grade2min, controle_v2, escores_sub34, sinais_v2_8ev, diário de alarmes),
CSV de produção de 2 min (bb85d853) e complemento de 30 s de 2024 (c9afd177). As planilhas do SAP NÃO vão para o worker (decisão do usuário, 22/10).
Enfileirar:  PYTHONPATH=src python scripts/worker_sota.py --exp f1_baseline --tags sota-baseline --enqueue --queue default
"""
import argparse, importlib, os, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "scripts")); sys.path.insert(0, str(RAIZ / "experiments"))
import worker_exp63 as w63

PACOTES = w63.PACOTES + ["river==0.24.1", "openpyxl==3.1.5"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--tags", nargs="*", default=[]); ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim"); ap.add_argument("--procs", default="16")
    a = ap.parse_args(); nome = f"sota | {a.exp}"
    if a.enqueue:
        from clearml import Task
        if not a.queue:
            raise SystemExit("--queue é obrigatório com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name=nome, tags=a.tags); t.set_base_docker(a.image, docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(PACOTES); t.set_parameters({"General/exp": a.exp, "General/insumos": w63.DS_INSUMOS60, "General/SOTA_PROCS": a.procs}); t.execute_remotely(queue_name=a.queue)
    if os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        tk = Task.init(project_name="Cabiunas", task_name=nome); a.exp = tk.get_parameter("General/exp") or a.exp
        os.environ.setdefault("SOTA_PROCS", str(tk.get_parameter("General/SOTA_PROCS") or a.procs))
        ins = tk.get_parameter("General/insumos") or w63.DS_INSUMOS60; w63.monta(ins)
        os.environ["ALARMES_CSV"] = str(w63.w44.obtem(ins, "INSUMOS_DIR") / "alarmes" / "alarmes_mapeados_colunas.csv"); os.environ["SAP_DIR"] = "/nao-enviado"
    m = importlib.import_module(f"sota.{a.exp}"); m.main()
    from clearml import Task
    tk = Task.current_task()
    if tk is not None and m.SAIDA.exists():
        for f in sorted(m.SAIDA.glob("*.csv")):
            tk.upload_artifact(f.stem, artifact_object=str(f))


if __name__ == "__main__":
    main()
