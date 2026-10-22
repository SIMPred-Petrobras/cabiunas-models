#!/usr/bin/env python3
"""Roda o EXP 63 (faixa física dos mancais) num worker do ClearML.

Insumos (sem dataset novo): pacote de produção (b71d7c5f...), insumos do EXP 60 (86e3990f...: grade2min, controle_v2, escores_sub34, sinais_v2_8ev.parquet) e, para o replay
causal do braço A, o CSV de produção de 2 min (bb85d853...) e o complemento de 30 s de 2024 (c9afd177...), os mesmos da verificação do harness.

Teste local (sem ClearML, caminhos locais, só encanamento):  PYTHONPATH=src python scripts/worker_exp63.py --local
Enfileirar:  PYTHONPATH=src python scripts/worker_exp63.py --enqueue --queue default
"""
import argparse, importlib.util, os, sys, tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "scripts"))
import worker_exp44 as w44
from worker_verifica_harness import DS_CSV, DS_2024

NOME = "modos-de-falha | EXP 63 faixa dos mancais"
DS_INSUMOS60 = "86e3990fde354c039ce38aab508f344d"   # insumos do EXP 60 (filho do 0367a2b1 + controle_v2/G168 e B800)
PACOTES = ["torch @ https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl", "numpy==1.26.4", "pandas==3.0.2", "pyarrow==23.0.1",
           "scikit-learn==1.9.0", "scipy==1.17.1", "statsmodels==0.15.0", "lightgbm==4.7.0", "clearml==2.1.6"]


def monta(insumos_id):
    p = w44.obtem(w44.DS_PACOTE, "PACOTE_DIR"); i = w44.obtem(insumos_id, "INSUMOS_DIR")
    r = Path(tempfile.mkdtemp(prefix="nvdados_")); pk = r / "estrutura_pra_prod" / "Cabiunas"; pk.mkdir(parents=True)
    for nome in ("scripts", "modelos", "registro_trips.csv"):
        (pk / nome).symlink_to(p / nome)
    (r / "scripts").mkdir(); (r / "scripts" / "pdm_fisico").symlink_to(i / "scripts_pdm_fisico")
    dl = RAIZ / "data_local"; dl.mkdir(exist_ok=True)
    for nome in ("controle_v2", "escores_sub34", "sinais_v2_8ev.parquet"):
        alvo = dl / nome
        if alvo.is_symlink() or alvo.exists():
            raise SystemExit(f"{alvo} já existe; não sobrescrevo")
        alvo.symlink_to(i / nome)
    b = w44.obtem(DS_CSV, "CSV_DIR"); c = w44.obtem(DS_2024, "COMPL_DIR"); (pk / "dados" / "2025_2026").mkdir(parents=True)
    (pk / "dados" / "2025_2026" / "data_2025-01-01_2026-04-30_raw.csv").symlink_to(b / "data_2025-01-01_2026-04-30_raw.csv")
    alvo = dl / "raw30s_2024q4_2025q1.parquet"
    if alvo.is_symlink() or alvo.exists():
        raise SystemExit(f"{alvo} já existe; não sobrescrevo")
    alvo.symlink_to(c / "raw30s_2024q4_2025q1.parquet")
    os.environ["NV_DADOS"] = str(r); return r


def carrega_exp():
    sys.path.insert(0, str(RAIZ / "experiments"))
    spec = importlib.util.spec_from_file_location("exp63", RAIZ / "experiments" / "63_faixa_dos_mancais.py"); m = importlib.util.module_from_spec(spec); sys.modules["exp63"] = m
    spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--insumos", default=os.environ.get("INSUMOS_ID", DS_INSUMOS60)); ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    if a.local:
        os.environ["EXP63_TESTE"] = "1"; return carrega_exp().main()
    if a.enqueue:
        from clearml import Task
        if not a.queue:
            raise SystemExit("--queue é obrigatório com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name=NOME); t.set_base_docker(a.image, docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(PACOTES); t.set_parameters({"General/insumos": a.insumos, "General/EXP63_PROCS": "4"}); t.execute_remotely(queue_name=a.queue)
    if not a.enqueue and os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        tk = Task.init(project_name="Cabiunas", task_name=NOME)
        a.insumos = tk.get_parameter("General/insumos") or a.insumos     # o argparse roda antes do Task.init
        os.environ.setdefault("EXP63_PROCS", str(tk.get_parameter("General/EXP63_PROCS") or 4))
    monta(a.insumos)
    print("EXP 63 no worker: CPUs", os.cpu_count(), "| PROCS", os.environ.get("EXP63_PROCS"), flush=True)
    m = carrega_exp(); m.main()
    from clearml import Task
    tk = Task.current_task()
    if tk is not None and (m.ROOT / "exp63_resultados.csv").exists():
        tk.upload_artifact("exp63_resultados", artifact_object=str(m.ROOT / "exp63_resultados.csv"))


if __name__ == "__main__":
    main()
