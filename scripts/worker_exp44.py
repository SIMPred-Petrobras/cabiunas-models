#!/usr/bin/env python3
"""Roda o EXP 44 (K-Means, análise fatorial e autoencoder LSTM contra o PCA) num worker do ClearML.

Insumos (só leitura, datasets do ClearML): pacote de produção (b71d7c5f...: scripts, modelos, registro_trips) e insumos do EXP 44 (a definir no --enqueue: grade2min, controle_v2/atual, escores_sub34 A de t e p).
O worker monta a árvore que `nv.io`/`nv.avalia` esperam e chama `main()` do experimento sem alterar critério nenhum.

Local de teste (sem ClearML, com pastas locais):  PACOTE_DIR=... INSUMOS_DIR=... python scripts/worker_exp44.py --teste
Enfileirar:  PYTHONPATH=src python scripts/worker_exp44.py --enqueue --queue default --insumos ID
Variáveis: EXP44_PROCS (processos do Pool; padrão = nº de CPUs até 12), EXP44_THREADS (threads do torch por processo), EXP44_AUDITORIA_PARALELA=1.
"""
import argparse, importlib.util, os, sys, tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src"))
DS_PACOTE = "b71d7c5fb84d450ba8f314e49923696f"


def obtem(ds_id, env):
    if os.environ.get(env):
        return Path(os.environ[env])
    from clearml import Dataset
    return Path(Dataset.get(dataset_id=ds_id).get_local_copy())


def monta(insumos_id):
    p = obtem(DS_PACOTE, "PACOTE_DIR"); i = obtem(insumos_id, "INSUMOS_DIR"); r = Path(tempfile.mkdtemp(prefix="nvdados_")); pk = r / "estrutura_pra_prod" / "Cabiunas"; pk.mkdir(parents=True)
    for nome in ("scripts", "modelos", "registro_trips.csv"):
        (pk / nome).symlink_to(p / nome)
    (r / "scripts").mkdir(); (r / "scripts" / "pdm_fisico").symlink_to(i / "scripts_pdm_fisico")
    dl = RAIZ / "data_local"; dl.mkdir(exist_ok=True)
    for nome in ("controle_v2", "escores_sub34"):
        alvo = dl / nome
        if alvo.is_symlink() or alvo.exists():
            raise SystemExit(f"{alvo} já existe; não sobrescrevo")
        alvo.symlink_to(i / nome)
    os.environ["NV_DADOS"] = str(r); return r


def carrega_exp():
    spec = importlib.util.spec_from_file_location("exp44", RAIZ / "experiments" / "44_modelos_da_literatura.py"); m = importlib.util.module_from_spec(spec); sys.path.insert(0, str(RAIZ / "experiments")); spec.loader.exec_module(m); return m


def teste(m):
    """Encanamento sem custo: lê grade, máscara, controle, registro e modelos; um walk curto da análise fatorial."""
    import contextlib, io
    import numpy as np, pandas as pd
    from nv import io as nvio
    from nv.avalia import carrega_controle
    from nv.causal import carrega_registro
    from nv.walk import walk_forward
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); print("grade", g.shape, g.index.min(), g.index.max(), "| estáveis", int(est.sum()), flush=True)
    al, vig = carrega_controle("atual"); print("controle", len(al), "| alarme", int(al.sum()), "| vigiado", int(vig.sum()), flush=True); reg = carrega_registro(); print("trips", len(reg), flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = m._ci().carregar_modelos(m.PKG / "modelos")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in m.E20.FAMILIAS.items()}; print("tags", {c: len(v) for c, v in tags.items()}, flush=True)
    for c in ("t", "p"):
        for n in (19000, 20000):
            f = m.ROOT / "escores_sub34" / f"{c}_A_{n}.parquet"; print("referência", f.name, "ok" if f.exists() else "AUSENTE", flush=True)
    ini = m.INICIO8; s = walk_forward(m.FABR["FA"], g, tags["t"], est, ini, ini + pd.Timedelta(days=3), n_treino=3000, semente=0); print("walk FA curto:", s.shape, "| NaN", int(s.isna().sum()), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim"); ap.add_argument("--insumos", default=os.environ.get("INSUMOS_ID", "")); ap.add_argument("--teste", action="store_true"); a = ap.parse_args()
    if a.enqueue:
        from clearml import Task
        if not a.queue or not a.insumos:
            raise SystemExit("--queue e --insumos são obrigatórios com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name="nova-visao | EXP 44 modelos da literatura (KM, FA, LSTM)"); t.set_base_docker(a.image)
        t.set_packages(["torch @ https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl", "numpy==1.26.4", "pandas==3.0.2", "pyarrow==23.0.1", "scikit-learn==1.9.0", "scipy==1.17.1", "statsmodels==0.15.0", "lightgbm==4.7.0", "clearml==2.1.6"])
        t.set_parameters({"General/insumos": a.insumos, "General/EXP44_PROCS": "12", "General/EXP44_THREADS": "1", "General/EXP44_AUDITORIA_PARALELA": "1"}); t.execute_remotely(queue_name=a.queue)
    ins = a.insumos
    if not a.enqueue:
        from clearml import Task
        tk = Task.current_task()
        if tk is not None:
            par = tk.get_parameters(); ins = ins or par.get("General/insumos", "")
            for k in ("EXP44_PROCS", "EXP44_THREADS", "EXP44_AUDITORIA_PARALELA"):
                if par.get(f"General/{k}"): os.environ.setdefault(k, par[f"General/{k}"])
    monta(ins); os.environ.setdefault("EXP44_PROCS", str(min(12, os.cpu_count() or 4)))
    print("EXP 44 no worker: CPUs", os.cpu_count(), "| PROCS", os.environ["EXP44_PROCS"], "| THREADS", os.environ.get("EXP44_THREADS"), flush=True)
    m = carrega_exp()
    teste(m) if a.teste else m.main()
    if not a.teste:
        from clearml import Task
        tk = Task.current_task()
        if tk is not None:
            for f in sorted((RAIZ / "data_local" / "escores_lit44").glob("*.parquet")):
                tk.upload_artifact(f.stem, artifact_object=str(f))


if __name__ == "__main__":
    main()
