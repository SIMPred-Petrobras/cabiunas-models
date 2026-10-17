#!/usr/bin/env python3
"""Verificação do harness no worker do ClearML: reconstrói os sinais causais do v2 a partir dos TRÊS datasets e compara com a impressão digital local (scripts/ref_sinais_v2_8ev.json).

Datasets (somente leitura): CSV de produção 2 min (bb85d853...), complemento 2024 (c9afd177...), pacote de produção (b71d7c5f...). NÃO usar o 1180990c (30 s): só 12% dos valores coincidem com o CSV de produção.
Local ou remoto:  PYTHONPATH=src python scripts/worker_verifica_harness.py            (executa onde estiver)
Enfileirar:       PYTHONPATH=src python scripts/worker_verifica_harness.py --enqueue --queue default
PASSA se: mesmo nº de linhas, mesmo nº de NaN e de máscara/reset, e média/dp/q50/q99 dos 4 sinais com erro relativo <= 1e-6.
"""
import argparse, json, os, sys, tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "scripts"))
DS_CSV, DS_2024, DS_PACOTE = "bb85d853f92545beb771d303354302ab", "c9afd177821d4ca39a9474cf50553e46", "b71d7c5fb84d450ba8f314e49923696f"


def monta_raiz():
    """Cria a árvore que `nv.io` espera (RAIZ/estrutura_pra_prod/Cabiunas/...) com links para os datasets; devolve (raiz, parquet do complemento de 2024)."""
    from clearml import Dataset
    b, c, p = (Path(Dataset.get(dataset_id=i).get_local_copy()) for i in (DS_CSV, DS_2024, DS_PACOTE)); r = Path(tempfile.mkdtemp(prefix="nvdados_")); pk = r / "estrutura_pra_prod" / "Cabiunas"; (pk / "dados" / "2025_2026").mkdir(parents=True)
    for nome in ("scripts", "modelos", "registro_trips.csv"):
        (pk / nome).symlink_to(p / nome)
    (pk / "dados" / "2025_2026" / "data_2025-01-01_2026-04-30_raw.csv").symlink_to(b / "data_2025-01-01_2026-04-30_raw.csv")
    return r, c / "raw30s_2024q4_2025q1.parquet"


def verifica(S, ref, tol=1e-6):
    from impressao_digital_sinais import impressao
    got = impressao(S); ok = True; linhas = []
    def cmp(nome, a, b, rel=False):
        nonlocal ok
        bom = (a == b) if not rel else (abs(a - b) <= tol * max(1.0, abs(b))); ok &= bool(bom); linhas.append(f"   {'OK ' if bom else 'FALHA'} {nome}: worker={a} referencia={b}")
    cmp("linhas", got["linhas"], ref["linhas"]); cmp("inicio", got["inicio"], ref["inicio"]); cmp("fim", got["fim"], ref["fim"])
    for c, v in ref["colunas"].items():
        for k, b in v.items():
            if k in ("soma", "maximo"): continue
            cmp(f"{c}.{k}", got["colunas"][c][k], b, rel=isinstance(b, float))
    return ok, linhas


def main():
    r, x_path = monta_raiz(); os.environ["NV_DADOS"] = str(r)
    import pandas as pd
    from nv.controle_sinais import sinais_causais
    ref = json.load(open(RAIZ / "scripts" / "ref_sinais_v2_8ev.json")); x = pd.read_parquet(x_path)
    print("SINAIS: iniciando o replay causal semanal (pode levar tempo)", flush=True)
    S = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=None, extra=x); print("SINAIS prontos:", S.shape, S.index.min(), S.index.max(), flush=True)
    ok, linhas = verifica(S, ref); print("\n".join(linhas)); print("RESULTADO:", "PASSA" if ok else "FALHA", flush=True)
    if not ok: raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim"); a = ap.parse_args()
    if a.enqueue:
        from clearml import Task
        if not a.queue: raise SystemExit("--queue é obrigatório com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name="nova-visao | verifica_harness (3 datasets)"); t.set_base_docker(a.image)
        t.set_packages(["numpy==1.26.4", "pandas==3.0.2", "pyarrow==23.0.1", "scikit-learn==1.9.0", "scipy==1.17.1", "clearml==2.1.6"]); t.execute_remotely(queue_name=a.queue)
    main()
