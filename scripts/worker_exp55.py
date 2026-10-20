#!/usr/bin/env python3
"""Roda o EXP 55 (reanálise por modo de falha, etapas 2 e 3) num worker do ClearML.

Insumos (só leitura, datasets do ClearML): pacote de produção (b71d7c5f...: scripts, modelos, registro_trips) e os insumos do EXP 55, filho do dataset do EXP 44 (grade2min,
controle_v2/atual) acrescido do cache causal `sinais_v2_8ev.parquet` e do diário de alarmes da planta (`alarmes/alarmes_mapeados_colunas.csv`).

Criar os insumos (uma vez):  PYTHONPATH=src python scripts/worker_exp55.py --cria-insumos
Teste local (sem ClearML, caminhos locais, só encanamento):  PYTHONPATH=src python scripts/worker_exp55.py --local
Enfileirar:  PYTHONPATH=src python scripts/worker_exp55.py --enqueue --queue default --insumos ID
"""
import argparse, importlib.util, os, sys, tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "scripts"))
import worker_exp44 as w44

NOME = "modos-de-falha | EXP 55 observabilidade por modo e atribuicao"
DS_PAI = w44.DS_INSUMOS
ALARMES_LOCAL = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv")
PACOTES = ["numpy==1.26.4", "pandas==3.0.2", "pyarrow==23.0.1", "scikit-learn==1.9.0", "scipy==1.17.1", "clearml==2.1.6"]


def cria_insumos():
    from clearml import Dataset
    ds = Dataset.create(dataset_name="nova-visao insumos EXP 55 (EXP 44 + sinais_v2_8ev + diario de alarmes)", dataset_project="TesteMLCab", parent_datasets=[DS_PAI])
    ds.add_files(str(RAIZ / "data_local" / "sinais_v2_8ev.parquet"))
    ds.add_files(str(ALARMES_LOCAL), dataset_path="alarmes")
    ds.upload(); ds.finalize(); print("insumos EXP 55:", ds.id, flush=True)


def monta(insumos_id):
    p = w44.obtem(w44.DS_PACOTE, "PACOTE_DIR"); i = w44.obtem(insumos_id, "INSUMOS_DIR")
    r = Path(tempfile.mkdtemp(prefix="nvdados_")); pk = r / "estrutura_pra_prod" / "Cabiunas"; pk.mkdir(parents=True)
    for nome in ("scripts", "modelos", "registro_trips.csv"):
        (pk / nome).symlink_to(p / nome)
    (r / "scripts").mkdir(); (r / "scripts" / "pdm_fisico").symlink_to(i / "scripts_pdm_fisico")
    dl = RAIZ / "data_local"; dl.mkdir(exist_ok=True)
    for nome in ("controle_v2", "sinais_v2_8ev.parquet"):
        alvo = dl / nome
        if alvo.is_symlink() or alvo.exists():
            raise SystemExit(f"{alvo} já existe; não sobrescrevo")
        alvo.symlink_to(i / nome)
    os.environ["NV_DADOS"] = str(r); os.environ["ALARMES_CSV"] = str(i / "alarmes" / "alarmes_mapeados_colunas.csv"); return r


def carrega_exp():
    spec = importlib.util.spec_from_file_location("exp55", RAIZ / "experiments" / "55_modos_de_falha.py"); m = importlib.util.module_from_spec(spec); sys.modules["exp55"] = m
    spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--insumos", default=os.environ.get("INSUMOS_ID")); ap.add_argument("--cria-insumos", action="store_true"); ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    if a.cria_insumos:
        return cria_insumos()
    if a.local:
        os.environ["EXP55_TESTE"] = "1"; return carrega_exp().main()
    if a.enqueue:
        from clearml import Task
        if not a.queue or not a.insumos:
            raise SystemExit("--queue e --insumos são obrigatórios com --enqueue")
        t = Task.init(project_name="Cabiunas", task_name=NOME); t.set_base_docker(a.image, docker_setup_bash_script=["apt-get update -qq", "apt-get install -y -qq libgomp1"])
        t.set_packages(PACOTES); t.set_parameters({"General/insumos": a.insumos}); t.execute_remotely(queue_name=a.queue)
    if not a.enqueue and os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        Task.init(project_name="Cabiunas", task_name=NOME)
    monta(a.insumos)
    print("EXP 55 no worker: CPUs", os.cpu_count(), flush=True)
    m = carrega_exp(); m.main()
    from clearml import Task
    tk = Task.current_task()
    if tk is not None:
        for f in ("exp55_observabilidade.csv", "exp55_episodios.csv", "exp55_deteccoes.csv"):
            if (m.ROOT / f).exists():
                tk.upload_artifact(f.split(".")[0], artifact_object=str(m.ROOT / f))


if __name__ == "__main__":
    main()
