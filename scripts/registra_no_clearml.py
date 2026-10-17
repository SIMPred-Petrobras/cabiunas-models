#!/usr/bin/env python3
"""Registra os resultados da frente `exp/nova-visao` numa ÚNICA tarefa do ClearML, no projeto atual (`Cabiunas`).

Sobe só texto e tabelas: o resumo (o veredito e o placar de cada experimento, extraídos dos docstrings), a tabela de experimentos com o commit de cada um e os logs de texto de data_local/exp*.log. NÃO sobe dados (grade, parquets, diário de alarmes) e NÃO enfileira nada em worker.
A tarefa é criada a partir deste worktree, então o ClearML registra o repositório, a branch (exp/nova-visao) e o commit.

Uso:  python scripts/registra_no_clearml.py --seco        (monta os arquivos em data_local/clearml_resumo/, sem tocar no servidor)
      CLEARML_CONFIG_FILE=<clearml.conf> python scripts/registra_no_clearml.py --projeto Cabiunas --enviar
"""
import argparse, re, subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]; SAIDA = RAIZ / "data_local" / "clearml_resumo"


def docstring(p):
    t = p.read_text(encoding="utf-8"); m = re.match(r'#!.*?\n"""(.*?)"""', t, re.S); return m.group(1) if m else ""


def commit(p):
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(p)], cwd=RAIZ, capture_output=True, text=True).stdout.strip()


def extrai(ds):
    titulo = ds.strip().splitlines()[0].strip() if ds.strip() else ""; r = re.search(r"RESULTADO \([^)]*\)\s*(?:--|—)?\s*(.*?)(?=\n  |\n\n|$)", ds, re.S); veredito = (r.group(1).strip() if r else "").replace("\n", " ")[:300]
    pl = re.search(r"PLACAR:(.*?)(?=\n\n|\nUso:|$)", ds, re.S); return titulo, veredito, (pl.group(1).strip() if pl else "")


def monta():
    SAIDA.mkdir(parents=True, exist_ok=True); linhas = []; md = ["# Frente exp/nova-visao: resultados dos experimentos (EXP 00 a 45)\n", "Resumo extraído dos docstrings dos experimentos (o código está na branch `exp/nova-visao`). Cada experimento foi PRÉ-REGISTRADO (critérios e expectativas commitados antes de rodar), com o placar de acertos e erros das expectativas.\n"]
    for p in sorted((RAIZ / "experiments").glob("[0-9]*.py")):
        ds = docstring(p); titulo, veredito, placar = extrai(ds); exp = p.stem.split("_")[0].upper(); c = commit(p); linhas.append((exp, p.name, c, titulo, veredito))
        md.append(f"## EXP {exp} ({p.name}, commit {c})\n**{titulo}**\n"); md.append(f"- Veredito: {veredito or '(sem bloco RESULTADO: descritivo ou pré-registro)'}\n"); md.append(f"- Placar: {placar[:700] if placar else '(sem placar)'}\n")
    (SAIDA / "RESUMO_NOVA_VISAO.md").write_text("\n".join(md), encoding="utf-8")
    import csv
    with open(SAIDA / "experimentos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["exp", "arquivo", "commit", "titulo", "veredito"]); w.writerows(linhas)
    logs = sorted((RAIZ / "data_local").glob("exp*.log")); return linhas, logs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seco", action="store_true"); ap.add_argument("--enviar", action="store_true"); ap.add_argument("--projeto", default="Cabiunas"); a = ap.parse_args()
    linhas, logs = monta(); print(f"{len(linhas)} experimentos; {len(logs)} logs de texto; arquivos em {SAIDA}")
    if a.enviar:
        from clearml import Task
        t = Task.init(project_name=a.projeto, task_name="nova-visao | resultados dos experimentos EXP 00 a 45 (exp/nova-visao)", task_type=Task.TaskTypes.custom, reuse_last_task_id=False, auto_connect_frameworks=False, auto_resource_monitoring=False)
        t.add_tags(["nova-visao", "exp/nova-visao", "resultados", "pre-registrado"]); t.set_comment("Resultados consolidados da frente exp/nova-visao (pré-registro, placar, veredito). Só texto e tabelas: nenhum dado foi enviado. O código está na branch exp/nova-visao do repositório.")
        t.upload_artifact("RESUMO_NOVA_VISAO", artifact_object=str(SAIDA / "RESUMO_NOVA_VISAO.md")); t.upload_artifact("experimentos", artifact_object=str(SAIDA / "experimentos.csv"))
        for lg in logs:
            t.upload_artifact(f"log_{lg.stem}", artifact_object=str(lg))
        print("tarefa:", t.id, t.get_output_log_web_page()); t.close()
