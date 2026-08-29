"""v13: testa `--per-family-thresholds` (recurso ja existente no codigo
original do Francisco, NUNCA usado ate agora) em torno do melhor ponto ja
encontrado (arquitetura `ae`, baseline 3000-4000h -- ver
docs/analise_pca_monitoramento_sistema.md, secoes "Reproducao literal" e
"LOEO").

MOTIVACAO (leitura de um especialista): com so 8-9 eventos rotulados,
qualquer "melhor configuracao de uma grade de milhares" esta sujeita a
overfitting -- o LOEO (leave-one-event-out) ja mostrou que o "6/9"
otimista cai pra "5/8" honesto fora da amostra. Em vez de continuar
caçando mais um hiperparametro global, a alavanca certa e a que o
proprio autor ja construiu e documentou mas nunca testamos:
`--per-family-thresholds` -- um limiar INDEPENDENTE por sinal (4 numeros,
nao 1), em vez de forcar temperatura/pressao_oleo/mancal_spread/selagem_z
a compartilhar o mesmo ponto de operacao. E a versao correta da ideia que
tentamos manualmente no v8 (que falhou na nossa reimplementacao) --
usando a infraestrutura JA VALIDADA dele, sem reescrever nada.

NAO edita `francisco_automl_clearml_original.py` -- importa as classes
dele (`DataBundle`, `BaselinePolicy`, `Trial`, `WalkForwardEvaluator`)
diretamente e monta uma grade pequena e cirurgica:
  - model="ae" (arquitetura vencedora da grade anterior)
  - baseline in {3000h, 4000h} (os 2 baselines vencedores ja encontrados)
  - limiar por sinal (percentil) in {99.5, 99.9, 99.97}^4 = 81 combinacoes
    -- ordem dos sinais: temperatura, pressao_oleo, mancal_spread, selagem_z
  - sustain in {30min, 2h}
  - ewma=30min, confirm=2, exclude_days=0, exclude_alarm_h=1h, min_alert=0min
    (fixos nos valores ja validados)
Total: 2 x 81 x 2 = 324 trials -- mas o `raw_scores()` dele cacheia por
(model, grid, baseline, exclude_days, exclude_alarm_h), entao so 2
ajustes caros de verdade (um por baseline); o resto (limiar/sustain) e
pos-processamento "quase gratuito", pela propria arquitetura de cache
dele.

Uso:
    PYTHONPATH=. python scripts/pca_monitoramento_sistema/v13_per_family_thresholds.py
"""
import itertools
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from clearml import Task

from francisco_automl_clearml_original import (
    DataBundle, BaselinePolicy, Trial, WalkForwardEvaluator, select_best,
)

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"
CLEARML_PROJECT_NAME = "TesteMLCab"
CLEARML_DOCKER_IMAGE = "python:3.12"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

EVAL_START, EVAL_END = "2024-05", "2026-04"
MAX_FP_PER_MONTH = 1.0
BASELINE_HOURS = [3000.0, 4000.0]
THRESHOLDS_PER_SIGNAL = [99.5, 99.9, 99.97]
SUSTAINS = ["30min", "2h"]
MODEL = "ae"
EWMA = "30min"
EXCLUDE_DAYS = 0
EXCLUDE_ALARM_H = 1.0
CONFIRM = 2
MIN_ALERT = "0min"
SIGNAL_ORDER = ["temperatura", "pressao_oleo", "mancal_spread", "selagem_z"]

task = Task.init(
    project_name=CLEARML_PROJECT_NAME,
    task_name="pca-walkforward::monitoramento_sistema_v13_per_family_thresholds",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(CLEARML_DOCKER_IMAGE)
task.set_packages(["pandas>=2.0", "numpy>=1.23", "pyarrow>=10", "scikit-learn>=1.1", "clearml"])

_config = {
    "eval_start": EVAL_START, "eval_end": EVAL_END, "max_fp_per_month": MAX_FP_PER_MONTH,
    "baseline_hours": BASELINE_HOURS, "thresholds_per_signal": THRESHOLDS_PER_SIGNAL,
    "sustains": SUSTAINS, "model": MODEL, "ewma": EWMA, "exclude_days": EXCLUDE_DAYS,
    "exclude_alarm_h": EXCLUDE_ALARM_H, "confirm": CONFIRM, "signal_order": SIGNAL_ORDER,
    "clearml_dataset_id": CLEARML_DATASET_ID,
}
task.connect(_config)

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

print("Carregando dados (DataBundle do codigo original, sem alteracao)...", flush=True)
bundle = DataBundle(CLEARML_DATASET_ID).load()
evaluator = WalkForwardEvaluator(bundle, EVAL_START, EVAL_END, MAX_FP_PER_MONTH)
print(f"[eventos] {len(evaluator.events)} eventos fisicos: "
      + ", ".join(f"{e['inicio']:%d/%m/%Y}" for e in evaluator.events), flush=True)

trials = []
for hours in BASELINE_HOURS:
    baseline = BaselinePolicy(window_hours=hours)
    for combo in itertools.product(THRESHOLDS_PER_SIGNAL, repeat=len(SIGNAL_ORDER)):
        for sustain in SUSTAINS:
            trials.append(Trial(
                model=MODEL, grid="2min", baseline=baseline, ewma=EWMA,
                exclude_days=EXCLUDE_DAYS, exclude_alarm_h=EXCLUDE_ALARM_H,
                threshold=combo, sustain=sustain, confirm=CONFIRM,
                min_alert=MIN_ALERT, threshold_kind="percentil",
            ))
print(f"[plano] {len(trials)} configuracoes (limiar independente por sinal, "
      f"ordem={SIGNAL_ORDER})", flush=True)

results = []
for i, trial in enumerate(trials, 1):
    try:
        result = evaluator.evaluate(trial)
    except Exception as exc:
        print(f"  [{i}/{len(trials)}] ERRO {type(exc).__name__}: {exc}", flush=True)
        continue
    results.append(result)
    if i % 20 == 0 or i == len(trials):
        print(f"  [{i}/{len(trials)}] processados...", flush=True)
    logger = task.get_logger()
    logger.report_scalar("deteccao", "eventos", result.eventos_detectados, i)
    logger.report_scalar("falso_positivo", "por_mes", result.fp_por_mes, i)

table = pd.json_normalize([asdict(r) for r in results])
best = select_best(results)
print(f"\n[fim] {len(results)} trials | aprovados (FP<={MAX_FP_PER_MONTH}/mes): "
      f"{sum(r.aprovado for r in results)}", flush=True)

# melhor por COBERTURA entre os aprovados (nao so o select_best padrao dele,
# que pesa robustez/distribuicao -- aqui queremos ver o teto real primeiro)
aprovados = table[table["aprovado"]]
if len(aprovados):
    top_cobertura = aprovados.sort_values(
        ["eventos_detectados", "fp_por_mes"], ascending=[False, True]
    ).head(10)
    print("\n=== TOP 10 POR COBERTURA (entre aprovados) ===")
    cols = ["eventos_detectados", "eventos_total", "lead_medio_h", "fp_por_mes",
            "trial.baseline_label", "trial.threshold", "trial.sustain"]
    print(top_cobertura[cols].to_string(index=False))

if best:
    print("\n=== MELHOR CONFIGURACAO (select_best padrao dele) ===")
    print(json.dumps({"trial": best.trial, "eventos": f"{best.eventos_detectados}/{best.eventos_total}",
                      "lead_medio_h": best.lead_medio_h, "fp_por_mes": best.fp_por_mes,
                      "aprovado": best.aprovado}, indent=2, ensure_ascii=False, default=str))

out_dir = os.path.join(OUTPUT_DIR, "resultado_v13_per_family_thresholds")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "automl_results_completo.csv.gz")
table.to_csv(csv_path, index=False, compression="gzip")
task.upload_artifact("automl_results", table)
if best:
    task.upload_artifact("best_trial", asdict(best))

print(f"\nresultados em {out_dir}", flush=True)
print("\nOK - fim do script v13 (per-family thresholds)", flush=True)
task.mark_completed(status_message="v13: per-family thresholds concluido com sucesso.")
task.close()
