"""CONTROLE: o pipeline do Francisco, sobre a NOSSA exportacao, reproduz os
numeros publicados dele? Sem isso, injetar o nosso sinal nao significa nada.

Alvo (notebook 10, secao 5 -- configuracao de producao):
  pca | 2min | b3000h | ewma 2h | p99,9 | sustain 30min | confirm 2
  -> 6/8 eventos · lead 18,3 h · 0,94 FP/mes · 22,0 h/mes · 350,5 dias avaliados

Usa DataBundle direto em vez de CachedBundle: este ultimo baixa o CSV do ClearML
no __init__, e o que queremos e justamente apontar para o nosso arquivo local.
"""
import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from pathlib import Path
import automl_clearml as automl
from cabiunas_pdm.replay import DetectionReplay

CSV = Path("dados_locais/sensores_brutos_2025_2026_30s.csv")
CONFIG = dict(model="pca", grid="2min", exclude_days=0, exclude_alarm_h=1.0,
              ewma="2h", threshold=99.9, sustain="30min", confirm=2, min_alert="0min")

t0 = time.time()
bundle = automl.DataBundle(automl.DATASET_ID, CSV, False, False).load()
print(f"[tempo] carga: {time.time()-t0:.0f} s", flush=True)

r = DetectionReplay(bundle=bundle, cache_dir=Path("cache_controle")).run(
    baseline=automl.BaselinePolicy(window_hours=3000), **CONFIG)
m = r.metrics
print("\n" + "=" * 78)
print("CONTROLE DE REPRODUCAO -- pipeline dele, dado nosso")
print("=" * 78)
alvo = {"eventos_detectados": "6 de 8", "lead_medio_h": 18.3, "fp_por_mes": 0.94,
        "fp_horas_por_mes": 22.0, "dias_avaliados": 350.5, "episodios_totais": 20}
for k, v in alvo.items():
    print(f"  {k:22s} nosso={str(m.get(k)):>10s}   publicado={v}")
print(f"\n[tempo] total: {time.time()-t0:.0f} s")
