"""Valida o gancho antes de varrer: o sinal externo entra, na posicao esperada,
e participa do voto? Sem isso a varredura mediria o nada."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np, pandas as pd
import automl_clearml as automl
from cabiunas_pdm.replay import DetectionReplay

CSV = Path("dados_locais/sensores_brutos_2025_2026_30s.csv")
POL = automl.BaselinePolicy(window_hours=3000)
BASE_CFG = dict(model="pca", grid="2min", exclude_days=0, exclude_alarm_h=1.0,
                ewma="2h", sustain="30min", confirm=2, min_alert="0min")

print("ORDEM DOS SINAIS (define o indice do limiar por sinal)")
print("=" * 74)
print(f"  familias  : {list(automl.FAMILIES)}")
print(f"  derivados : ['mancal_spread', 'selagem_z', '{automl.SINAL_EXTERNO_NOME}']")
ordem = list(automl.FAMILIES) + ["mancal_spread", "selagem_z", automl.SINAL_EXTERNO_NOME]
print(f"  ordem     : {ordem}")
print(f"  indice do nosso sinal: {ordem.index(automl.SINAL_EXTERNO_NOME)}\n")

for rot, externo, thr in (("4 sinais (controle)", False, 99.9),
                          ("5 sinais, vb em p99,9", True, 99.9),
                          ("5 sinais, vb em p70", True, (99.9, 99.9, 99.9, 99.9, 70.0))):
    b = automl.DataBundle(automl.DATASET_ID, CSV, False, False).load()
    b.sinal_externo = externo
    r = DetectionReplay(bundle=b, cache_dir=Path(f"cache_val_{int(externo)}_{str(thr)[:6]}")
                        ).run(baseline=POL, threshold=thr, **BASE_CFG)
    m = r.metrics
    print(f"  {rot:24s} {m['eventos_detectados']}/8 · lead {m['lead_medio_h']:5.1f} h · "
          f"{m['fp_por_mes']:.2f} FP/mes · {m['fp_horas_por_mes']:.1f} h/mes · "
          f"{m['episodios_totais']} eps")
