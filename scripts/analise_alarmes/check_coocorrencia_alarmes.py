"""Verifica se os alarmes do catalogo (temperatura, pressao, vibracao) sao
configurados por sensor individual mas disparam em cascata quando ha uma
falha fisica real -- ou seja, um unico evento fisico aciona muitos tags
distintos quase ao mesmo tempo, nao porque "e o mesmo alarme", mas porque
sao sintomas simultaneos da mesma causa raiz no mesmo equipamento.

Agrupa os eventos ACT do catalogo em "episodios" por proximidade temporal
(gap entre eventos consecutivos > WINDOW_MIN minutos = novo episodio) e
conta quantos tags DISTINTOS aparecem em cada episodio.

Ver ALARMES_POR_SENSOR_EFEITO_CASCATA.md (raiz do repo) para a
interpretacao e as implicacoes desse achado.

Uso:
    PYTHONPATH=. python scripts/analise_alarmes/check_coocorrencia_alarmes.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
from clearml import Dataset

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"  # "Cabiunas full 2024-2026 30s"

_dataset_root = Dataset.get(dataset_id=CLEARML_DATASET_ID).get_local_copy()
ALARM_CSV = os.path.join(_dataset_root, "alarmes_selecionados_turbina_a.csv")

alarm = pd.read_csv(ALARM_CSV)
if "Data da Ocorrência" in alarm.columns and "Data da Ocorrencia" not in alarm.columns:
    alarm["Data da Ocorrencia"] = alarm["Data da Ocorrência"]
if "Tag Alarme" in alarm.columns and "Tag" not in alarm.columns:
    alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrencia"], errors="coerce")
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
print("total eventos ACT:", len(alarm), " tags distintos:", alarm["Tag"].nunique())

for WINDOW_MIN in [30, 60, 180, 1440]:
    gaps = alarm["Data da Ocorrencia"].diff().dt.total_seconds().fillna(99999) / 60.0
    ep_id = (gaps > WINDOW_MIN).cumsum()
    grp = alarm.groupby(ep_id)
    n_tags_per_episode = grp["Tag"].nunique()
    print(f"\n=== janela de agrupamento: {WINDOW_MIN} min ===")
    print(f"total de episodios: {len(n_tags_per_episode)}")
    print(f"episodios com >=2 tags distintos: {(n_tags_per_episode>=2).sum()} "
          f"({(n_tags_per_episode>=2).mean()*100:.1f}%)")
    print(f"episodios com >=3 tags distintos: {(n_tags_per_episode>=3).sum()} "
          f"({(n_tags_per_episode>=3).mean()*100:.1f}%)")
    print(f"media de tags distintos por episodio: {n_tags_per_episode.mean():.2f}")
    print(f"maximo de tags distintos num episodio: {n_tags_per_episode.max()}")

WINDOW_MIN = 180
gaps = alarm["Data da Ocorrencia"].diff().dt.total_seconds().fillna(99999) / 60.0
alarm["ep_id"] = (gaps > WINDOW_MIN).cumsum()
biggest = alarm.groupby("ep_id")["Tag"].nunique().sort_values(ascending=False).head(8)
print(f"\n=== os 8 maiores episodios (janela {WINDOW_MIN}min), tags envolvidos ===")
for ep, ntags in biggest.items():
    sub = alarm[alarm["ep_id"] == ep]
    t0, t1 = sub["Data da Ocorrencia"].min(), sub["Data da Ocorrencia"].max()
    print(f"\nepisodio {ep}: {t0} -> {t1} ({ntags} tags distintos, {len(sub)} eventos)")
    print("  tags:", sorted(sub["Tag"].unique()))
