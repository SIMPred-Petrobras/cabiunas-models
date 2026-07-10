# =========================
# FILE: src/cnn1d_ae/model_card.py
# =========================
"""Geração de "model card" (.md) por modelo treinado.

Cada modelo (um por sensor no modo por-sensor, ou um por grupo no modo
multivariado) recebe um arquivo Markdown descrevendo como foi treinado e
configurado: dados de origem, pré-processamento, arquitetura/hiperparâmetros
encontrados pelo tuner, limiar de anomalia, máscara operacional e a avaliação
frente às falhas documentadas.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import PipelineConfig


def _fmt_dates(dates: List[pd.Timestamp]) -> str:
    if not dates:
        return "—"
    return ", ".join(pd.Timestamp(d).strftime("%d/%m/%Y %H:%M") for d in dates)


def _hp_table(best_hp: Optional[Dict[str, Any]]) -> str:
    if not best_hp:
        return "_Hiperparâmetros não disponíveis._"
    lines = ["| Hiperparâmetro | Valor |", "|---|---|"]
    for k, v in best_hp.items():
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def _figs_list(figs_dir: str) -> str:
    if not os.path.isdir(figs_dir):
        return "_Sem figuras._"
    names = sorted(n for n in os.listdir(figs_dir) if n.lower().endswith(".png"))
    if not names:
        return "_Sem figuras._"
    return "\n".join(f"- `figs/{n}`" for n in names)


def write_model_card(
    cfg: PipelineConfig,
    out_dirs: Dict[str, str],
    *,
    kind: str,
    name: str,
    sensors: List[str],
    best_hp: Optional[Dict[str, Any]],
    threshold: float,
    calibration_report: Dict[str, Any],
    failure_times: List[pd.Timestamp],
    n_train_seq: Optional[int] = None,
    n_features: Optional[int] = None,
    data_period: Optional[str] = None,
) -> str:
    """Escreve MODEL_CARD.md no diretório do modelo e retorna o caminho."""
    figs_dir = out_dirs.get("figs", os.path.join(out_dirs["root"], "figs"))
    eval_bits = {
        "n_alarms": calibration_report.get("n_alarms"),
        "hits": calibration_report.get("alarms_with_detected_anomaly_in_window"),
        "hit_rate": calibration_report.get("hit_rate"),
        "rate_per_day": calibration_report.get("anomaly_rate_points_per_day"),
    }

    op_mask = "habilitada" if cfg.ENABLE_OPERATIONAL_MASK else "desabilitada"
    ref_line = (f" (sensor de referência: `{cfg.OPERATIONAL_REF_SENSOR}`)"
                if cfg.ENABLE_OPERATIONAL_MASK and cfg.OPERATIONAL_REF_SENSOR else "")

    md = f"""# Model Card — {cfg.EQUIPMENT_ID or 'equipamento'} · {name}

Modelo **CNN-1D Autoencoder** para detecção de anomalias ({'univariado (por sensor)' if kind == 'sensor' else 'multivariado (grupo)'}).

## Equipamento e falha

- **Equipamento:** `{cfg.EQUIPMENT_ID or '—'}`
- **{'Sensor' if kind == 'sensor' else 'Grupo'}:** `{name}`
- **Sensores do modelo:** {', '.join(f'`{s}`' for s in sensors) if sensors else '—'}
- **Falha(s) documentada(s):** {_fmt_dates(failure_times)}
- **Descrição da falha:** {cfg.FAILURE_DESCRIPTION or '—'}

## Dados

- **Origem:** `{cfg.FEATHER_PATH or '(ClearML Dataset)'}`
- **Período coberto:** {data_period or '—'}
- **Fonte de treino:** `{cfg.TRAIN_SOURCE}`
- **Sequências de treino:** {n_train_seq if n_train_seq is not None else '—'} | **Canais (features):** {n_features if n_features is not None else '—'}

## Pré-processamento

- **Exclusão em torno de alarmes:** ±{cfg.EXCLUDE_MINUTES_AROUND_ALARM} min
- **Gaps longos excluídos do treino:** {cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN} (limite de interpolação: {cfg.INTERPOLATE_LIMIT})
- **Outliers:** modo `{cfg.OUTLIER_MODE}` (q_low={cfg.OUTLIER_Q_LOW}, q_high={cfg.OUTLIER_Q_HIGH}, mad_k={cfg.OUTLIER_MAD_K})
- **Normalização:** `{cfg.NORMALIZE_MODE}` (estatísticas apenas do treino)

## Janela e split

- **TIME_STEPS:** {cfg.TIME_STEPS} | **STRIDE:** {cfg.STRIDE}
- **Split:** `{cfg.SPLIT_MODE}` | **VAL_FRAC:** {cfg.VAL_FRAC} | **SHUFFLE_TRAIN:** {cfg.SHUFFLE_TRAIN}

## Busca de hiperparâmetros (KerasTuner)

- **MAX_TRIALS:** {cfg.MAX_TRIALS} | **EXECUTIONS_PER_TRIAL:** {cfg.EXECUTIONS_PER_TRIAL}
- **EPOCHS:** {cfg.EPOCHS} | **BATCH_SIZE:** {cfg.BATCH_SIZE} | **PATIENCE:** {cfg.PATIENCE}

### Melhores hiperparâmetros

{_hp_table(best_hp)}

## Limiar e regra de anomalia

- **Modo do limiar:** `{cfg.THRESH_MODE}` | **taxa-alvo:** {cfg.TARGET_ANOMALY_RATE}
- **Limiar (MAE):** {threshold:.6g}
- **Regra ponto:** `{cfg.POINT_RULE}` (window={cfg.POINT_WINDOW}, min_count={cfg.POINT_MIN_COUNT})
- **Máscara operacional:** {op_mask}{ref_line}

## Avaliação frente às falhas

- **Alarmes avaliados:** {eval_bits['n_alarms']}
- **Alarmes com anomalia detectada na janela:** {eval_bits['hits']}
- **Hit rate:** {eval_bits['hit_rate']}
- **Anomalias/dia:** {eval_bits['rate_per_day']}

## Artefatos gerados

- `best_model/model.keras` — modelo treinado
- `best_model/best_hyperparameters.json`
- `csv/calibration_report.json`, `csv/evaluation_alarm_hit_rate.json`
- `csv/sequence_scores_all.csv`, `csv/point_anomalies_all.csv`, `csv/trials_ranking.csv`

### Figuras

{_figs_list(figs_dir)}

---
_Card gerado automaticamente pela pipeline CNN1D-AE (Transpetro)._
"""
    card_path = os.path.join(out_dirs["root"], "MODEL_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(md)
    return card_path
