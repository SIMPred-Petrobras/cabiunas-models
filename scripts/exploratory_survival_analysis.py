"""EXP8 item 5 -- analise exploratoria de sobrevivencia (RUL "light").

Escopo deliberadamente limitado: NAO e um modelo de RUL pronto pra uso
operacional. Usa so o historico de timestamps dos alarmes reais de
TC382_03_A/T5_AVG_A (sem covariaveis de sensor) para caracterizar o padrao
de recorrencia -- tempo entre alarmes segue um processo com desgaste
(hazard crescente), aleatorio (hazard constante), ou clustering (hazard
decrescente)? Ver docs/analise_automl_exp7_planejamento.md (item 5) para o
porque do escopo reduzido: com so ~40 eventos no periodo OOS usado no
resto do EXP7/EXP8, um modelo de sobrevivencia com covariaveis (Cox, etc.)
teria risco real de overfitting e daria falsa confianca.

Uso: PYTHONPATH=. python scripts/exploratory_survival_analysis.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Dataset
from scipy import stats

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"
ALARM_CSV = "alarmes_selecionados_turbina_a.csv"
EVAL_SENSORS = ["TC382_03_A", "T5_AVG_A"]
DECLUSTER_MIN = pd.Timedelta(minutes=30)
OUTPUT_JSON = "exp8_survival_result.json"


def _ensure_clearml_config() -> None:
    if os.getenv("CLEARML_CONFIG_FILE"):
        return
    local_config = Path.cwd() / "clearml.conf"
    if local_config.is_file():
        os.environ["CLEARML_CONFIG_FILE"] = str(local_config)
        return
    raise RuntimeError(
        "ClearML nao esta configurado. Rode da raiz do projeto com clearml.conf "
        "ou exporte CLEARML_CONFIG_FILE=/caminho/para/clearml.conf"
    )


def main() -> None:
    _ensure_clearml_config()
    root = Path(Dataset.get(dataset_id=CLEARML_DATASET_ID).get_local_copy())
    alarm = pd.read_csv(root / ALARM_CSV)
    alarm["Tag"] = alarm.get("Tag Alarme", alarm.get("Tag"))
    alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
    alarm["Data da Ocorrencia"] = pd.to_datetime(
        alarm.get("Data da Ocorrência", alarm.get("Data da Ocorrencia")), errors="coerce"
    )

    times_raw = alarm.loc[alarm["Tag"].isin(EVAL_SENSORS), "Data da Ocorrencia"].dropna().sort_values().reset_index(drop=True)
    print(f"Total de alarmes brutos ({'+'.join(EVAL_SENSORS)}, onset, todo o historico): {len(times_raw)}")
    print(f"Periodo: {times_raw.min()} a {times_raw.max()}")

    # Decluster: T5_AVG_A e TC382_03_A costumam disparar quase juntos pro
    # mesmo evento fisico (gap de segundos a minutos) -- sem agrupar isso,
    # o ajuste fica dominado por uma massa artificial perto de zero e nao
    # reflete o padrao real de recorrencia.
    gaps_raw = times_raw.diff()
    new_event = gaps_raw.isna() | (gaps_raw > DECLUSTER_MIN)
    event_id = new_event.cumsum()
    times = times_raw.groupby(event_id).first().reset_index(drop=True)
    print(f"Apos decluster (gap>{DECLUSTER_MIN}): {len(times)} eventos distintos (de {len(times_raw)} linhas brutas)")

    gaps_h = times.diff().dropna().dt.total_seconds() / 3600.0
    print(f"\nIntervalos entre eventos consecutivos (n={len(gaps_h)}):")
    print(gaps_h.describe())

    shape, loc, scale = stats.weibull_min.fit(gaps_h.values, floc=0)
    print(f"\nAjuste Weibull: shape (k) = {shape:.3f} | scale (lambda) = {scale:.2f}h")
    if shape > 1.15:
        interp = ("hazard CRESCENTE -- padrao de desgaste: quanto mais tempo desde o "
                   "ultimo alarme, maior a chance do proximo vir logo (consistente com "
                   "degradacao acumulando).")
    elif shape < 0.85:
        interp = ("hazard DECRESCENTE -- alarmes tendem a vir em rajadas logo apos o "
                   "anterior, depois ficam mais raros (nao e o padrao classico de "
                   "desgaste continuo -- mais parecido com um processo autoexcitante).")
    else:
        interp = ("hazard aproximadamente CONSTANTE (shape~1, exponencial) -- processo "
                   "quase-aleatorio no tempo, sem sinal forte de acumulo de desgaste.")
    print(interp)

    exp_scale = gaps_h.mean()
    ks_exp = stats.kstest(gaps_h, "expon", args=(0, exp_scale))
    ks_weib = stats.kstest(gaps_h, "weibull_min", args=(shape, 0, scale))
    print(f"\nKS test vs exponencial (shape=1 fixo): D={ks_exp.statistic:.3f}, p={ks_exp.pvalue:.3f}")
    print(f"KS test vs Weibull ajustado: D={ks_weib.statistic:.3f}, p={ks_weib.pvalue:.3f}")

    print("\nSobrevivencia empirica x Weibull -- S(t) = P(intervalo > t horas):")
    for t in [1, 6, 12, 24, 48, 72, 168]:
        s_emp = float((gaps_h > t).mean())
        s_weib = float(stats.weibull_min.sf(t, shape, loc=0, scale=scale))
        print(f"  t={t:4d}h: empirica={s_emp:.3f} | Weibull={s_weib:.3f}")

    print("\nRisco condicional (dado X horas sem alarme, P(proximo em ate +Yh)):")
    cond_table = []
    for x in [6, 12, 24, 48]:
        s_x = stats.weibull_min.sf(x, shape, loc=0, scale=scale)
        if s_x <= 0:
            continue
        for y in [1, 6, 24]:
            s_xy = stats.weibull_min.sf(x + y, shape, loc=0, scale=scale)
            cond_prob = 1.0 - (s_xy / s_x)
            cond_table.append({"x_hours_since_last": x, "y_hours_window": y, "cond_prob": float(cond_prob)})
            print(f"  {x:3d}h sem alarme -> P(proximo em ate +{y}h) = {cond_prob:.1%}")

    result = {
        "n_alarmes_brutos": int(len(times_raw)),
        "n_eventos_declustered": int(len(times)),
        "decluster_window_min": DECLUSTER_MIN.total_seconds() / 60,
        "periodo_inicio": str(times.min()),
        "periodo_fim": str(times.max()),
        "gap_mediana_h": float(gaps_h.median()),
        "gap_media_h": float(gaps_h.mean()),
        "gap_min_h": float(gaps_h.min()),
        "gap_max_h": float(gaps_h.max()),
        "weibull_shape_k": float(shape),
        "weibull_scale_lambda_h": float(scale),
        "interpretacao": interp,
        "ks_vs_exponencial_D": float(ks_exp.statistic),
        "ks_vs_exponencial_p": float(ks_exp.pvalue),
        "ks_vs_weibull_D": float(ks_weib.statistic),
        "ks_vs_weibull_p": float(ks_weib.pvalue),
        "risco_condicional": cond_table,
    }
    Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
