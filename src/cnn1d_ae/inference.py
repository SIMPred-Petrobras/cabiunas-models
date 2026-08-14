"""Inferência de produção a partir do bundle salvo no treino.

O bundle (`best_model/inference_bundle.json`) carrega o scaler (center/scale),
clip bounds, threshold e parâmetros fixados no TREINO. Aplicar essas estatísticas
de treino — em vez de refitar na nova distribuição — é o que mantém o threshold
calibrado válido em dados novos.

Uso típico:
    from tensorflow import keras
    from src.cnn1d_ae.inference import load_bundle, score_dataframe
    bundle = load_bundle("…/best_model/inference_bundle.json")
    model  = keras.models.load_model("…/best_model/model.keras")
    scores = score_dataframe(model, bundle, df_novo)   # df indexado por tempo
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .sequences import make_sequences
from .scoring import reconstruction_mae_per_seq


def load_bundle(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transform_features(df: pd.DataFrame, bundle: dict, clip: bool = False) -> np.ndarray:
    """Aplica (x − center)/scale com as estatísticas de treino do bundle.

    clip=False (default) NÃO clipa os dados: clipar a entrada de scoring nos limites
    de treino removeria as anomalias fora-de-faixa (UNDER/drift) que precisam aparecer
    no erro de reconstrução. clip=True existe só para reproduzir a normalização de
    treino sobre dados normais.
    """
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para inferência: {missing}")
    out = df[cols].apply(pd.to_numeric, errors="coerce").copy()

    if clip:
        clip_bounds = bundle.get("clip_bounds") or {}
        for c in cols:
            if c in clip_bounds:
                low, high = clip_bounds[c]
                out[c] = out[c].clip(lower=low, upper=high)

    center = bundle.get("center", {})
    scale = bundle.get("scale", {})
    for c in cols:
        sc = float(scale.get(c, 1.0))
        if sc == 0:
            sc = 1.0
        out[c] = (out[c] - float(center.get(c, 0.0))) / sc

    return out.to_numpy(dtype=np.float32)


def score_dataframe(model, bundle: dict, df_sensor: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Reproduz o scoring por sequência em dados novos.

    df_sensor: DataFrame indexado por tempo com `feature_columns` (e, opcionalmente,
    `running_col` para a máscara de operação). Retorna um DataFrame por sequência
    com seq_end_time, mae_seq, is_anom_seq e operational_state (se aplicável).
    """
    time_steps = int(bundle["time_steps"])
    stride = max(1, int(bundle["stride"]))
    values = transform_features(df_sensor, bundle)
    x = make_sequences(values, time_steps, stride)
    mae = reconstruction_mae_per_seq(model, x, batch_size)
    anom = mae > float(bundle["threshold"])

    idx = df_sensor.index
    end_pos = (time_steps - 1) + np.arange(len(mae)) * stride
    end_pos = np.clip(end_pos, 0, len(idx) - 1)

    out = pd.DataFrame(
        {
            "seq_end_time": idx[end_pos],
            "mae_seq": mae,
            "is_anom_seq": anom.astype(int),
        }
    )

    running_col = bundle.get("running_col")
    if running_col and running_col in df_sensor.columns:
        rthr = float(bundle.get("running_threshold", 0.5))
        run = pd.to_numeric(df_sensor[running_col], errors="coerce").to_numpy()[end_pos]
        on = run > rthr
        out["operational_state"] = np.where(on, "on", "off")
        out.loc[~on, "is_anom_seq"] = 0  # suprime anomalia fora de operação

    return out


def score_production(model, bundle: dict, df_sensor: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Scoring de produção usando o bloco `production_alerting` do bundle finalizado
    (finalize_bundle.py): erro de reconstrução → EWMA(half_life) → comparação com o
    threshold ABSOLUTO calibrado → máscara de operação → debounce. É o ponto de
    operação que minimiza FP sem perder recall, pronto para streaming.
    """
    pa = bundle.get("production_alerting")
    if pa is None:
        raise ValueError("bundle sem 'production_alerting' — rode scripts/finalize_bundle.py antes")

    base = score_dataframe(model, bundle, df_sensor, batch_size)
    s = pd.Series(base["mae_seq"].to_numpy(), index=pd.DatetimeIndex(base["seq_end_time"]))

    dt = pd.Series(s.index).diff().dt.total_seconds().median()
    dt = dt if (dt and dt > 0) else 300.0
    hl_pts = max(1, int(round(float(pa["half_life_hours"]) * 3600.0 / dt)))
    ewma = s.ewm(halflife=hl_pts).mean()

    base = base.copy()
    base["health_ewma"] = ewma.to_numpy()
    alert = (ewma.to_numpy() >= float(pa["ewma_abs_threshold"])).astype(int)
    if "operational_state" in base.columns:
        alert = np.where(base["operational_state"].to_numpy() == "off", 0, alert)

    debounce_h = float(pa.get("debounce_hours", 0.0) or 0.0)
    if debounce_h > 0:
        # exige alerta sustentado por >= debounce_h (runs contíguos mais curtos são zerados)
        need = max(1, int(round(debounce_h * 3600.0 / dt)))
        a = alert.copy()
        i = 0
        while i < len(a):
            if a[i]:
                j = i
                while j + 1 < len(a) and a[j + 1]:
                    j += 1
                if (j - i + 1) < need:
                    a[i:j + 1] = 0
                i = j + 1
            else:
                i += 1
        alert = a
    base["alert"] = alert
    return base


# ---------------------------------------------------------------------------
# Graduação de confiança (pós-processamento; nada é suprimido nem atrasado)
# ---------------------------------------------------------------------------

def health_to_reference_rank(values, reference_quantiles) -> np.ndarray:
    """Mapeia EWMA absoluto → rank [0,1] contra a distribuição de CALIBRAÇÃO.

    A graduação foi calibrada em unidades de rank (`ewm(mae).rank(pct=True)`), que não
    existe em streaming — o rank de um ponto dependeria do futuro. Congelar a ECDF da
    janela de calibração no bundle resolve, pela mesma razão que se persiste o scaler:
    a régua tem de ser a do treino, não a do dado novo.
    """
    rq = np.asarray(reference_quantiles, dtype=float)
    probs = np.linspace(0.0, 1.0, len(rq))
    return np.clip(np.interp(np.asarray(values, dtype=float), rq, probs), 0.0, 1.0)


def grade_episodes(
    health: pd.Series,
    episodes,
    *,
    threshold: float,
    window_hours: float,
    dens_min: float,
    incl_min: float,
) -> pd.DataFrame:
    """Classifica cada episódio em `acao` ou `observacao` pela SUSTENTAÇÃO inicial.

    O alerta acende no mesmo instante de hoje — o lead é preservado. Em `onset +
    window_hours` o nível é revisto por dois números medidos na janela:

        densidade   fração dos pontos com health >= threshold
        inclinação  coeficiente linear do health por hora

    Por que não amplitude: o pico do score no falso positivo é tão alto quanto no
    evento real (AUC 0,44) — nada que olhe a altura separa. O que separa é a
    sustentação (densidade 0,73 / inclinação 0,74 em 6h).

    ⚠️ `health`, `threshold` e `incl_min` têm de estar na MESMA unidade. A calibração
    é em rank; em produção use `health_to_reference_rank()` antes de chamar.

    ⚠️ Episódio curto demais para medir a janela fica em `acao`: não se rebaixa o que
    não deu para avaliar. O custo relevante é TP rebaixado, não FP mantido.

    Devolve um DataFrame com onset, fim, densidade, inclinação e nível por episódio.
    """
    w = pd.Timedelta(hours=float(window_hours))
    rows = []
    for t0, t1 in episodes:
        seg = health[(health.index >= t0) & (health.index <= t0 + w)]
        if len(seg) < 3:
            rows.append(dict(onset=t0, fim=t1, densidade=np.nan, inclinacao=np.nan,
                             medido=False, nivel="acao"))
            continue
        x = (seg.index - seg.index[0]).total_seconds().to_numpy() / 3600.0
        y = seg.to_numpy(dtype=float)
        dens = float((y >= threshold).mean())
        incl = 0.0 if np.ptp(x) <= 0 else float(np.polyfit(x, y, 1)[0])
        nivel = "acao" if (dens >= dens_min and incl >= incl_min) else "observacao"
        rows.append(dict(onset=t0, fim=t1, densidade=dens, inclinacao=incl,
                         medido=True, nivel=nivel))
    return pd.DataFrame(rows, columns=["onset", "fim", "densidade", "inclinacao",
                                       "medido", "nivel"])


def grade_production_episodes(base: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Aplica a graduação sobre a saída de `score_production()`.

    Requer o bloco `grading` no bundle (`scripts/finalize_bundle.py --grading`).
    """
    g = bundle.get("grading")
    if g is None:
        raise ValueError("bundle sem 'grading' — rode finalize_bundle.py --grading antes")

    h_abs = pd.Series(base["health_ewma"].to_numpy(),
                      index=pd.DatetimeIndex(base["seq_end_time"]))
    health = pd.Series(health_to_reference_rank(h_abs.to_numpy(), g["reference_quantiles"]),
                       index=h_abs.index)

    alert = base["alert"].to_numpy().astype(bool)
    episodes, i = [], 0
    while i < len(alert):
        if alert[i]:
            j = i
            while j + 1 < len(alert) and alert[j + 1]:
                j += 1
            episodes.append((h_abs.index[i], h_abs.index[j]))
            i = j + 1
        else:
            i += 1

    return grade_episodes(health, episodes,
                          threshold=float(g["threshold_rank"]),
                          window_hours=float(g["window_hours"]),
                          dens_min=float(g["dens_min"]),
                          incl_min=float(g["incl_min"]))
