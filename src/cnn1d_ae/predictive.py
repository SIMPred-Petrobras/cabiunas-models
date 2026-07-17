"""Camada preditiva (Fase 2): converte o erro de reconstrucao (deteccao pontual)
em ALERTA ANTECIPADO via EWMA + persistencia. Mede a capacidade preditiva pela
curva lead-time x falso-alarme/dia contra incidentes genuinos.

Esta logica foi validada localmente (scripts/ae_predictive_layer.py) com:
- AUC nivel-janela 0.95 nos incidentes genuinos
- ~72% recall a ~0.1 FA/dia com horizonte de 72h
- lead time multi-dia para a maioria dos incidentes capturados
"""
from __future__ import annotations
from typing import Iterable, Mapping, Optional
import numpy as np
import pandas as pd


def extract_incidents(
    df_alarm: pd.DataFrame,
    priorities: Optional[Iterable[str]] = None,
    conditions: Optional[Iterable[str]] = None,
    incident_gap_hours: float = 4.0,
) -> pd.DatetimeIndex:
    """Extrai timestamps de incidentes genuinos a partir do df_alarm.

    Filtros aplicados (em ordem):
    1. condicao do alarme != 'OK' (descarta clears que nao sao onsets reais);
    2. opcionalmente filtra por condicao (ex: ["HI","HIHI"] exclui UNDER que
       sao desligamentos globais, nao anomalias termicas);
    3. opcionalmente filtra por prioridade;
    4. dedup: onsets separados por <= incident_gap_hours sao o mesmo incidente.
    """
    if "Data da Ocorrencia" not in df_alarm.columns:
        return pd.DatetimeIndex([])
    ts = pd.to_datetime(df_alarm["Data da Ocorrencia"], errors="coerce")
    mask = ts.notna()
    if "Condição do Alarme" in df_alarm.columns:
        mask &= df_alarm["Condição do Alarme"].astype(str) != "OK"
    if conditions:
        mask &= df_alarm["Condição do Alarme"].astype(str).isin(list(conditions))
    if priorities and "Prioridade" in df_alarm.columns:
        mask &= df_alarm["Prioridade"].astype(str).isin(list(priorities))
    sel = ts[mask].sort_values().reset_index(drop=True)
    if sel.empty:
        return pd.DatetimeIndex([])
    g = (sel.diff().dt.total_seconds() / 3600.0 > float(incident_gap_hours)).cumsum()
    inc = sel.groupby(g).min().reset_index(drop=True)
    return pd.DatetimeIndex(inc)


def compute_health_index_ewma(
    score_seq: np.ndarray,
    seq_running_frac: np.ndarray,
    half_life_hours: float,
    dt_seconds: float,
) -> np.ndarray:
    """EWMA do score por sequencia, NaN durante desligamento, ffill carrega o ultimo valor.

    alpha derivado de meia-vida (mais interpretavel que alpha "cego"):
        alpha = 1 - 0.5 ** (dt / half_life)
    """
    score = np.asarray(score_seq, dtype=float)
    rf = np.asarray(seq_running_frac, dtype=float)
    masked = np.where(rf >= 0.999, score, np.nan)
    if half_life_hours <= 0:
        alpha = 1.0
    else:
        alpha = 1.0 - 0.5 ** (float(dt_seconds) / 3600.0 / float(half_life_hours))
    out = np.empty(len(masked), dtype=float)
    prev = np.nan
    for i, v in enumerate(masked):
        if np.isnan(v):
            out[i] = prev
        else:
            prev = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return pd.Series(out).ffill().bfill().to_numpy()


def _detect_episodes(
    alert_idx: np.ndarray,
    t_end_seconds: np.ndarray,
    debounce_seconds: float,
):
    """Agrupa indices contiguos de alerta em episodios (inicio_s, fim_s)."""
    if alert_idx.size == 0:
        return []
    episodes = []
    cur = [alert_idx[0]]
    for j in alert_idx[1:]:
        if t_end_seconds[j] - t_end_seconds[cur[-1]] <= debounce_seconds:
            cur.append(j)
        else:
            episodes.append((float(t_end_seconds[cur[0]]), float(t_end_seconds[cur[-1]])))
            cur = [j]
    episodes.append((float(t_end_seconds[cur[0]]), float(t_end_seconds[cur[-1]])))
    return episodes


def compute_predictive_curve(
    health_ewma: np.ndarray,
    seq_running_full: np.ndarray,
    t_end_seconds: np.ndarray,
    incident_seconds: np.ndarray,
    horizon_hours: float,
    debounce_hours: float = 8.0,
    n_threshold_steps: int = 40,
    sigma_y_min: float = 0.5,
    sigma_y_max: float = 5.0,
) -> pd.DataFrame:
    """Para uma malha de thresholds mean+y·σ, computa (recall, fa_per_day, lead_h, n_episodes).

    A grade varre y em [sigma_y_min, sigma_y_max]: threshold = μ + y·σ do EWMA em ON.
    Interpretação direta (SPC / Shewhart): y=2 dispara no top ~2.3% do regime normal.

    - Incidente "pego" se ha alerta ATIVO em [t_inc - horizon, t_inc].
    - Lead = (t_inc - primeiro alerta na janela) em horas.
    - Falso-alarme = episodio cuja faixa nao intersecta nenhuma janela pre-incidente.
    """
    ew = np.asarray(health_ewma, dtype=float)
    runfull = np.asarray(seq_running_full, dtype=bool)
    t_s = np.asarray(t_end_seconds, dtype=float)
    inc_s = np.asarray(incident_seconds, dtype=float)
    if ew.size == 0 or t_s.size == 0:
        return pd.DataFrame()
    span_days = max((t_s.max() - t_s.min()) / 86400.0, 1e-9)
    valid_ew = ew[runfull]
    if valid_ew.size == 0:
        return pd.DataFrame()
    mu  = float(valid_ew.mean())
    q25, q75 = np.percentile(valid_ew, [25, 75])
    sig = max(float(q75 - q25) / 1.35, 1e-9)  # IQR-based σ robusto a outliers de anomalia
    y_vals = np.linspace(float(sigma_y_min), float(sigma_y_max), int(n_threshold_steps))
    grid   = mu + y_vals * sig
    H = float(horizon_hours) * 3600.0
    deb = float(debounce_hours) * 3600.0
    rows = []
    for y, thr in zip(y_vals, grid):
        alert = (ew >= thr) & runfull
        idx = np.where(alert)[0]
        episodes = _detect_episodes(idx, t_s, deb)
        alert_s = t_s[idx]
        hits = 0
        leads = []
        for ti in inc_s:
            w = alert_s[(alert_s >= ti - H) & (alert_s <= ti)]
            if w.size:
                hits += 1
                leads.append((ti - w.min()) / 3600.0)
        recall = hits / len(inc_s) if len(inc_s) else 0.0
        fa = 0
        for (s0, s1) in episodes:
            useful = bool((((inc_s - H) <= s1) & (inc_s >= s0)).any()) if inc_s.size else False
            if not useful:
                fa += 1
        rows.append(dict(
            threshold=float(thr),
            y_sigma=float(y),
            recall=float(recall),
            fa_per_day=float(fa / span_days),
            median_lead_hours=float(np.median(leads)) if leads else 0.0,
            n_episodes=int(len(episodes)),
        ))
    return pd.DataFrame(rows)


def compute_predictive_curve_per_sensor(
    per_sensor_health: np.ndarray,
    seq_running_full: np.ndarray,
    t_end_seconds: np.ndarray,
    incident_seconds: np.ndarray,
    horizon_hours: float,
    debounce_hours: float = 8.0,
    n_threshold_steps: int = 40,
    sigma_y_min: float = 0.5,
    sigma_y_max: float = 5.0,
) -> pd.DataFrame:
    """Curva preditiva para o backend per_sensor (OR-de-sigma entre N sensores).

    Cada sensor tem o SEU PROPRIO threshold = μ_j + y·σ_j (SPC / Shewhart).
    Alerta dispara se QUALQUER sensor cruzar o seu proprio threshold (operacao OR).
    A grade varre y em [sigma_y_min, sigma_y_max]: interpretavel e escala-livre.

    per_sensor_health: shape (n_seq, n_sensors), ewma do MAE de cada sensor.
    Retorna DataFrame com cols (threshold, y_sigma, recall, fa_per_day,
    median_lead_hours, n_episodes). 'threshold' = y usado (escala-livre p/ OR).
    """
    health = np.asarray(per_sensor_health, dtype=float)
    runfull = np.asarray(seq_running_full, dtype=bool)
    t_s = np.asarray(t_end_seconds, dtype=float)
    inc_s = np.asarray(incident_seconds, dtype=float)
    if health.ndim != 2 or health.shape[0] != t_s.size:
        return pd.DataFrame()
    span_days = max((t_s.max() - t_s.min()) / 86400.0, 1e-9)
    n_seq, n_sens = health.shape

    # μ e σ por sensor (calculados em running)
    mu_per  = np.array([float(health[runfull, j].mean()) if runfull.any() else 0.0
                        for j in range(n_sens)])
    sig_per = np.array([
        max(float(np.percentile(health[runfull, j], 75) - np.percentile(health[runfull, j], 25)) / 1.35, 1e-9)
        if runfull.any() else 1e-9
        for j in range(n_sens)
    ])

    y_vals = np.linspace(float(sigma_y_min), float(sigma_y_max), int(n_threshold_steps))
    H = float(horizon_hours) * 3600.0
    deb = float(debounce_hours) * 3600.0
    rows = []
    for y in y_vals:
        per_sensor_thr = mu_per + y * sig_per
        # alerta OR: qualquer sensor acima do seu threshold
        alert = ((health >= per_sensor_thr[None, :]).any(axis=1)) & runfull
        idx = np.where(alert)[0]
        episodes = _detect_episodes(idx, t_s, deb)
        alert_s = t_s[idx]
        hits = 0
        leads = []
        for ti in inc_s:
            w = alert_s[(alert_s >= ti - H) & (alert_s <= ti)]
            if w.size:
                hits += 1
                leads.append((ti - w.min()) / 3600.0)
        recall = hits / len(inc_s) if len(inc_s) else 0.0
        fa = 0
        for (s0, s1) in episodes:
            useful = bool((((inc_s - H) <= s1) & (inc_s >= s0)).any()) if inc_s.size else False
            if not useful:
                fa += 1
        rows.append(dict(
            threshold=float(y),          # y é escala-livre; thresholds absolutos variam por sensor
            y_sigma=float(y),
            recall=float(recall),
            fa_per_day=float(fa / span_days),
            median_lead_hours=float(np.median(leads)) if leads else 0.0,
            n_episodes=int(len(episodes)),
        ))
    return pd.DataFrame(rows)


def pick_operating_point(
    curve: pd.DataFrame,
    fa_budget_per_day: float,
) -> Optional[Mapping[str, float]]:
    """Escolhe o threshold com maior recall sob a restricao fa/dia <= budget."""
    if curve is None or curve.empty:
        return None
    feasible = curve[curve["fa_per_day"] <= float(fa_budget_per_day)]
    if len(feasible):
        row = feasible.loc[feasible["recall"].idxmax()]
    else:
        # Sem ponto factivel: devolve o de maior recall (o operador decide ampliar o budget).
        row = curve.loc[curve["recall"].idxmax()]
    return {k: float(v) for k, v in row.to_dict().items()}
