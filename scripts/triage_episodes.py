#!/usr/bin/env python3
"""Triagem automática de episódios (is_anom_point contínuo) em categorias
físicas, para não tratar todo "far" como ruído homogêneo. Baseado na inspeção
visual de 5 episódios far (B-6511502A, B-4064A) que revelou 3 mecanismos
distintos:

  1. GLITCH_SENSOR      — salto instantâneo implausível no sinal bruto (poucos
                          pontos), tipicamente junto de off_curto/transiente.
  2. MUDANCA_REGIME     — o sinal se estabiliza num patamar NOVO depois do
                          episódio (não volta à baseline anterior) — o modelo
                          não reconhece o novo regime operacional.
  3. PRECURSOR_PARADA   — sinal sustentado, sem salto nem mudança de patamar,
                          seguido de desligamento em poucas horas — candidato
                          a evento real não documentado.
  4. SUSTENTADO_SEM_CAUSA — MAE elevado por tempo considerável sem nenhuma
                          mudança visível no sinal bruto nem parada depois —
                          fica para revisão manual (pode ser real ou gap do
                          modelo).
  5. TRANSIENTE_CURTO   — sobra: episódios curtos/fracos sem nenhum dos
                          padrões acima — candidatos mais fracos a falso
                          positivo "comum".

Não decide uma regra de corte final — classifica, para que o if/else (se
vier a existir) trate cada classe com a lógica adequada (filtro de glitch,
threshold por regime, revisão manual, etc.), em vez de um corte único.

Uso:
    PYTHONPATH=. python scripts/triage_episodes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.transpetro.io import _is_long_format, _pivot_long
from src.cnn1d_ae.pipeline import parse_failure_dates

FEATHER_BASE = Path("/home/dvar/transpetro/PROJETO")
UNI_DIR = Path("resultados/experimento_1_mascara_v3/Uni_sensor")
MULT_DIR = Path("resultados/experimento_1_mascara_v3/Mult_sensor")
OUT_DIR = Path("analysis")

PRE_POST_WINDOW = pd.Timedelta(hours=2)      # baseline antes/depois do episódio
SHUTDOWN_LOOKAHEAD = pd.Timedelta(hours=6)   # janela p/ checar parada após o episódio
GLITCH_STEP_MULT = 8.0                       # salto >= N x o "passo típico" = glitch
REGIME_SHIFT_MULT = 4.0                      # |baseline_after-before| >= N x passo típico = regime


def _load_raw_cached(cache: dict, eq: str, feather_path: str) -> pd.DataFrame:
    if eq in cache:
        return cache[eq]
    df = pd.read_feather(FEATHER_BASE / feather_path)
    if _is_long_format(df):
        df = _pivot_long(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next(c for c in ("Timestamp", "Data Hora", "data_datetime") if c in df.columns)
        df = df.set_index(pd.to_datetime(df[tcol], errors="coerce"))
    df = df.sort_index()
    cache[eq] = df
    return df


def load_context(eq_dir: Path, raw_cache: dict) -> dict | None:
    calib_p = eq_dir / "csv" / "calibration_report.json"
    cfg_p = eq_dir / "csv" / "run_config.json"
    pt_p = eq_dir / "csv" / "point_anomalies_all.csv"
    if not (calib_p.exists() and cfg_p.exists() and pt_p.exists()):
        return None
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    sensor = calib.get("sensor") or calib.get("target_sensor")
    if not sensor:
        return None

    df_raw_all = _load_raw_cached(raw_cache, eq_dir.name, cfg["FEATHER_PATH"])
    if sensor not in df_raw_all.columns:
        return None
    raw = pd.to_numeric(df_raw_all[sensor], errors="coerce")

    pt = pd.read_csv(pt_p)
    pt["data_datetime"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
    pt = pt.set_index("data_datetime")
    state = pt["operational_state"] if "operational_state" in pt.columns else None

    # "passo típico" do sinal em operação normal — mediana do |diff| absoluto,
    # restrita a trechos com estado "on" (senão os longos trechos desligados,
    # quase planos, puxam o passo típico pra perto de zero e qualquer ruído
    # normal de operação passa a parecer "mudança de regime").
    if state is not None:
        on_times = state.index[state.reindex(state.index).eq("on")]
        raw_on = raw.reindex(raw.index.intersection(on_times))
    else:
        raw_on = raw
    diffs_nz = raw_on.diff().abs()
    diffs_nz = diffs_nz[diffs_nz > 0]
    typical_step = float(diffs_nz.median()) if len(diffs_nz) else np.nan
    robust_scale = float(diffs_nz.quantile(0.95)) if len(diffs_nz) else np.nan

    return {"raw": raw, "typical_step": typical_step, "robust_scale": robust_scale,
            "state": state, "fails": parse_failure_dates(cfg.get("FAILURE_DATE", ""))}


def classify_episode(ep: pd.Series, ctx: dict) -> dict:
    raw, state = ctx["raw"], ctx["state"]
    step = ctx["typical_step"] if np.isfinite(ctx.get("typical_step", np.nan)) else ctx["robust_scale"]
    t0, t1 = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])

    before = raw[(raw.index >= t0 - PRE_POST_WINDOW) & (raw.index < t0)]
    after = raw[(raw.index > t1) & (raw.index <= t1 + PRE_POST_WINDOW)]
    during = raw[(raw.index >= t0) & (raw.index <= t1)]
    # Janela alargada p/ detectar glitch: o salto que causa o MAE elevado pode
    # preceder o início do episódio (is_anom_point só liga após acumular k_of_window).
    lead_in = raw[(raw.index >= t0 - pd.Timedelta(hours=1)) & (raw.index <= t1)]

    base_before = float(before.median()) if len(before) else np.nan
    base_after = float(after.median()) if len(after) else np.nan
    peak_dev = float((during - base_before).abs().max()) if len(during) and np.isfinite(base_before) else np.nan
    # maior salto ponto-a-ponto (episódio + 1h antes, ver comentário acima)
    max_step_in_ep = float(lead_in.diff().abs().max()) if len(lead_in) > 1 else 0.0
    level_shift = abs(base_after - base_before) if np.isfinite(base_before) and np.isfinite(base_after) else np.nan

    st_during = state.reindex(during.index).fillna("on") if state is not None and len(during) else None
    frac_off_like = float(st_during.isin(["off_curto", "off_longo", "transiente"]).mean()) if st_during is not None and len(st_during) else 0.0
    st_lead = state.reindex(lead_in.index).fillna("on") if state is not None and len(lead_in) else None
    frac_off_like_lead = float(st_lead.isin(["off_curto", "off_longo", "transiente"]).mean()) if st_lead is not None and len(st_lead) else 0.0

    shutdown_after = False
    if state is not None:
        st_after = state[(state.index > t1) & (state.index <= t1 + SHUTDOWN_LOOKAHEAD)]
        shutdown_after = bool(st_after.isin(["off_curto", "off_longo"]).any())
    # "Parada real" = terminou desligado (não apenas mudou de patamar operando).
    ended_stopped = shutdown_after or (np.isfinite(base_after) and np.isfinite(step) and step > 0
                                       and base_after <= max(0.0, base_before - REGIME_SHIFT_MULT * step)
                                       and base_after < 1e-6 + (base_before * 0.2 if np.isfinite(base_before) else 0))

    is_glitch = (np.isfinite(step) and step > 0 and
                 max_step_in_ep >= GLITCH_STEP_MULT * step and
                 frac_off_like_lead >= 0.15)
    is_precursor = (not is_glitch and ended_stopped and
                    ep["duration_min"] >= 30 and frac_off_like < 0.5)
    is_regime = (not is_glitch and not is_precursor and np.isfinite(level_shift) and np.isfinite(step) and step > 0 and
                 level_shift >= REGIME_SHIFT_MULT * step)
    is_sustained_unexplained = (not is_glitch and not is_regime and not is_precursor and
                                ep["duration_min"] >= 30)

    if is_glitch:
        cls = "glitch_sensor"
    elif is_precursor:
        cls = "precursor_parada"
    elif is_regime:
        cls = "mudanca_regime"
    elif is_sustained_unexplained:
        cls = "sustentado_sem_causa"
    else:
        cls = "transiente_curto"

    return {
        "trig_class": cls,
        "base_before": round(base_before, 4) if np.isfinite(base_before) else None,
        "base_after": round(base_after, 4) if np.isfinite(base_after) else None,
        "peak_dev": round(peak_dev, 4) if np.isfinite(peak_dev) else None,
        "max_step_in_ep": round(max_step_in_ep, 4),
        "typical_step": round(step, 4) if np.isfinite(step) else None,
        "level_shift": round(level_shift, 4) if np.isfinite(level_shift) else None,
        "frac_off_like": round(frac_off_like, 3),
        "shutdown_after": shutdown_after,
    }


def process(mode: str, base_dir: Path, episodes_csv: Path) -> pd.DataFrame:
    df_ep = pd.read_csv(episodes_csv, parse_dates=["start", "end"])
    raw_cache: dict = {}
    ctx_cache: dict = {}
    rows = []
    for _, ep in df_ep.iterrows():
        eq = ep["equip"]
        if eq not in ctx_cache:
            ctx_cache[eq] = load_context(base_dir / eq, raw_cache)
        ctx = ctx_cache[eq]
        if ctx is None:
            continue
        info = classify_episode(ep, ctx)
        rows.append({**ep.to_dict(), **info})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label}", ""]
    if df.empty:
        return lines + ["_sem dados_\n"]
    lines.append("| Classe | far | near_10d | near_48h | dur mediana (min) | peak_ratio mediana |")
    lines.append("|---|---|---|---|---|---|")
    for cls, g in df.groupby("trig_class"):
        far_n = (g["category"] == "far").sum()
        n10 = (g["category"] == "near_10d").sum()
        n48 = (g["category"] == "near_48h").sum()
        lines.append(f"| {cls} | {far_n} | {n10} | {n48} | {g['duration_min'].median():.1f} | {g['peak_ratio'].median():.2f} |")
    lines.append("")
    return lines


def main() -> None:
    df_uni = process("uni", UNI_DIR, OUT_DIR / "episodes_uni.csv")
    df_mult = process("mult", MULT_DIR, OUT_DIR / "episodes_mult.csv")

    df_uni.to_csv(OUT_DIR / "episodes_uni_classified.csv", index=False)
    df_mult.to_csv(OUT_DIR / "episodes_mult_classified.csv", index=False)

    lines = [
        "# Triagem automática de episódios — glitch vs regime vs precursor vs indefinido",
        "",
        "Classifica cada episódio (`is_anom_point` contínuo) por mecanismo físico provável,",
        "usando o sinal bruto ao redor (baseline antes/depois, salto máximo dentro do",
        "episódio, estado operacional, se há parada logo depois). Critérios:",
        "",
        f"- **glitch_sensor**: salto ≥{GLITCH_STEP_MULT:g}x o passo típico do sinal E ≥20% do "
        "episódio em estado off/transiente.",
        f"- **mudanca_regime**: sinal se estabiliza num patamar novo (|baseline_depois−antes| "
        f"≥{REGIME_SHIFT_MULT:g}x o passo típico) sem ser glitch.",
        "- **precursor_parada**: episódio ≥30min, sem glitch/regime, seguido de parada em até "
        f"{SHUTDOWN_LOOKAHEAD.total_seconds()/3600:.0f}h.",
        "- **sustentado_sem_causa**: ≥30min sem nenhum dos padrões acima — revisão manual.",
        "- **transiente_curto**: sobra (curto/fraco, sem padrão claro).",
        "",
    ]
    lines += summarize(df_uni, "Uni_sensor (univariado)")
    lines += summarize(df_mult, "Mult_sensor (multivariado, canal-alvo)")
    lines += summarize(pd.concat([df_uni, df_mult], ignore_index=True), "Pooled (uni + mult)")

    (OUT_DIR / "EPISODE_TRIAGE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] analysis/EPISODE_TRIAGE.md | episodes_uni_classified.csv ({len(df_uni)}) | "
         f"episodes_mult_classified.csv ({len(df_mult)})")


if __name__ == "__main__":
    main()
