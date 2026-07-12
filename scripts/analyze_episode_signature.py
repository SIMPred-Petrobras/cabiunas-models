#!/usr/bin/env python3
"""Assinatura de episódios de anomalia (magnitude + duração) perto da falha
real vs em outros pontos da série — para investigar se dá pra usar duração
sustentada (não só valor de threshold pontual) como critério anti-falso-positivo.

v2: episódios extraídos do `is_anom_point` de point_anomalies_all.csv — ou
seja, JÁ pós k_of_window (POINT_WINDOW/POINT_MIN_COUNT), que suaviza quedas
breves abaixo do threshold. A v1 usava cruzamento bruto de mae_seq>threshold,
que gerava episódios de 0-1 min em toda parte (ruído de limiar, sem relação
com sinal sustentado) e não separava nada — ver histórico no fim do arquivo.

Para cada equipamento (Uni_sensor e Mult_sensor):
  - Episódio = trecho contínuo com is_anom_point==1.
  - Magnitude = MAE (mae_seq, alinhado por tempo de fim de janela) no trecho.
  - Categoria: near_48h (±2d da falha) | near_10d (2-10d) | far (>10d).

Saída:
  analysis/episodes_uni.csv, analysis/episodes_mult.csv (uma linha por episódio)
  analysis/EPISODE_SIGNATURE_STUDY.md (sumário estatístico, sem decidir regra)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn1d_ae.pipeline import parse_failure_dates

UNI_DIR = Path("resultados/experimento_1_mascara_v3/Uni_sensor")
MULT_DIR = Path("resultados/experimento_1_mascara_v3/Mult_sensor")
OUT_DIR = Path("analysis")


def load_equip(eq_dir: Path) -> dict | None:
    pt_p = eq_dir / "csv" / "point_anomalies_all.csv"
    seq_p = eq_dir / "csv" / "sequence_scores_all.csv"
    calib_p = eq_dir / "csv" / "calibration_report.json"
    cfg_p = eq_dir / "csv" / "run_config.json"
    if not (pt_p.exists() and seq_p.exists() and calib_p.exists() and cfg_p.exists()):
        return None

    df_pt = pd.read_csv(pt_p, usecols=["data_datetime", "is_anom_point"])
    df_pt["data_datetime"] = pd.to_datetime(df_pt["data_datetime"], errors="coerce")
    df_pt = df_pt.dropna(subset=["data_datetime"]).sort_values("data_datetime").reset_index(drop=True)

    df_seq = pd.read_csv(seq_p, usecols=["seq_start_time", "mae_seq"])
    df_seq["seq_start_time"] = pd.to_datetime(df_seq["seq_start_time"], errors="coerce")
    df_seq = df_seq.dropna(subset=["seq_start_time"]).sort_values("seq_start_time").reset_index(drop=True)

    # Guarda de qualidade: equipamentos com sensor quase constante (ex.: B-5401A,
    # já descartado pelo time PdM por instrumentação insuficiente) têm MAE
    # degenerada (IQR≈0) e o threshold calibra colado no "chão" numérico —
    # qualquer MAE normal vira um peak_ratio absurdo (milhares de x). Excluir
    # da análise agregada evita que esse artefato domine as conclusões.
    q25, q75 = df_seq["mae_seq"].quantile([.25, .75])
    if (q75 - q25) <= 1e-12:
        print(f"[SKIP] {eq_dir.name}: mae_seq degenerada (IQR≈0, threshold no chão numérico) — "
              f"excluído da análise (dado de baixa qualidade, ver ESTUDO_E_DECISOES_TRANSPETRO.md)")
        return None

    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))

    # A MAE de uma sequência corresponde ao FIM da janela (convenção usada em
    # mask_anomaly_seq_by_operational_state) — desloca o índice em
    # (TIME_STEPS-1) passos de tempo (mediana do dt observado) para alinhar
    # com o timestamp de ponto correspondente.
    dt_med = df_pt["data_datetime"].diff().dt.total_seconds().median()
    dt_med = dt_med if np.isfinite(dt_med) and dt_med > 0 else 60.0
    time_steps = int(cfg.get("TIME_STEPS", 60))
    shift = pd.Timedelta(seconds=dt_med * (time_steps - 1))
    mae_by_end = pd.Series(df_seq["mae_seq"].values, index=df_seq["seq_start_time"] + shift).sort_index()
    mae_by_end = mae_by_end[~mae_by_end.index.duplicated(keep="last")]

    return {"pt": df_pt, "mae_by_end": mae_by_end,
            "threshold": float(calib.get("threshold", np.nan)), "fails": fails}


def extract_episodes(eq: str, mode: str, data: dict) -> list[dict]:
    df_pt, mae_by_end, thr, fails = data["pt"], data["mae_by_end"], data["threshold"], data["fails"]
    anom = df_pt["is_anom_point"].values.astype(bool)
    if not anom.any():
        return []

    grp = (anom != np.r_[False, anom[:-1]]).cumsum()
    episodes = []
    for gid, idx in pd.Series(grp).groupby(grp).groups.items():
        idx = list(idx)
        if not anom[idx[0]]:
            continue
        sub = df_pt.iloc[idx]
        t0, t1 = sub["data_datetime"].iloc[0], sub["data_datetime"].iloc[-1]
        duration_min = (t1 - t0).total_seconds() / 60.0
        n_points = len(sub)

        mae_win = mae_by_end[(mae_by_end.index >= t0 - pd.Timedelta(minutes=5)) &
                             (mae_by_end.index <= t1 + pd.Timedelta(minutes=5))]
        if len(mae_win) == 0:
            nearest_pos = mae_by_end.index.get_indexer([t0], method="nearest")[0]
            mae_win = mae_by_end.iloc[[nearest_pos]] if nearest_pos >= 0 else pd.Series(dtype=float)
        peak = float(mae_win.max()) if len(mae_win) else np.nan
        mean_mae = float(mae_win.mean()) if len(mae_win) else np.nan

        if fails:
            dist_days = min([abs((t0 - f).total_seconds()) / 86400.0 for f in fails] +
                            [0.0 for f in fails if t0 <= f <= t1])
            cat = "near_48h" if dist_days <= 2 else ("near_10d" if dist_days <= 10 else "far")
        else:
            dist_days, cat = np.nan, "sem_falha_doc"

        episodes.append({
            "equip": eq, "mode": mode, "start": t0, "end": t1,
            "duration_min": round(duration_min, 1), "n_points": n_points,
            "peak_mae": peak, "peak_ratio": round(peak / thr, 3) if np.isfinite(peak) and thr else None,
            "mean_mae": mean_mae, "mean_ratio": round(mean_mae / thr, 3) if np.isfinite(mean_mae) and thr else None,
            "dist_to_failure_days": round(dist_days, 2) if np.isfinite(dist_days) else None,
            "category": cat,
        })
    return episodes


def summarize(df_ep: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label}", ""]
    if df_ep.empty:
        lines.append("_Sem episódios._\n")
        return lines
    lines.append(f"Total de episódios: {len(df_ep)} | equipamentos com episódio: {df_ep['equip'].nunique()}")
    lines.append("")
    lines.append("| Categoria | n | duração mediana (min) | duração p75 | peak_ratio mediana | peak_ratio p75 |")
    lines.append("|---|---|---|---|---|---|")
    for cat, g in df_ep.groupby("category"):
        lines.append(
            f"| {cat} | {len(g)} | {g['duration_min'].median():.1f} | {g['duration_min'].quantile(.75):.1f} | "
            f"{g['peak_ratio'].median():.2f} | {g['peak_ratio'].quantile(.75):.2f} |"
        )
    lines.append("")

    near = df_ep[df_ep["category"].isin(["near_48h", "near_10d"])]
    far = df_ep[df_ep["category"] == "far"]
    if len(near) and len(far):
        lines.append(f"**Perto da falha (near_48h+near_10d), n={len(near)}:** "
                     f"duração mediana={near['duration_min'].median():.1f}min "
                     f"(p25={near['duration_min'].quantile(.25):.1f}, p75={near['duration_min'].quantile(.75):.1f}) | "
                     f"peak_ratio mediana={near['peak_ratio'].median():.2f}")
        lines.append(f"**Longe da falha (far), n={len(far)}:** "
                     f"duração mediana={far['duration_min'].median():.1f}min "
                     f"(p25={far['duration_min'].quantile(.25):.1f}, p75={far['duration_min'].quantile(.75):.1f}) | "
                     f"peak_ratio mediana={far['peak_ratio'].median():.2f}")
        dur_p25_near = near["duration_min"].quantile(.25)
        peak_p25_near = near["peak_ratio"].quantile(.25)
        far_below_dur = (far["duration_min"] < dur_p25_near).mean() * 100
        far_below_peak = (far["peak_ratio"] < peak_p25_near).mean() * 100
        far_below_both = ((far["duration_min"] < dur_p25_near) | (far["peak_ratio"] < peak_p25_near)).mean() * 100
        lines.append("")
        lines.append(f"Se usarmos como corte o p25 dos episódios `near` (duração≥{dur_p25_near:.1f}min "
                     f"OU peak_ratio≥{peak_p25_near:.2f}): **{far_below_both:.1f}%** dos episódios `far` "
                     f"seriam eliminados (duração isolada eliminaria {far_below_dur:.1f}%, "
                     f"peak_ratio isolado {far_below_peak:.1f}%).")
    lines.append("")
    return lines


def summarize_per_equip(df_ep: pd.DataFrame, label: str) -> list[str]:
    """Quebra near vs far por equipamento — a média pooled pode esconder que
    só alguns equipamentos têm sinal (os outros nem têm episódio near)."""
    lines = [f"### {label} — por equipamento", "",
             "| Equip | n_near | dur mediana near | peak_ratio near | n_far | dur mediana far | peak_ratio far |",
             "|---|---|---|---|---|---|---|"]
    for eq, g in df_ep.groupby("equip"):
        near = g[g["category"].isin(["near_48h", "near_10d"])]
        far = g[g["category"] == "far"]
        lines.append(
            f"| `{eq}` | {len(near)} | {near['duration_min'].median() if len(near) else '—'} | "
            f"{near['peak_ratio'].median() if len(near) else '—'} | {len(far)} | "
            f"{far['duration_min'].median() if len(far) else '—':.1f} | {far['peak_ratio'].median() if len(far) else '—':.2f} |"
            if len(far) else
            f"| `{eq}` | {len(near)} | {near['duration_min'].median() if len(near) else '—'} | "
            f"{near['peak_ratio'].median() if len(near) else '—'} | 0 | — | — |"
        )
    lines.append("")
    return lines


def main() -> None:
    all_uni, all_mult = [], []
    for eq_dir in sorted(UNI_DIR.iterdir()):
        if eq_dir.is_dir() and eq_dir.name.startswith("B-"):
            data = load_equip(eq_dir)
            if data:
                all_uni += extract_episodes(eq_dir.name, "uni", data)
    for eq_dir in sorted(MULT_DIR.iterdir()):
        if eq_dir.is_dir() and eq_dir.name.startswith("B-"):
            data = load_equip(eq_dir)
            if data:
                all_mult += extract_episodes(eq_dir.name, "mult", data)

    df_uni = pd.DataFrame(all_uni)
    df_mult = pd.DataFrame(all_mult)
    OUT_DIR.mkdir(exist_ok=True)
    df_uni.to_csv(OUT_DIR / "episodes_uni.csv", index=False)
    df_mult.to_csv(OUT_DIR / "episodes_mult.csv", index=False)

    lines = [
        "# Estudo de assinatura de episódio (magnitude + duração) — near falha vs far",
        "",
        "v2 — episódios extraídos de `is_anom_point` (já pós k_of_window, suaviza",
        "quedas breves abaixo do threshold), não do cruzamento bruto de mae_seq>threshold.",
        "`near_48h` = episódio dentro de ±2 dias da falha documentada; `near_10d` = 2–10",
        "dias; `far` = >10 dias (candidato a falso positivo). Não decide regra nova — só",
        "caracteriza o padrão.",
        "",
    ]
    lines += summarize(df_uni, "Uni_sensor (univariado) — pooled")
    lines += summarize_per_equip(df_uni, "Uni_sensor (univariado)")
    lines += summarize(df_mult, "Mult_sensor (multivariado, canal-alvo) — pooled")
    lines += summarize_per_equip(df_mult, "Mult_sensor (multivariado, canal-alvo)")

    (OUT_DIR / "EPISODE_SIGNATURE_STUDY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] analysis/EPISODE_SIGNATURE_STUDY.md | episodes_uni.csv ({len(df_uni)}) | episodes_mult.csv ({len(df_mult)})")


if __name__ == "__main__":
    main()
