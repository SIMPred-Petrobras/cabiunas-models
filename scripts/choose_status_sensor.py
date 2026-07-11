#!/usr/bin/env python3
"""Escolha do sensor de status (ligado/desligado) por equipamento.

Motivado pela doc do time PdM (DOC/): para bombas, o melhor indicador de
"parada" é a PRESSÃO DE DESCARGA (bomba parada não gera pressão), não a
corrente — que pode estar zerada/ruidosa (ex.: B-0302C tem canais de motor
zerados).

Para cada equipamento e cada sensor candidato, mede quão bem ele separa
operação de parada e, principalmente, se ele está ALTO quando o sinal-alvo
(vibração/temperatura) está alto — ou seja, se preserva o sinal da falha.

Gera analysis/ESCOLHA_SENSOR_STATUS.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.transpetro.io import _is_long_format, _pivot_long

EQUIPS = ["B-0302C", "B-24001B", "B-3403C", "B-402E", "B-4064A", "B-4703.24001B",
          "B-5401A", "B-5501B", "B-6511502A", "B-8801C", "B-8802B", "B-90001A"]
OUT_MD = Path("analysis/ESCOLHA_SENSOR_STATUS.md")

# Palavras que indicam bom candidato a sensor de status (processo).
CAND_PATTERNS = [
    ("descarga", r"press.*desc|desc.*bomba"),
    ("velocidade", r"velocidad"),
    ("potencia_ativa", r"pot.*ativ"),
    ("vazao", r"vaz"),
    ("corrente", r"^corrente$|corrente el|corrente m"),
]


def load_wide(cfg: dict) -> pd.DataFrame | None:
    fp = cfg.get("FEATHER_PATH")
    if not fp or not Path(fp).exists():
        return None
    df = pd.read_feather(fp)
    if _is_long_format(df):
        df = _pivot_long(df)
    tcol = next((c for c in ("Timestamp", "Data Hora", "data_datetime") if c in df.columns), None)
    if tcol is None:
        return None
    df = df.set_index(pd.to_datetime(df[tcol], errors="coerce")).sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def find_candidates(cols: list[str]) -> dict:
    out = {}
    for label, pat in CAND_PATTERNS:
        rx = re.compile(pat, re.I)
        hit = next((c for c in cols if rx.search(c)), None)
        if hit:
            out[label] = hit
    return out


def score_candidate(s: pd.Series, target: pd.Series) -> dict:
    """Mede: %parado (<=q05*), e a preservação — dos pontos de alvo alto (p90),
    quantos têm o candidato acima de um limiar operacional (q40)."""
    s = s.dropna()
    if len(s) < 100:
        return {}
    # limiar operacional: metade da mediana dos valores 'altos' (bimodal 0/op)
    hi = s[s > s.quantile(0.6)]
    op_thr = 0.5 * float(hi.median()) if len(hi) and hi.median() > 0 else float(s.quantile(0.5))
    pct_parado = float((s <= op_thr).mean() * 100)
    joined = pd.concat([s.rename("ref"), target.rename("tgt")], axis=1).dropna()
    if joined.empty:
        return {"op_thr": round(op_thr, 2), "pct_parado": round(pct_parado, 1), "preserva": None}
    hi_t = joined[joined["tgt"] > joined["tgt"].quantile(0.90)]
    preserva = float((hi_t["ref"] > op_thr).mean() * 100) if len(hi_t) else None
    return {"op_thr": round(op_thr, 2), "pct_parado": round(pct_parado, 1),
            "preserva": None if preserva is None else round(preserva, 1)}


def main() -> None:
    lines = [
        "# Escolha do sensor de status (ligado/desligado) por equipamento",
        "",
        "Baseado na doc do time PdM (DOC/): para bombas o melhor indicador de parada é a",
        "**pressão de descarga** (bomba parada não gera pressão). Aqui comparamos candidatos por:",
        "",
        "- **%parado** = fração do tempo abaixo do limiar operacional estimado.",
        "- **preserva** = dos pontos com sinal-alvo alto (vibração/temp no p90), quantos têm o",
        "  candidato em operação. **Alto = o sensor NÃO apaga o sinal da falha** (é o que queremos).",
        "",
        "| Equip | Alvo | Candidato | %parado | preserva sinal |",
        "|---|---|---|---|---|",
    ]
    best_choice = {}
    for eq in EQUIPS:
        cfg = json.loads(Path(f"configs/transpetro/{eq}.json").read_text(encoding="utf-8"))
        grp = (cfg.get("SENSOR_GROUPS") or [{}])[0]
        target = grp.get("target_sensor")
        df = load_wide(cfg)
        if df is None or target not in (df.columns if df is not None else []):
            lines.append(f"| `{eq}` | {target} | — | — | (dados/alvo indisponível) |")
            continue
        cands = find_candidates(list(df.columns))
        tgt_series = df[target]
        rows = []
        for label, col in cands.items():
            sc = score_candidate(df[col], tgt_series)
            if sc:
                rows.append((label, col, sc))
        # Melhor = maior preservação do sinal (desempata por menor %parado)
        rows_ok = [r for r in rows if r[2].get("preserva") is not None]
        rows_ok.sort(key=lambda r: (-(r[2]["preserva"]), r[2]["pct_parado"]))
        for i, (label, col, sc) in enumerate(rows_ok or rows):
            mark = " ⭐" if (rows_ok and i == 0) else ""
            lines.append(f"| `{eq}` | {target if i==0 else ''} | {col} ({label}){mark} | "
                         f"{sc['pct_parado']}% | {sc.get('preserva','—')}% |")
        if rows_ok:
            best_choice[eq] = {"sensor": rows_ok[0][1], "op_thr": rows_ok[0][2]["op_thr"],
                               "preserva": rows_ok[0][2]["preserva"]}

    lines += [
        "",
        "## Decisão (sensor de status por equipamento)",
        "",
        "Regra: escolher o candidato que **melhor preserva o sinal** (⭐). Para bombas, tende a",
        "ser a pressão de descarga; corrente fica como fallback (e falha onde está zerada).",
        "",
        "```json",
        json.dumps(best_choice, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    Path("analysis/status_sensor_choice.json").write_text(
        json.dumps(best_choice, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] {OUT_MD}")


if __name__ == "__main__":
    main()
