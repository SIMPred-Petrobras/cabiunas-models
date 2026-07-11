#!/usr/bin/env python3
"""Análise da máscara operacional: o sensor de referência (OPERATIONAL_REF_SENSOR)
com OFF_ABS_THRESHOLD fixo=5.0 marca a máquina como 'desligada' a maior parte do
tempo em vários equipamentos, apagando anomalias — inclusive no período da falha.

Este script mede, por equipamento:
  - Distribuição do sensor de referência (quantis).
  - % do tempo classificado como OFF pelo limiar atual (5.0).
  - Comportamento do ref NO PERÍODO DA FALHA (janela -10d..+2d): a máquina estava
    ligada? qual a mediana do ref ali?
  - Bimodalidade (desligado≈0 vs faixa operacional) e um limiar sugerido (vale).

Gera analysis/ANALISE_MASCARA_OPERACIONAL.md com o diagnóstico e a recomendação.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.transpetro.io import _is_long_format, _pivot_long
from src.cnn1d_ae.pipeline import parse_failure_dates

EQUIPS = ["B-0302C", "B-24001B", "B-3403C", "B-402E", "B-4064A", "B-4703.24001B",
          "B-5401A", "B-5501B", "B-6511502A", "B-8801C", "B-8802B", "B-90001A"]
OUT_MD = Path("analysis/ANALISE_MASCARA_OPERACIONAL.md")


def load_ref_series(cfg: dict):
    ref = cfg.get("OPERATIONAL_REF_SENSOR")
    fp = cfg.get("FEATHER_PATH")
    if not ref or not fp or not Path(fp).exists():
        return None, ref
    df = pd.read_feather(fp)
    if _is_long_format(df):
        df = _pivot_long(df)
    tcol = None
    for c in ("Timestamp", cfg.get("TIME_COL"), "data_datetime", "Data Hora"):
        if c and c in df.columns:
            tcol = c
            break
    if ref not in df.columns or tcol is None:
        return None, ref
    s = pd.Series(pd.to_numeric(df[ref], errors="coerce").values,
                  index=pd.to_datetime(df[tcol], errors="coerce"))
    return s.dropna(), ref


def suggest_threshold(s: pd.Series) -> float:
    """Limiar sugerido: separa o modo 'desligado' (≈0) do modo operacional.
    Heurística robusta = 15% da mediana dos valores acima do 60º percentil
    (faixa claramente operacional). Cai para o 5º percentil se degenerar."""
    hi = s[s > s.quantile(0.60)]
    if len(hi) and hi.median() > 0:
        thr = 0.15 * float(hi.median())
        if thr > 0:
            return round(thr, 2)
    return round(float(s.quantile(0.05)), 2)


def analyze(eq: str) -> dict:
    cfg = json.loads(Path(f"configs/transpetro/{eq}.json").read_text(encoding="utf-8"))
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))
    r = {"eq": eq, "ref": cfg.get("OPERATIONAL_REF_SENSOR"),
         "mask": cfg.get("ENABLE_OPERATIONAL_MASK"), "fails": len(fails)}
    if not cfg.get("ENABLE_OPERATIONAL_MASK"):
        r["nota"] = "máscara desativada"
        return r
    s, ref = load_ref_series(cfg)
    if s is None:
        r["nota"] = f"ref '{ref}' indisponível"
        return r
    r["med"] = round(float(s.median()), 1)
    r["q75"] = round(float(s.quantile(0.75)), 1)
    r["max"] = round(float(s.max()), 1)
    r["pct_off_5"] = round(float((s <= 5.0).mean() * 100), 1)
    r["thr_sug"] = suggest_threshold(s)
    r["pct_off_sug"] = round(float((s <= r["thr_sug"]).mean() * 100), 1)

    # Comportamento perto da falha
    near = []
    for f in fails:
        w = s[(s.index >= f - pd.Timedelta(days=10)) & (s.index <= f + pd.Timedelta(days=2))]
        if len(w):
            near.append({
                "on_pct_5": round(float((w > 5.0).mean() * 100), 1),
                "med": round(float(w.median()), 1),
            })
    r["near_fail"] = near
    return r


def main() -> None:
    rows = [analyze(eq) for eq in EQUIPS]

    lines = [
        "# Análise da máscara operacional — diagnóstico e recomendação",
        "",
        "**Contexto:** a máscara zera qualquer anomalia fora do estado `on`. O estado é",
        "definido pelo sensor `OPERATIONAL_REF_SENSOR`: fica `off` quando `ref ≤ OFF_ABS_THRESHOLD`",
        "(hoje **fixo = 5.0** para todos). Onde a corrente opera em faixa baixa, 5.0 marca a",
        "máquina como desligada a maior parte do tempo e apaga o sinal da falha.",
        "",
        "## Tabela por equipamento",
        "",
        "| Equip | Ref | Mediana | q75 | %OFF@5.0 | Limiar sugerido | %OFF sugerido | Máquina LIGADA perto da falha? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "med" not in r:
            lines.append(f"| `{r['eq']}` | {r.get('ref','—')} | — | — | — | — | — | {r.get('nota','')} |")
            continue
        nf = r.get("near_fail") or []
        nf_txt = "; ".join(f"{x['on_pct_5']}% on (med={x['med']})" for x in nf) or "—"
        lines.append(
            f"| `{r['eq']}` | {r['ref']} | {r['med']} | {r['q75']} | **{r['pct_off_5']}%** | "
            f"{r['thr_sug']} | {r['pct_off_sug']}% | {nf_txt} |")

    # Classificação de recomendação
    rec_recal, rec_ok, rec_intermit = [], [], []
    for r in rows:
        if "med" not in r:
            continue
        nf = r.get("near_fail") or []
        on_near = max((x["on_pct_5"] for x in nf), default=0)
        if r["pct_off_5"] < 30:
            rec_ok.append(r["eq"])
        elif on_near >= 40:
            # máquina LIGADA perto da falha mas máscara marca muito off no geral -> recalibrar
            rec_recal.append(r["eq"])
        else:
            rec_intermit.append(r["eq"])

    lines += [
        "",
        "## Leitura",
        "",
        "- **`%OFF@5.0`** = fração do tempo que o limiar atual marca como desligado.",
        "- **`Máquina LIGADA perto da falha?`** = % dos pontos com `ref > 5` na janela −10d..+2d da falha.",
        "  Se está alto, a máquina operava no período da falha e a máscara está **apagando sinal válido**.",
        "",
        "## Recomendação por grupo",
        "",
        f"- **Recalibrar limiar (máscara atrapalha):** {', '.join(f'`{e}`' for e in rec_recal) or '—'}",
        "  → máquina operando perto da falha, mas limiar 5.0 corta demais. Usar limiar sugerido.",
        f"- **Máquina genuinamente intermitente:** {', '.join(f'`{e}`' for e in rec_intermit) or '—'}",
        "  → fica realmente desligada a maior parte do tempo; o sinal só existe nas janelas `on` (curtas).",
        "  Recalibrar ajuda pouco; considerar avaliar só trechos operacionais e/ou relaxar a regra de ponto.",
        f"- **Máscara já ok (máquina quase sempre ligada):** {', '.join(f'`{e}`' for e in rec_ok) or '—'}",
        "",
        "## Decisão adotada",
        "",
        "Trocar `OFF_ABS_THRESHOLD` fixo=5.0 pelo **limiar sugerido por equipamento** (≈15% da",
        "mediana operacional), que se adapta à faixa real de cada corrente. Para os intermitentes,",
        "isso ao menos deixa de marcar como `off` os trechos de operação real. Equipamentos já ok",
        "permanecem praticamente inalterados.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] salvo em {OUT_MD}")

    # Também emite um JSON com os limiares sugeridos p/ configurar os configs.
    thr_map = {r["eq"]: r["thr_sug"] for r in rows if "thr_sug" in r}
    Path("analysis/mask_thresholds_suggested.json").write_text(
        json.dumps(thr_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] limiares sugeridos: analysis/mask_thresholds_suggested.json -> {thr_map}")


if __name__ == "__main__":
    main()
