#!/usr/bin/env python3
"""
map_alarm_tags_to_columns.py
Casa cada tag da base de alarmes com a coluna de curva correspondente em
`sensores_brutos_2025_2026_30s.csv`, onde os instrumentos aparecem com nome de
historiador (`954005_624_PI_0319`) em vez do nome de alarme (`PI_6240319_AL`).

Por que não dá para casar só pelo número: o número de loop se repete entre famílias
de instrumento diferentes. `PAL_6240315` (Pressão Baixa Gás Comb.) e `TI_6240315`
(Temperatura Ar de Exaustão) dividem o 0315 e apontam para colunas DIFERENTES
(`PI_0315` e `TI_0315`). O mesmo em 0305 (`TAH`→`TI_0305`, `PDAH`→`PDIT_0305`) e em
0307 (`PI_6240307_AL`→`PI_0307`, `TAH_6240307`→`TI_0307`). Casar por número sozinho
erra em 4 tags — silenciosamente, o que contaminaria recall e FA sem dar erro.

A regra usada é família de instrumento + número de loop:
    prefixo do alarme → família (P | PD | T)  →  coluna da mesma família com o número

A `Descrição Alarme` NÃO entra no casamento (é texto livre, com grafias inconsistentes
para a mesma tag); entra no relatório como conferência humana e para classificar o
alarme em `falha` (defeito do próprio instrumento) vs `processo` (evento do
equipamento). Essa distinção importa: as tags de maior volume da base são de FALHA
de transmissor, não de excursão de processo.

Uso:
    PYTHONPATH=. python scripts/map_alarm_tags_to_columns.py
    PYTHONPATH=. python scripts/map_alarm_tags_to_columns.py --out configs/calibracao_v12_pressao/tag_column_map.csv
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata

import pandas as pd

ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
RAW_CSV = "../dados/sensores_brutos_2025_2026_30s.csv"
DEFAULT_OUT = "configs/calibracao_v12_pressao/tag_column_map.csv"

# Colunas de historiador: 954005_<area>_<instrumento>_<loop>
COL_RE = re.compile(r"^(\d+)_(\d+)_([A-Z]+)_(\d+)$")

# Famílias por instrumento da COLUNA
COL_FAMILY = {"PI": "P", "PDI": "PD", "PDIT": "PD", "TI": "T"}


def normalize(text: str) -> str:
    """Remove acentos e baixa a caixa — as descrições vêm com grafia inconsistente."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def alarm_family(tag: str) -> str | None:
    """Família de instrumento a partir do prefixo da tag de alarme.

    Ordem importa: 'PD*' tem de ser testado antes de 'P*', senão PDAL cai em pressão
    simples. 'FL_PT' é falha de transmissor de pressão → família P.
    """
    t = tag.upper()
    if t.startswith("FL_PT"):
        return "P"
    prefix = re.match(r"^([A-Z]+)", t)
    if not prefix:
        return None
    p = prefix.group(1)
    if p.startswith("PD"):
        return "PD"
    if p.startswith("P"):
        return "P"
    if p.startswith("T"):
        return "T"
    return None


def loop_number(tag: str) -> str | None:
    """Número de loop de 7 dígitos (ex.: 6240315 em PAL_6240315, FL_PT6240339)."""
    m = re.search(r"(\d{7})", tag)
    return m.group(1) if m else None


def alarm_kind(descriptions: list[str]) -> str:
    """Classifica em 'falha' (defeito do instrumento) vs 'processo' (evento real).

    Alarmes de 'Falha PI/PDI/PDT/PT' indicam transmissor com defeito — são úteis para
    detecção de falha de sensor, mas NÃO são excursão do equipamento. Misturar os dois
    no ground-truth mediria duas coisas diferentes com o mesmo número.
    """
    joined = " ".join(normalize(d) for d in descriptions)
    return "falha" if "falha" in joined else "processo"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alarm_csv", default=ALARM_CSV)
    p.add_argument("--raw_csv", default=RAW_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    cols = pd.read_csv(args.raw_csv, nrows=1).columns.tolist()
    # índice (família, número) → coluna
    by_key: dict[tuple[str, str], list[str]] = {}
    for c in cols:
        m = COL_RE.match(c)
        if not m:
            continue
        _, area, instr, loop = m.groups()
        fam = COL_FAMILY.get(instr)
        if fam is None:
            continue
        by_key.setdefault((fam, area + loop), []).append(c)

    # Os 17 sensores já modelados aparecem com o próprio nome de tag como coluna
    # (TC382_03_A, TV_351X_A, T5_AVG_A) — não seguem o padrão de historiador.
    direct = {c for c in cols if not COL_RE.match(c)}

    alarms = pd.read_csv(args.alarm_csv)
    alarms["_t"] = pd.to_datetime(alarms["Data da Ocorrência"], errors="coerce")

    rows = []
    for tag, g in alarms.groupby("Tag Alarme"):
        fam, num = alarm_family(tag), loop_number(tag)
        if tag in direct:
            cands = [tag]          # sensor já modelado: a coluna é a própria tag
        else:
            cands = by_key.get((fam, num), []) if (fam and num) else []
        if len(cands) > 1:
            # não deve acontecer com este dataset; se acontecer, é ambiguidade real
            # e precisa de decisão humana — não escolher em silêncio.
            status = "AMBIGUO"
        elif cands:
            status = "ok"
        else:
            status = "sem_curva"
        conds = g["Condição do Alarme"].astype(str).str.upper()
        rows.append({
            "tag": tag,
            "familia": fam or "",
            "loop": num or "",
            "coluna": cands[0] if len(cands) == 1 else "",
            "status": status,
            "ja_modelado": tag in direct,
            "tipo": alarm_kind(g["Descrição Alarme"].dropna().unique().tolist()),
            "eventos_total": len(g),
            "onsets": int((conds != "OK").sum()),
            "primeiro": g["_t"].min(),
            "ultimo": g["_t"].max(),
            "condicoes": "|".join(sorted(conds.unique())),
            "descricao": g["Descrição Alarme"].dropna().mode().iloc[0]
            if g["Descrição Alarme"].notna().any() else "",
        })

    df = pd.DataFrame(rows).sort_values(["status", "tipo", "onsets"],
                                        ascending=[True, True, False])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 44)
    show = ["tag", "coluna", "familia", "tipo", "onsets", "condicoes", "descricao"]
    novas = df[(df["status"] == "ok") & ~df["ja_modelado"]]
    print(f"\n=== CANDIDATAS NOVAS ({len(novas)} tags — curva existe, nunca modelada) ===")
    print(novas[show].to_string(index=False))
    for st in ["AMBIGUO", "sem_curva"]:
        sub = df[df["status"] == st]
        if sub.empty:
            continue
        print(f"\n=== {st} ({len(sub)} tags) ===")
        print(sub[show].to_string(index=False))

    mapped = df[df["status"] == "ok"]
    print(f"\n{len(mapped)}/{len(df)} tags casadas "
          f"({int(mapped['ja_modelado'].sum())} já modeladas, {len(novas)} novas).")
    for tipo in ["processo", "falha"]:
        s = novas[novas["tipo"] == tipo]
        print(f"  novas/{tipo:8}: {len(s):>2} tags, {int(s['onsets'].sum()):>4} onsets, "
              f"{s['coluna'].nunique()} colunas distintas")
    print(f"\nGravado em {args.out}")

    orphan = [c for c in cols if COL_RE.match(c) and
              c not in set(df["coluna"]) - {""}]
    if orphan:
        print(f"\nColunas sem tag de alarme casada ({len(orphan)}): {orphan}")


if __name__ == "__main__":
    main()
