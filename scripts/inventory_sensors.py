#!/usr/bin/env python3
"""
inventory_sensors.py
Inventário único de "que sensores dá para usar", separando duas perguntas que
costumam ser confundidas:

  ALVO   — o AE é treinado nele e o resultado é medido contra alarmes. Exige
           ground-truth: incidentes rotulados COM A MÁQUINA LIGADA. Sem isso o
           sensor pode ser treinado, mas o recall dele é inverificável.
  ENTRADA— entra como canal de contexto de outro modelo (multivariado ou
           CONTEXT_COLS). NÃO exige rótulo nenhum — só exige curva com cobertura.

Um sensor sem rótulo não é inútil: é inútil como alvo. Essa distinção muda o que
fazer com os 10 sensores de vibração e com os instrumentos auxiliares.

Regra de contagem do ground-truth (protocolo honesto do projeto):
  - só HI/HIHI para termopar/vibração (UNDER = trip de planta, LOLO = ver memória
    `hihi-only-recall-corrige-t5`);
  - só CFN aprovado no portão para os instrumentos novos (`validate_pressure_labels`);
  - só com equipamento LIGADO (`off-period-alarms-fora-de-escopo`).

Uso:
    PYTHONPATH=. python scripts/inventory_sensors.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

CSV_LONGO = "../dados/sensores_2024h2_2025_2026_30s.csv"   # jun/2024→abr/2026
CSV_BRUTOS = "../dados/sensores_brutos_2025_2026_30s.csv"  # 2025→abr/2026, tem os novos
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
MAP_CSV = "configs/calibracao_v12_pressao/tag_column_map.csv"
VAL_CSV = "eval_pressure_out/label_validation.csv"
OUT_CSV = "eval_pressure_out/inventario_sensores.csv"

MIN_ALVO = 5        # incidentes ON abaixo disto: qualquer métrica é ruído amostral
MIN_COBERTURA = 0.95
MAX_POS_PARTIDA = 0.25   # >25% dos onsets logo após partida = confound de ramp-up


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], format="ISO8601", utc=True)
    return d.set_index("data_datetime").apply(pd.to_numeric, errors="coerce").sort_index()


def onsets_on(alarms: pd.DataFrame, d: pd.DataFrame, tag: str,
              conds: set[str], startup_h: float = 2.0) -> tuple[int, int, float]:
    """(onsets na janela, onsets com equipamento ligado, fração pós-partida).

    A fração pós-partida é o confound que o portão de excursão NÃO pega: um alarme
    de "pressão baixa" logo depois de a máquina ligar mede o ramp-up, não anomalia.
    Foi o que reprovou visualmente o PDAL_6240302, aprovado na estatística.
    """
    g = alarms[(alarms["Tag Alarme"] == tag) &
               (alarms["Condição do Alarme"].astype(str).str.upper().isin(conds))]
    g = g[(g["_t"] >= d.index.min()) & (g["_t"] <= d.index.max())]
    if g.empty:
        return 0, 0, float("nan")
    on = d["RUNNING_A"] > 0.5
    idx = d.index.get_indexer(g["_t"], method="nearest")
    n_on = int(on.iloc[idx].sum())
    starts = d.index[(on & ~on.shift(fill_value=False)).values]
    # searchsorted precisa dos dois lados tz-aware; g["_t"].values perde o fuso
    pos = starts.searchsorted(pd.DatetimeIndex(g["_t"]), side="right") - 1
    dt_h = [(t - starts[i]).total_seconds() / 3600.0 if i >= 0 else float("inf")
            for t, i in zip(g["_t"], pos)]
    frac = float(pd.Series(dt_h).le(startup_h).mean())
    return len(g), n_on, round(frac, 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_CSV)
    args = p.parse_args()

    alarms = pd.read_csv(ALARM_CSV)
    alarms["_t"] = pd.to_datetime(alarms["Data da Ocorrência"], errors="coerce", utc=True)

    d_long, d_brut = load(CSV_LONGO), load(CSV_BRUTOS)
    rows = []

    # 1) sensores já em produção — janela longa, HI/HIHI
    prod = ["T5_AVG_A"] + [f"TC382_0{i}_A" for i in range(1, 7)] + \
           [f"TV_35{i}{x}_A" for i in range(1, 6) for x in "XY"]
    for s in prod:
        if s not in d_long.columns:
            continue
        tot, on, fs = onsets_on(alarms, d_long, s, {"HI", "HIHI"})
        rows.append({"sensor": s, "coluna": s, "grupo": "produção",
                     "janela": "jun/24–abr/26", "incidentes": tot, "incidentes_ON": on,
                     "frac_pos_partida": fs,
                     "cobertura": round(float(d_long[s].notna().mean()), 4)})

    # 2) candidatos novos — janela dos brutos, condição aprovada no portão
    try:
        val = pd.read_csv(VAL_CSV)
    except FileNotFoundError:
        val = pd.DataFrame(columns=["tag", "coluna", "veredito", "onsets_ON"])
    mp = pd.read_csv(MAP_CSV)
    novas = mp[(mp["status"] == "ok") & (~mp["ja_modelado"])]
    for _, m in novas.iterrows():
        tot, on, fs = onsets_on(alarms, d_brut, m["tag"], {"CFN", "HI", "HIHI", "LOLO"})
        v = val[val["tag"] == m["tag"]]
        rows.append({"sensor": m["tag"], "coluna": m["coluna"], "grupo": "candidato",
                     "janela": "2025–abr/26", "incidentes": tot, "incidentes_ON": on,
                     "frac_pos_partida": fs,
                     "cobertura": round(float(d_brut[m["coluna"]].notna().mean()), 4),
                     "portao": v.iloc[0]["veredito"] if len(v) else ""})

    # 3) colunas sem tag de alarme nenhuma — só servem como ENTRADA
    usadas = set(mp.loc[mp["coluna"].notna(), "coluna"])
    for c in d_brut.columns:
        if c in usadas or c in {"RUNNING_A", "any_sensor_constant_run"}:
            continue
        rows.append({"sensor": "—", "coluna": c, "grupo": "sem_alarme",
                     "janela": "2025–abr/26", "incidentes": 0, "incidentes_ON": 0,
                     "frac_pos_partida": float("nan"),
                     "cobertura": round(float(d_brut[c].notna().mean()), 4), "portao": ""})

    df = pd.DataFrame(rows).fillna({"portao": ""})

    def papel(r) -> str:
        if r["cobertura"] < MIN_COBERTURA:
            return "descartar (cobertura)"
        if r["grupo"] == "candidato" and r["portao"] not in ("APROVADA", ""):
            return "entrada"          # rótulo reprovado, mas a curva serve de contexto
        if r["incidentes_ON"] < MIN_ALVO:
            return "entrada"
        # confound de partida: se boa parte dos onsets vem logo após a máquina
        # ligar, o "evento" é o ramp-up e o alvo não mede anomalia.
        if float(r.get("frac_pos_partida") or 0) > MAX_POS_PARTIDA:
            return "ALVO (ressalva: partida)"
        return "ALVO"

    df["papel"] = df.apply(papel, axis=1)
    # agrega por coluna: várias tags podem apontar para o mesmo instrumento
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.sort_values(["papel", "incidentes_ON"], ascending=[True, False]).to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    show = ["sensor", "coluna", "grupo", "incidentes", "incidentes_ON",
            "frac_pos_partida", "cobertura", "papel"]
    alvo = df[df["papel"].str.startswith("ALVO")].sort_values("incidentes_ON", ascending=False)
    print(f"=== ALVOS UTILIZÁVEIS ({len(alvo)}) — têm ≥{MIN_ALVO} incidentes com máquina ON ===")
    print(alvo[show].to_string(index=False))

    ent = df[df["papel"] == "entrada"]
    print(f"\n=== SÓ COMO ENTRADA ({len(ent)}) — curva boa, ground-truth insuficiente ===")
    print(ent[show].to_string(index=False))

    desc = df[df["papel"].str.startswith("descartar")]
    if len(desc):
        print(f"\n=== DESCARTAR ({len(desc)}) ===")
        print(desc[show].to_string(index=False))

    prod_df = df[df["grupo"] == "produção"]
    print(f"\nResumo: {len(alvo)} alvos | {len(ent)} entradas | {len(desc)} descartes")
    print(f"  dos {len(prod_df)} sensores JÁ em produção, "
          f"{int(prod_df['papel'].str.startswith('ALVO').sum())} têm ground-truth verificável")
    print(f"\nGravado em {args.out}")


if __name__ == "__main__":
    main()
