#!/usr/bin/env python3
"""
validate_pressure_labels.py
Portão de decisão da Frente A: as tags novas (pressão, temperatura de mancal, etc.)
disparam quase só `CFN`, e não existe linha para elas em `limites_sensores_*.csv`,
então não dá para derivar limiar como se faz com os termopares (760/788). Antes de
gastar treino é preciso responder empiricamente:

    1. o onset do alarme corresponde a uma excursão VISÍVEL na curva?
    2. a excursão vai na direção que o nome da tag promete (PAL desce, PAH sobe)?
    3. quanto do volume é chatter (alarme que ativa e limpa em minutos)?

Um rótulo que dispara sem a curva se mexer é ruído de instrumentação: entraria no
ground-truth como incidente impossível de detectar e derrubaria o recall medido sem
que o modelo tivesse culpa.

Roda 100% offline (só CSVs locais, sem ClearML).

Uso:
    PYTHONPATH=. python scripts/validate_pressure_labels.py
    PYTHONPATH=. python scripts/validate_pressure_labels.py --tags PAL_6240315 PALL_6240340
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd

RAW_CSV = "../dados/sensores_brutos_2025_2026_30s.csv"
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
MAP_CSV = "configs/calibracao_v12_pressao/tag_column_map.csv"
OUT_DIR = "eval_pressure_out"

PRE_WINDOW_MIN = 30.0     # janela imediatamente antes do onset onde a excursão deve estar
MAD_K = 1.4826            # MAD → sigma equivalente para distribuição normal
EXCURSION_MAD = 3.0       # |desvio| acima disto conta como excursão detectável
CHATTER_MIN = 5.0         # evento com duração < isto é fleeting alarm (ISA-18.2)


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def expected_direction(tag: str, descricao: str) -> int:
    """+1 se a curva deve SUBIR no alarme, -1 se deve DESCER, 0 se indefinido.

    A direção vem do nome da tag (padrão ISA: AL/ALL = alarm low, AH/AHH = alarm high)
    e é confirmada pela descrição. Alarmes de 'Falha' não têm direção esperada — o
    transmissor pode travar em qualquer valor.
    """
    d = _norm(descricao)
    if "falha" in d:
        return 0
    t = tag.upper()
    if re.search(r"A(LL|L)_?\d", t) or re.search(r"^TALL|^TAL_", t):
        return -1
    if re.search(r"A(HH|H)_?\d", t) or re.search(r"^TAHH|^TAH_", t):
        return +1
    # fallback pela descrição
    if any(w in d for w in ["baixa", "bx.", "bx ", "m.bx", "mt.bx", "lolo"]):
        return -1
    if any(w in d for w in ["alta", "m.alta", "mt.alta"]):
        return +1
    return 0


MIN_ONSETS = 5            # abaixo disto qualquer taxa é ruído amostral
MAX_CHATTER = 0.5         # metade dos eventos durando < CHATTER_MIN = alarme repetitivo
MAX_RARITY = 0.05         # valor de onset precisa estar na cauda da operação normal


def verdict(covered: int, com_curva: int, detect: float, rarity: float,
            durs: np.ndarray) -> str:
    """Decide se a tag entra no ground-truth. Critérios, em ordem de eliminação."""
    if covered == 0:
        # Distinção que importa: se a curva EXISTE no onset e mesmo assim o evento
        # foi descartado, quem descartou foi o RUNNING_A — não é falta de dado, e o
        # RUNNING_A já foi confirmado não-confiável. Esses casos são recuperáveis com
        # uma definição melhor de "equipamento ligado"; os outros, não.
        return "mascarado_RUNNING_A" if com_curva else "sem_curva_no_onset"
    if covered < MIN_ONSETS:
        return "poucos_eventos"
    if len(durs) and float(np.mean(durs < CHATTER_MIN)) > MAX_CHATTER:
        return "chatter"
    if not (detect >= 0.5) or not (rarity <= MAX_RARITY):
        return "sem_excursao"         # dispara sem a curva sair da faixa normal
    return "APROVADA"


def load_curves(raw_csv: str, cols: list[str]) -> pd.DataFrame:
    need = ["data_datetime", "RUNNING_A"] + cols
    df = pd.read_csv(raw_csv, usecols=lambda c: c in set(need), low_memory=False)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], format="ISO8601",
                                         errors="coerce", utc=True)
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    # o CSV bruto vem com tipos mistos por coluna (valores + marcadores de texto do
    # historiador); tudo que não for número vira NaN e sai da conta.
    return df.apply(pd.to_numeric, errors="coerce")


def alarm_events(alarms: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Pares onset→OK. Cada onset abre um evento; o OK seguinte o fecha."""
    g = alarms[alarms["Tag Alarme"] == tag].sort_values("_t")
    events, onset = [], None
    for _, r in g.iterrows():
        cond = str(r["Condição do Alarme"]).upper()
        if cond == "OK":
            if onset is not None:
                events.append((onset, r["_t"]))
                onset = None
        elif onset is None:
            onset = r["_t"]
    if onset is not None:
        events.append((onset, pd.NaT))
    return pd.DataFrame(events, columns=["onset", "ok"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv", default=RAW_CSV)
    p.add_argument("--alarm_csv", default=ALARM_CSV)
    p.add_argument("--map_csv", default=MAP_CSV)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tags", nargs="*", default=None, help="restringe a estas tags")
    p.add_argument("--pre_window_min", type=float, default=PRE_WINDOW_MIN)
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mp = pd.read_csv(args.map_csv)
    mp = mp[(mp["status"] == "ok") & (~mp["ja_modelado"])]
    if args.tags:
        mp = mp[mp["tag"].isin(args.tags)]
    cols = sorted(mp["coluna"].unique())
    print(f"[1/3] Carregando {len(cols)} curvas de {args.raw_csv} ...")
    curves = load_curves(args.raw_csv, cols)
    t0, t1 = curves.index.min(), curves.index.max()
    print(f"      {len(curves)} amostras | {t0} → {t1}")

    alarms = pd.read_csv(args.alarm_csv)
    alarms["_t"] = pd.to_datetime(alarms["Data da Ocorrência"], errors="coerce", utc=True)
    alarms = alarms.dropna(subset=["_t"])

    running = curves["RUNNING_A"] > 0.5 if "RUNNING_A" in curves else None
    pre = pd.Timedelta(minutes=args.pre_window_min)

    print(f"[2/3] Avaliando {len(mp)} tags (janela pré-onset {args.pre_window_min:.0f}min)...")
    rows = []
    for _, m in mp.iterrows():
        tag, col = m["tag"], m["coluna"]
        s_all = curves[col].dropna()
        s = s_all[running.reindex(s_all.index).fillna(False)] if running is not None else s_all
        if s.empty:
            continue
        base_med = float(s.median())
        base_mad = float((s - base_med).abs().median()) * MAD_K
        if not (base_mad > 0):
            base_mad = float(s.std()) or 1e-9

        ev = alarm_events(alarms, tag)
        ev = ev[(ev["onset"] >= t0) & (ev["onset"] <= t1)]
        want = expected_direction(tag, m["descricao"])

        devs, dirs, durs, extremes = [], [], [], []
        com_curva = 0      # onsets com curva viva, ignorando o mask de RUNNING_A
        for _, e in ev.iterrows():
            if len(s_all[(s_all.index >= e["onset"] - pre) & (s_all.index <= e["onset"])]) >= 3:
                com_curva += 1
            w = s[(s.index >= e["onset"] - pre) & (s.index <= e["onset"])]
            if len(w) >= 3:
                # ponto mais extremo da janela na direção esperada (ou o de maior |z|)
                z = (w - base_med) / base_mad
                dev = float(z.min() if want < 0 else z.max() if want > 0
                            else z.iloc[np.argmax(np.abs(z.values))])
                devs.append(dev)
                dirs.append(np.sign(dev))
                extremes.append(float(w.min() if want < 0 else w.max() if want > 0
                                      else w.iloc[np.argmax(np.abs(z.values))]))
            if pd.notna(e["ok"]):
                durs.append((e["ok"] - e["onset"]).total_seconds() / 60.0)

        covered = len(devs)
        devs_a, durs_a = np.array(devs), np.array(durs)
        detect = float(np.mean(np.abs(devs_a) >= EXCURSION_MAD)) if covered else np.nan
        dir_ok = float(np.mean(np.array(dirs) == want)) if (covered and want) else np.nan
        # Raridade: que fração da operação normal é tão extrema quanto o valor típico
        # de onset. É mais interpretável que MAD quando a curva é assimétrica — e foi
        # o que separou PDAL_6240302 (1,7% → evento real) de PAL_6240315 (11% → dentro
        # da faixa normal, alarme dispara sem a curva sair do lugar).
        if covered:
            ref = float(np.median(extremes))
            rarity = float((s < ref).mean() if want < 0 else
                           (s > ref).mean() if want > 0 else
                           min((s < ref).mean(), (s > ref).mean()))
        else:
            rarity = np.nan

        rows.append({
            "tag": tag, "coluna": col, "tipo": m["tipo"],
            "direcao_esperada": {1: "sobe", -1: "desce", 0: "—"}[want],
            "onsets_total": len(ev),
            # Dois denominadores de propósito: `onsets_curva` só exige curva viva;
            # `onsets_ON` exige também RUNNING_A=1. A diferença entre os dois mede
            # quanto o mask de operação está descartando — e o RUNNING_A já foi
            # confirmado não-confiável pela Petrobras, então esconder um dos dois
            # daria uma falsa sensação de escassez (ou de abundância) de rótulo.
            "onsets_curva": com_curva,
            "onsets_ON": covered,
            "dev_mediano_mad": round(float(np.median(devs_a)), 2) if covered else np.nan,
            "raridade": round(rarity, 4) if covered else np.nan,
            "frac_excursao": round(detect, 3) if covered else np.nan,
            "frac_direcao_ok": round(dir_ok, 3) if covered and want else np.nan,
            "dur_mediana_min": round(float(np.median(durs_a)), 1) if len(durs) else np.nan,
            "frac_chatter": round(float(np.mean(durs_a < CHATTER_MIN)), 3) if len(durs) else np.nan,
            "veredito": verdict(covered, com_curva, detect, rarity, durs_a),
            "descricao": m["descricao"],
        })

    res = pd.DataFrame(rows).sort_values(["tipo", "onsets_ON"], ascending=[True, False])
    out_csv = os.path.join(args.out_dir, "label_validation.csv")
    res.to_csv(out_csv, index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    show = ["tag", "coluna", "direcao_esperada", "onsets_total", "onsets_curva",
            "onsets_ON", "dev_mediano_mad", "raridade", "frac_excursao",
            "frac_direcao_ok", "dur_mediana_min", "frac_chatter", "veredito"]
    for tipo in ["processo", "falha"]:
        sub = res[res["tipo"] == tipo]
        if sub.empty:
            continue
        print(f"\n=== {tipo.upper()} ===")
        print(sub[show].to_string(index=False))

    aprovadas = res[res["veredito"] == "APROVADA"]
    print(f"\n[3/3] Veredito: {len(aprovadas)}/{len(res)} tags aprovadas para o "
          f"ground-truth.")
    print(res["veredito"].value_counts().to_string())
    if len(aprovadas):
        print(f"\nColunas utilizáveis: {sorted(aprovadas['coluna'].unique())}")
        print(f"Total de incidentes avaliáveis: {int(aprovadas['onsets_ON'].sum())}")
    print(f"\nGravado em {out_csv}")

    if not args.no_plots and len(aprovadas):
        plot(curves, alarms, aprovadas, args.out_dir, pre)


def plot(curves, alarms, aprovadas, out_dir, pre) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for _, r in aprovadas.iterrows():
        ev = alarm_events(alarms, r["tag"])
        ev = ev[(ev["onset"] >= curves.index.min()) & (ev["onset"] <= curves.index.max())]
        if ev.empty:
            continue
        n = min(4, len(ev))
        picks = ev.iloc[np.linspace(0, len(ev) - 1, n).astype(int)]
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
        s = curves[r["coluna"]].dropna()
        for ax, (_, e) in zip(axes[0], picks.iterrows()):
            w = s[(s.index >= e["onset"] - pd.Timedelta(hours=6)) &
                  (s.index <= e["onset"] + pd.Timedelta(hours=6))]
            ax.plot(w.index, w.values, lw=0.8)
            ax.axvline(e["onset"], color="crimson", lw=1.2, label="onset")
            if pd.notna(e["ok"]):
                ax.axvline(e["ok"], color="seagreen", lw=1.0, ls="--", label="OK")
            ax.axvspan(e["onset"] - pre, e["onset"], color="crimson", alpha=0.10)
            ax.set_title(str(e["onset"])[:16], fontsize=8)
            ax.tick_params(labelsize=6, axis="x", rotation=30)
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{r['tag']} → {r['coluna']}  ({r['descricao']})", fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"onsets_{r['tag']}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
    print(f"Figuras em {out_dir}/onsets_*.png")


if __name__ == "__main__":
    main()
