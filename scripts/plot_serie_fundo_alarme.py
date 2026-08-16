#!/usr/bin/env python3
"""
plot_serie_fundo_alarme.py
Série temporal do sensor com o FUNDO PINTADO pela condição de alarme ativa.

Uma faixa por mês (calendário vertical), x = dia do mês, para que um alarme de
~1h continue visível — no eixo inteiro de 23 meses ele teria menos de um pixel.

Estado de alarme reconstruído do CSV do DCS: um registro não-OK ABRE o estado,
o registro seguinte (outra condição ou OK) o FECHA. É por isso que o fundo pode
trocar de cor dentro do mesmo ciclo: HI abre, HIHI substitui segundos depois,
OK fecha.

Cinza = equipamento desligado (RUNNING_A <= 0.5). Sem ele o gráfico engana:
excursão de temperatura com a máquina parada não é anomalia.

Cor: mesma atribuição fixa de plot_intervalos_alarmes.py (4 slots categóricos
validados all-pairs). OK não é pintado — é a ausência de alarme.

Uso:
    PYTHONPATH=. python scripts/plot_serie_fundo_alarme.py
    PYTHONPATH=. python scripts/plot_serie_fundo_alarme.py --sensor T5_AVG_A
    PYTHONPATH=. python scripts/plot_serie_fundo_alarme.py --de 2024-09 --ate 2025-03
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

COR = {"HI": "#2a78d6", "HIHI": "#eb6834", "UNDER": "#1baf7a", "OVER": "#4a3aa7"}
COR_OUTROS = "#77756e"
# OK = normalidade declarada. Neutro de propósito: não é alarme, não compete
# com os slots categóricos e não estoura o limite de 4 cores validadas.
COR_OK, COR_OK_LINHA = "#eef2ee", "#b9c4b9"
COR_OFF = "#e6e4de"
LINHA = "#0b0b0b"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#dedcd6"
SUP = "#fcfcfb"
ORDEM = ["HIHI", "HI", "OVER", "UNDER", "OUTROS"]


def _resolve_dados() -> str:
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado a partir do repo.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--de", default=None, help="mês inicial, ex 2024-09")
    p.add_argument("--ate", default=None, help="mês final, ex 2025-03")
    p.add_argument("--out", default=None)
    return p.parse_args()


def estados_alarme(df: pd.DataFrame) -> tuple[list, list]:
    """(inicio, fim, condicao) dos estados de ALARME, e (inicio, fim) dos estados OK.

    Um registro não-OK abre estado de alarme; o registro seguinte o fecha. Um OK
    abre estado OK, que vai até o próximo alarme (ou até o fim da série).
    O estado OK é a NORMALIDADE declarada pelo DCS — distinto de "sem registro",
    que é o trecho antes do primeiro alarme do sensor.
    """
    alarmes, oks, ini, cond, ini_ok = [], [], None, None, None
    for _, r in df.iterrows():
        if ini is not None:
            alarmes.append((ini, r.t, cond))
            ini, cond = None, None
        if r.c == "OK":
            ini_ok = r.t
        else:
            if ini_ok is not None:
                oks.append((ini_ok, r.t))
                ini_ok = None
            ini, cond = r.t, (r.c if r.c in COR else "OUTROS")
    if ini_ok is not None:
        oks.append((ini_ok, pd.Timestamp("2100-01-01", tz="UTC")))
    return alarmes, oks


def main() -> None:
    args = parse_args()
    sensor = args.sensor
    dados = _resolve_dados()
    out_png = args.out or f"eval_predictive_out/fig_serie_fundo_alarme_{sensor}.png"

    raw = pd.read_csv(os.path.join(dados, "sensores_2024h2_2025_2026_30s.csv"),
                      usecols=["data_datetime", sensor, "RUNNING_A"], low_memory=False)
    raw["t"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["t"]).set_index("t").sort_index()
    val = pd.to_numeric(raw[sensor], errors="coerce")
    on = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0) > 0.5

    al = pd.read_csv(os.path.join(dados, "alarmes_selecionados_turbina_a.csv"))
    al["t"] = pd.to_datetime(al["Data da Ocorrência"], utc=True, errors="coerce")
    al = al.dropna(subset=["t"])
    al = al[al["Tag Alarme"] == sensor].copy()
    al["c"] = al["Condição do Alarme"].str.upper().str.strip()
    est, oks = estados_alarme(al.sort_values("t").reset_index(drop=True))

    # envelope de 5 min: preserva o pico sem plotar 2 milhões de pontos
    env = val.resample("5min").agg(["min", "max", "mean"]).dropna(how="all")
    off = (~on).resample("5min").max().fillna(False)

    meses = sorted(val.index.to_period("M").unique())
    if args.de:
        meses = [m for m in meses if str(m) >= args.de]
    if args.ate:
        meses = [m for m in meses if str(m) <= args.ate]
    # Escala pela FAIXA DE OPERAÇÃO (só ON): com o range completo as paradas
    # (que caem a ~0 °C) achatam a banda de 600-750 onde os alarmes acontecem.
    # As paradas saem da escala, mas já estão marcadas em cinza.
    # Piso = SENTINEL_LOW (500 °C): abaixo disso é rampa de partida/parada ou
    # leitura de falha, já tratada como não-incidente pelo filtro de fantasma.
    # São só 1,4% dos pontos ON e comprimiriam toda a banda útil de 600-800 °C.
    v_on = val[on.reindex(val.index, fill_value=False)]
    lo, hi = 500.0, float(np.nanpercentile(v_on, 99.95))
    pad = (hi - lo) * 0.06

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRADE,
                         "text.color": TINTA, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2,
                         "figure.facecolor": SUP, "axes.facecolor": SUP})
    n = len(meses)
    fig, axes = plt.subplots(n, 1, figsize=(13.5, 0.95 * n + 1.9), squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.085, right=0.975, top=1 - 1.45 / (0.95 * n + 1.9),
                        bottom=0.5 / (0.95 * n + 1.9), hspace=0.42)

    usadas = set()
    for ax, mes in zip(axes, meses):
        t0 = pd.Timestamp(mes.start_time, tz="UTC")
        t1 = pd.Timestamp(mes.end_time, tz="UTC")
        e = env.loc[t0:t1]
        o = off.loc[t0:t1]

        # ordem de pintura: OK no fundo (cobre quase tudo), depois DESLIGADO
        # por cima dele (informação mais importante), depois os alarmes.
        if len(o):
            blocos, cur = [], (o.index[0] if o.iloc[0] else None)
            for ts, prev, nxt in zip(o.index[1:], o.values[:-1], o.values[1:]):
                if nxt and not prev:
                    cur = ts
                elif prev and not nxt and cur is not None:
                    blocos.append((cur, ts)); cur = None
            if cur is not None:
                blocos.append((cur, o.index[-1]))
            for a, b in blocos:
                ax.axvspan(a, b, color=COR_OFF, lw=0, zorder=1)

        # estado OK: normalidade DECLARADA pelo DCS (distinta de "sem registro",
        # que é o trecho antes do primeiro alarme do sensor)
        for a, b in oks:
            if b < t0 or a > t1:
                continue
            ax.axvspan(max(a, t0), min(b, t1), color=COR_OK, alpha=1.0, lw=0, zorder=0)
        # o instante do OK: é ele que fecha o ciclo e define a duração do alarme
        for a, _b in oks:
            if t0 <= a <= t1:
                ax.axvline(a, color=COR_OK_LINHA, lw=0.7, zorder=2)

        for a, b, c in est:
            if b < t0 or a > t1:
                continue
            cor = COR.get(c, COR_OUTROS)
            # alarmes duram ~1h num painel de 30 dias: largura mínima para não sumir
            b2 = max(b, a + pd.Timedelta(hours=2.5))
            ax.axvspan(max(a, t0), min(b2, t1), color=cor, alpha=0.65, lw=0, zorder=3)
            usadas.add(c)

        if len(e):
            ax.fill_between(e.index, e["min"], e["max"], color=LINHA, alpha=0.35,
                            lw=0, zorder=5)
            ax.plot(e.index, e["mean"], color=LINHA, lw=0.6, zorder=6)

        ax.set_xlim(t0, t1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([500, 650, 800])
        ax.tick_params(axis="y", labelsize=7.5, length=2)
        ax.set_xticks(pd.date_range(t0, t1, freq="7D", tz="UTC"))
        ax.set_xticklabels([f"{d.day}" for d in
                            pd.date_range(t0, t1, freq="7D", tz="UTC")], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1)
        ax.grid(axis="y", color=GRADE, lw=0.5, alpha=0.5, zorder=4)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.text(-0.012, 0.5, f"{mes.year}\n{mes.month:02d}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, color=TINTA2, linespacing=1.25)

    fig.text(0.085, 0.985, f"Série temporal com o fundo pintado pelo alarme — {sensor}",
             fontsize=15, fontweight="bold", color=TINTA, va="top")
    fig.text(0.085, 0.968,
             "Uma faixa por mês, eixo x = dia. Linha preta = temperatura (envelope "
             "min–máx de 5 min). Fundo colorido = condição de alarme ativa no DCS.",
             fontsize=9, color=TINTA2, va="top")
    fig.text(0.085, 0.9545,
             "Escala fixa em 500-800 °C (banda de operação); paradas saem por baixo, marcadas em cinza. Faixas de alarme com largura mínima de 2,5 h — a duração "
             "mediana real é ~1 h e sumiria num painel de 30 dias.",
             fontsize=9, color=TINTA2, va="top")

    handles = [Patch(facecolor=COR.get(c, COR_OUTROS), alpha=0.55, label=c)
               for c in ORDEM if c in usadas]
    handles.append(Patch(facecolor=COR_OK, edgecolor=COR_OK_LINHA,
                         label="OK (normalidade declarada)"))
    handles.append(Patch(facecolor=COR_OFF, label="equipamento desligado"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.975, 0.995), handlelength=1.4)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SUP)
    print(f"Gravado: {out_png}")
    print(f"  meses={len(meses)}  estados de alarme={len(est)}  "
          f"condições pintadas={sorted(usadas)}")


if __name__ == "__main__":
    main()
