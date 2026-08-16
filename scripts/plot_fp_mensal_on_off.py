#!/usr/bin/env python3
"""
plot_fp_mensal_on_off.py
Distribuição MENSAL dos falsos positivos do TC382_03_A (braço b2024 da Frente B),
separando os FP pelo estado do equipamento — que é o que decide se o FP é artefato
de transiente ou anomalia de operação estável.

Três categorias mutuamente exclusivas, por episódio FP (prioridade nessa ordem):
  1. pós-partida      — episódio começa < LIMIAR_PARTIDA h depois de uma partida
                        (OFF→ON). É a rampa térmica de startup: o AE nunca viu
                        aquele regime como "normal". Causa-raiz já documentada em
                        plot_fp_agosto2025.py.
  2. contém parada    — a janela [início, fim] do episódio atravessa um desligamento
                        (o alerta fica pendurado por cima da parada, via sticky 12h).
  3. operação estável — nem uma coisa nem outra: máquina ligada e longe de transição.
                        São os FP que NÃO têm transiente para culpar — os que de fato
                        precisam de explicação (ex: 17–18/08/2025 = carga baixa).

Painéis:
  1 — barras empilhadas: nº de episódios FP por mês, cor por categoria;
  2 — contexto de operação: horas LIGADAS por mês + FP por dia-ligado. Sem isso,
      um mês parado parece "melhora" quando na verdade só não houve operação;
  3 — dispersão de "horas desde a última partida" por FP (log), que mostra a
      distribuição real em vez de depender do corte escolhido.

Roda OFFLINE (cache ~/.clearml/cache). Reusa a impressão digital de
plot_frenteB_recortes_7d.py, que aborta se o braço b2024 não for identificado
sem ambiguidade.

Uso:
    PYTHONPATH=. python scripts/plot_fp_mensal_on_off.py
    PYTHONPATH=. python scripts/plot_fp_mensal_on_off.py --limiar_partida 24
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec = _load("plot_frenteB_recortes_7d")   # traz o resolvedor de dados + fingerprint
ev = rec.ev
sw = rec.sw
on_mod = rec.on

SENSOR = rec.SENSOR
OUT = "eval_predictive_out/fig_fp_mensal_on_off_TC382_03_A.png"
OUT_CSV = "eval_predictive_out/fp_mensal_on_off_TC382_03_A.csv"

# Um desligamento só conta como parada de verdade a partir dessa duração; abaixo
# disso é blip de instrumentação e não uma partida térmica nova.
MIN_PARADA = pd.Timedelta(hours=1)

INK, MUTED, GRID = "#141414", "#5b5b5b", "#e2e2e2"
C_PARTIDA = "#e07b39"    # FP pós-partida
C_PARADA = "#8c6bb1"     # FP cuja janela atravessa uma parada
C_ESTAVEL = "#c0392b"    # FP em operação estável (os que incomodam)
C_ON = "#b8c4cc"         # horas ligadas (contexto)
C_RATE = "#1f6fd0"       # FP por dia-ligado

CATS = [("pos_partida", "pós-partida (rampa de startup)", C_PARTIDA),
        ("contem_parada", "janela atravessa uma parada", C_PARADA),
        ("estavel", "operação estável (sem transiente)", C_ESTAVEL)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limiar_partida", type=float, default=12.0,
                   help="horas após a partida em que o FP conta como pós-partida (default: 12)")
    return p.parse_args()


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def blocos_off(on_mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Blocos contíguos de equipamento desligado com duração >= MIN_PARADA."""
    blk = (on_mask != on_mask.shift()).cumsum()
    out = []
    for _, g in on_mask.groupby(blk):
        if not bool(g.iloc[0]) and (g.index[-1] - g.index[0]) >= MIN_PARADA:
            out.append((g.index[0], g.index[-1]))
    return out


def classifica(eps_fp: list, offs: list, limiar_h: float) -> pd.DataFrame:
    """Uma linha por episódio FP: categoria + horas desde a última partida."""
    fim_off = np.array([b.value for _, b in offs], dtype="int64")   # fim da parada = partida
    ini_off = np.array([a.value for a, _ in offs], dtype="int64")
    rows = []
    for s0, s1 in eps_fp:
        ant = fim_off[fim_off <= s0.value]
        horas = (s0.value - ant.max()) / 3.6e12 if ant.size else float("nan")
        # a janela do episódio atravessa alguma parada?
        cruza = bool(np.any((ini_off <= s1.value) & (fim_off >= s0.value)))
        if np.isfinite(horas) and horas < limiar_h:
            cat = "pos_partida"
        elif cruza:
            cat = "contem_parada"
        else:
            cat = "estavel"
        rows.append(dict(inicio=s0, fim=s1,
                         dur_h=(s1 - s0).total_seconds() / 3600.0,
                         horas_desde_partida=horas, categoria=cat,
                         mes=pd.Period(s0, freq="M")))
    return pd.DataFrame(rows)


def horas_ligadas_por_mes(on_mask: pd.Series) -> pd.Series:
    """Horas de operação por mês, a partir do passo real da série."""
    dt = pd.Series(on_mask.index).diff().dt.total_seconds().median() / 3600.0
    s = on_mask.astype(float).groupby(pd.PeriodIndex(on_mask.index, freq="M")).sum() * dt
    return s


def main() -> None:
    args = parse_args()
    tgt = rec.target_from_onset_csv()

    running, tc03, _ = sw.load_raw()
    on_mask = running > 0.5

    mae, hl, inc = rec.find_b2024(running, tc03, tgt)
    h = sw.ewma_on(mae, hl, running).rank(pct=True)
    alert = on_mod.sticky_bool(h >= rec.Q_B2024, sw.STICKY)
    eps = ev.detect_episodes_gap(alert)

    # FP = episódio sem nenhum incidente em [início, fim + horizonte] — mesma regra
    # do n_fp da auditoria (best_point_for_sensor).
    inc_s = np.array([t.timestamp() for t in inc], dtype=float)
    hs = sw.HORIZON * 3600.0
    eps_fp = [(s0, s1) for s0, s1 in eps
              if not (inc_s.size and np.any((inc_s - hs <= s1.timestamp())
                                            & (inc_s >= s0.timestamp())))]
    print(f"[2/3] {len(eps)} episódios, {len(eps_fp)} FP "
          f"(auditoria: {tgt['n_eps']} / {tgt['n_fp']})")
    if abs(len(eps_fp) - tgt["n_fp"]) > 1:
        raise SystemExit("contagem de FP não reproduz a auditoria — abortando.")

    offs = blocos_off(on_mask)
    df = classifica(eps_fp, offs, args.limiar_partida)
    print(f"      {len(offs)} paradas >= {MIN_PARADA} na janela")
    for k, lbl, _ in CATS:
        n = int((df.categoria == k).sum())
        print(f"      {lbl:<38} {n:>3} FP ({n / len(df):.0%})")
    hp = df.loc[df.categoria == "pos_partida", "horas_desde_partida"]
    if len(hp):
        print(f"      dos pós-partida: {int((hp < 0.5).sum())} começam na 1a meia hora "
              f"após a partida (mediana {hp.median():.2f}h)")
    he = df.loc[df.categoria == "estavel", "horas_desde_partida"]
    if len(he):
        print(f"      dos estáveis:    mediana {he.median():.0f}h desde a partida "
              f"(mín {he.min():.0f}h, máx {he.max():.0f}h)")

    # grade mensal completa (meses sem FP têm de aparecer como zero)
    meses = pd.period_range(on_mask.index.min(), on_mask.index.max(), freq="M")
    tab = (df.groupby(["mes", "categoria"]).size().unstack(fill_value=0)
             .reindex(meses, fill_value=0))
    for k, _, _ in CATS:
        if k not in tab.columns:
            tab[k] = 0
    tab = tab[[k for k, _, _ in CATS]]

    horas_on = horas_ligadas_por_mes(on_mask).reindex(meses, fill_value=0.0)
    dias_on = horas_on / 24.0
    total_fp = tab.sum(axis=1)
    taxa = total_fp / dias_on.replace(0, np.nan)

    saida = tab.copy()
    saida["total_fp"] = total_fp
    saida["horas_ligadas"] = horas_on.round(1)
    saida["fp_por_dia_ligado"] = taxa.round(4)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    saida.to_csv(OUT_CSV, index_label="mes")

    # ---------------- figura ----------------
    x = np.arange(len(meses))
    rot = [str(m) for m in meses]
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.0),
                             gridspec_kw={"height_ratios": [1.25, 0.85, 0.85]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    base = np.zeros(len(meses))
    for k, lbl, c in CATS:
        v = tab[k].to_numpy(dtype=float)
        ax.bar(x, v, bottom=base, color=c, width=0.74, label=lbl, zorder=2)
        base += v
    for xi, tot in zip(x, total_fp.to_numpy()):
        if tot:
            ax.text(xi, tot + 0.12, str(int(tot)), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(rot, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("episódios FP no mês", fontsize=9, color=INK)
    ax.set_title(
        f"Falsos positivos por mês — {SENSOR}, braço b2024 (hl={hl}h, q={rec.Q_B2024}, "
        f"ponto único da auditoria: {len(df)} FP em {len(meses)} meses)",
        fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")

    ax = axes[1]
    ax.bar(x, horas_on.to_numpy(), color=C_ON, width=0.74, zorder=2,
           label="horas ligadas no mês")
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(rot, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("horas ligadas", fontsize=9, color=INK)
    ax2 = ax.twinx()
    ax2.plot(x, taxa.to_numpy(), color=C_RATE, lw=1.5, marker="o", ms=3.5, zorder=3,
             label="FP por dia-ligado")
    ax2.set_ylabel("FP / dia ligado", fontsize=9, color=C_RATE)
    ax2.tick_params(colors=C_RATE, labelsize=8, length=3)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax.text(0.004, 0.93,
            "normaliza a leitura: mês parado gera pouco FP por falta de operação, não por acerto",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    hs_, ls_ = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(hs_ + h2, ls_ + l2, fontsize=8, frameon=False, loc="upper right")

    ax = axes[2]
    jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(df))
    for k, lbl, c in CATS:
        m = df.categoria == k
        if not m.any():
            continue
        xs = df.loc[m, "mes"].map({mm: i for i, mm in enumerate(meses)}).to_numpy(dtype=float)
        ys = df.loc[m, "horas_desde_partida"].to_numpy(dtype=float)
        ax.scatter(xs + jitter[m.to_numpy()], ys, s=26, color=c,
                   alpha=0.85, edgecolor="white", lw=0.5, zorder=3, label=lbl)
    ax.axhline(args.limiar_partida, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(len(meses) - 0.4, args.limiar_partida * 1.25,
            f"corte pós-partida = {args.limiar_partida:g}h", fontsize=7.5,
            color=INK, ha="right", va="bottom")
    # symlog: linear até 1h (onde mora a maioria dos pós-partida, muitos em ~0h)
    # e log acima — sem clip, então o zero aparece como zero.
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_ylim(-0.15, None)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(rot, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlim(-0.7, len(meses) - 0.3)
    ax.set_ylabel("horas desde a\núltima partida", fontsize=9, color=INK)
    ax.text(0.004, 0.06,
            "cada ponto é um FP; quanto mais alto, mais longe de qualquer transiente de partida",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"\n[3/3] Figura: {OUT}\n      Tabela: {OUT_CSV}")


if __name__ == "__main__":
    main()
