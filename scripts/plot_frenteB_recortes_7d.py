#!/usr/bin/env python3
"""
plot_frenteB_recortes_7d.py
Recortes de 7 DIAS ANTES de cada alarme HI/HIHI do TC382_03_A, no braço b2024 da
Frente B (treino jun/24→jul/25, o braço vencedor: recall_raw 86,2% = 50/58).

Formato de cada painel (especificado pelo cliente):
  - série temporal da temperatura em LINHA AZUL;
  - anomalias detectadas pelo modelo como PONTOS VERMELHOS sobre a linha;
  - momento do alarme como LISTRA TRACEJADA AMARELA (vertical);
  - trechos com o equipamento DESLIGADO sombreados em CINZA.

A janela é [t_alarme − 7d, t_alarme]; o alarme cai na borda direita, então o painel
mostra exatamente o que o detector enxergava ANTES do evento — que é a pergunta
preditiva. O ponto de operação é o MESMO da auditoria (um único threshold global,
sem recalibrar por incidente): q=0.8858 sobre o rank da EWMA(hl) do erro.

Roda OFFLINE, lendo o MAE do cache do ClearML (~/.clearml/cache). O braço é
identificado por IMPRESSÃO DIGITAL — reproduz (n_hit, n_fp, duty, n_eps) da linha
'base' de eval_predictive_out/onset_rules_TC382_03_A.csv — e ABORTA se não bater,
em vez de chutar por data de arquivo.

Uso:
    PYTHONPATH=. python scripts/plot_frenteB_recortes_7d.py
    PYTHONPATH=. python scripts/plot_frenteB_recortes_7d.py --only_hits
    PYTHONPATH=. python scripts/plot_frenteB_recortes_7d.py --days 3 --per_page 4
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")
on = _load("sweep_onset_rules_offline")


def _resolve_dados() -> str:
    """Acha o diretório `dados/`. Os scripts irmãos apontam para `../dados/`, que só
    valia quando o repo era `analise_cabiunas/cabiunas-models`; neste checkout
    (`analise_cabiunas/cabv2/cabiunas-models`) o caminho certo é `../../dados/`.
    Resolve pelo primeiro que existir em vez de fixar um nível de profundidade."""
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado a partir do repo.")


DADOS = _resolve_dados()
sw.RAW_CSV = os.path.join(DADOS, "sensores_2024h2_2025_2026_30s.csv")
sw.ALARM_CSV = os.path.join(DADOS, "alarmes_selecionados_turbina_a.csv")
ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

SENSOR = "TC382_03_A"
CACHE = os.path.expanduser("~/.clearml/cache/storage_manager/global")
ONSET_CSV = "eval_predictive_out/onset_rules_TC382_03_A.csv"
OUT_DIR = "eval_predictive_out/recortes_7d_frenteB"

# Impressão digital do braço b2024 no ponto FULL da auditoria (linha 'base' do
# onset_rules, gerada da task ClearML 1a15c26d994e44febb77f0bec8c2b378).
Q_B2024 = 0.8858
TASK_B2024 = "1a15c26d994e44febb77f0bec8c2b378"

# Paleta: exatamente o que foi pedido — azul / vermelho / amarelo / cinza.
C_LINE = "#1f6fd0"    # série de temperatura
C_ANOM = "#d62728"    # pontos de anomalia
C_ALARM = "#e8b400"   # listra do alarme
C_OFF = "#c9c9c9"     # equipamento desligado
INK, MUTED, GRID = "#141414", "#5b5b5b", "#e2e2e2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=float, default=7.0,
                   help="tamanho do recorte antes do alarme (default: 7)")
    p.add_argument("--per_page", type=int, default=3, help="painéis por página do PDF")
    p.add_argument("--only_hits", action="store_true",
                   help="só incidentes detectados pelo modelo")
    p.add_argument("--only_misses", action="store_true",
                   help="só incidentes perdidos pelo modelo")
    p.add_argument("--png", action="store_true",
                   help="também grava um PNG por incidente")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Identificação do braço por impressão digital (sem servidor ClearML)
# ---------------------------------------------------------------------------

def target_from_onset_csv() -> dict:
    """Lê a linha 'base' do onset_rules — o ponto de operação auditado do b2024."""
    df = pd.read_csv(ONSET_CSV)
    r = df[df["braco"] == "base"].iloc[0]
    return dict(n_hit=int(r["n_hit"]), n_inc=int(r["n_inc"]), n_fp=int(r["n_fp"]),
                duty=float(r["duty_sticky"]), n_eps=int(r["n_eps"]),
                recall_raw=float(r["recall_raw"]))


def find_b2024(running: pd.Series, tc03: pd.Series, tgt: dict) -> tuple[pd.Series, float, list]:
    """Varre o cache e devolve (mae, hl, incidentes) do único arquivo que reproduz
    a impressão digital. Aborta se achar 0 ou >1 — não chuta."""
    files = sorted(glob.glob(os.path.join(CACHE, "*sequence_scores_all.csv")))
    print(f"[1/3] Impressão digital em {len(files)} arquivos de cache")
    print(f"      alvo: n_hit={tgt['n_hit']}/{tgt['n_inc']} n_fp={tgt['n_fp']} "
          f"n_eps={tgt['n_eps']} duty={tgt['duty']:.6f}  (q={Q_B2024})")
    hits = []
    for f in files:
        try:
            mae = sw.read_mae(f)
        except Exception:
            continue
        if mae.empty or mae.index.min() > pd.Timestamp("2024-07-05", tz="UTC"):
            continue
        inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
        if len(inc) != tgt["n_inc"]:
            continue
        for hl in sw.HL_GRID:
            h = sw.ewma_on(mae, hl, running).rank(pct=True)
            if h.empty:
                continue
            days = (h.index[-1] - h.index[0]).total_seconds() / 86400.0
            r = on.evaluate(h >= Q_B2024, inc, days)
            if (r["n_hit"] == tgt["n_hit"] and abs(r["n_fp"] - tgt["n_fp"]) <= 1
                    and abs(r["duty_sticky"] - tgt["duty"]) < 0.003
                    and abs(r["n_eps"] - tgt["n_eps"]) <= 1):
                print(f"      MATCH {os.path.basename(f)[:32]} hl={hl} "
                      f"n_hit={r['n_hit']} n_fp={r['n_fp']} n_eps={r['n_eps']} "
                      f"duty={r['duty_sticky']:.6f} lead={r['lead_med_h']:.1f}h")
                hits.append((mae, hl, inc))
    if len(hits) != 1:
        raise SystemExit(
            f"esperava exatamente 1 match, achei {len(hits)} — impressão digital "
            "ambígua, não vou chutar qual arquivo é o b2024.")
    return hits[0]


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------

def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def shade_off(ax, on_win: pd.Series):
    """Sombreia em cinza os blocos contíguos com o equipamento desligado."""
    if on_win.empty:
        return 0
    blk = (on_win != on_win.shift()).cumsum()
    n = 0
    for _, g in on_win.groupby(blk):
        if not bool(g.iloc[0]):
            ax.axvspan(g.index[0], g.index[-1], color=C_OFF, alpha=0.55, lw=0, zorder=0)
            n += 1
    return n


def panel(ax, t_alarm, temp, on_mask, anom_t, days: float, idx: int, n_tot: int,
          detected: bool, lead_h: float):
    lo, hi = t_alarm - pd.Timedelta(days=days), t_alarm

    temp_w = temp[(temp.index >= lo) & (temp.index <= hi)]
    on_w = on_mask[(on_mask.index >= lo) & (on_mask.index <= hi)]

    shade_off(ax, on_w)
    ax.plot(temp_w.index, temp_w.values, lw=0.9, color=C_LINE, zorder=2)

    # anomalias: pontos vermelhos sobre a própria linha
    a_in = [t for t in anom_t if lo <= t <= hi]
    if a_in and not temp_w.empty:
        y = temp_w.reindex(pd.DatetimeIndex(a_in), method="nearest").values
        ax.plot(a_in, y, linestyle="none", marker="o", ms=2.6, color=C_ANOM,
                alpha=0.85, zorder=4)

    # listra tracejada amarela = momento do alarme
    ax.axvline(t_alarm, color=C_ALARM, lw=2.0, ls="--", zorder=5)

    style(ax)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("°C", fontsize=8, color=INK)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    if detected:
        tag, tag_c = f"DETECTADO {lead_h:.1f}h antes", "#1b7f4b"
    else:
        tag, tag_c = "NÃO DETECTADO", C_ANOM
    off_frac = float((~on_w).mean()) if len(on_w) else 0.0
    # margem no topo para o título curto não encostar na série
    ax.set_title(
        f"#{idx:02d}/{n_tot}  ·  alarme {t_alarm:%d/%m/%Y %H:%M} UTC  ·  "
        f"{len(a_in)} pontos de anomalia  ·  {off_frac:.0%} do recorte desligado",
        fontsize=8.5, color=INK, loc="left", pad=6)
    # status à direita do título (fora dos eixos) — não cobre dado nenhum
    ax.text(1.0, 1.012, tag, transform=ax.transAxes, fontsize=8, color=tag_c,
            ha="right", va="bottom", weight="bold")


def legend_handles():
    return [
        Line2D([], [], color=C_LINE, lw=1.4, label=f"{SENSOR} (°C)"),
        Line2D([], [], color=C_ANOM, marker="o", ms=4, ls="none",
               label="anomalia detectada pelo modelo"),
        Line2D([], [], color=C_ALARM, lw=2.0, ls="--", label="momento do alarme (DCS)"),
        Patch(facecolor=C_OFF, alpha=0.7, label="equipamento desligado"),
    ]


def main() -> None:
    args = parse_args()
    tgt = target_from_onset_csv()

    running, tc03, _ = sw.load_raw()
    on_mask = running > 0.5

    mae, hl, inc = find_b2024(running, tc03, tgt)
    h = sw.ewma_on(mae, hl, running).rank(pct=True)
    anom = h.index[h >= Q_B2024]           # instantes de disparo BRUTO (pré-sticky)
    print(f"[2/3] braço b2024: hl={hl}h  q={Q_B2024}  "
          f"{len(anom)} disparos brutos em {len(h)} amostras ON")

    # lead por incidente: primeiro disparo dentro da janela de horizonte
    anom_s = np.array([t.timestamp() for t in anom], dtype=float)
    hs = sw.HORIZON * 3600.0
    status = []
    for t in inc:
        w = anom_s[(anom_s >= t.timestamp() - hs) & (anom_s <= t.timestamp())]
        status.append((t, bool(w.size), (t.timestamp() - w.min()) / 3600.0 if w.size else float("nan")))
    n_hit = sum(1 for _, d, _ in status if d)
    print(f"      recall_raw={n_hit}/{len(inc)}={n_hit / len(inc):.1%} "
          f"(auditoria: {tgt['n_hit']}/{tgt['n_inc']}={tgt['recall_raw']:.1%})")
    if n_hit != tgt["n_hit"]:
        raise SystemExit("recall não reproduz a auditoria — abortando.")

    sel = status
    suffix = ""
    if args.only_hits:
        sel, suffix = [s for s in status if s[1]], "_hits"
    elif args.only_misses:
        sel, suffix = [s for s in status if not s[1]], "_misses"

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"recortes_{int(args.days)}d_{SENSOR}{suffix}.pdf")
    n = len(sel)
    print(f"[3/3] {n} recortes de {args.days:g} dias → {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        for start in range(0, n, args.per_page):
            chunk = sel[start:start + args.per_page]
            fig, axes = plt.subplots(len(chunk), 1,
                                     figsize=(11.0, 2.5 * len(chunk) + 1.1))
            fig.patch.set_facecolor("white")
            if len(chunk) == 1:
                axes = [axes]
            for ax, (t, det, lead) in zip(axes, chunk):
                i = sel.index((t, det, lead)) + 1
                panel(ax, t, tc03, on_mask, anom, args.days, i, n, det, lead)
            fig.suptitle(
                f"Frente B — {SENSOR}: {args.days:g} dias antes de cada alarme HI/HIHI "
                f"(braço b2024, hl={hl}h, q={Q_B2024}, ponto único da auditoria)",
                fontsize=10.5, color=INK, x=0.045, ha="left", y=0.995)
            axes[-1].legend(handles=legend_handles(), loc="upper center",
                            bbox_to_anchor=(0.5, -0.30), ncol=4, fontsize=8,
                            frameon=False)
            fig.tight_layout(rect=[0, 0.012, 1, 0.975])
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

    if args.png:
        for i, (t, det, lead) in enumerate(sel, 1):
            fig, ax = plt.subplots(figsize=(11.0, 3.4))
            fig.patch.set_facecolor("white")
            panel(ax, t, tc03, on_mask, anom, args.days, i, n, det, lead)
            # legenda abaixo do eixo: a série ocupa toda a altura útil do painel
            ax.legend(handles=legend_handles(), loc="upper center",
                      bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)
            fig.tight_layout()
            p = os.path.join(OUT_DIR, f"{i:02d}_{t:%Y%m%d_%H%M}.png")
            fig.savefig(p, dpi=140, facecolor="white")
            plt.close(fig)
        print(f"      + {n} PNGs em {OUT_DIR}/")

    print(f"\nPDF: {pdf_path}")


if __name__ == "__main__":
    main()
