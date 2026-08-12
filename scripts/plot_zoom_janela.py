#!/usr/bin/env python3
"""
plot_zoom_janela.py
Recorte de uma janela de dias mostrando o que os dois braços fizeram, ponto a ponto:
temperatura, alarme do DCS, health do autoencoder e health do limiar trivial.

⚠️ O ponto de operação (half-life e threshold) é o GLOBAL, calculado na janela inteira de
avaliação e só então recortado. Recalibrar dentro do zoom faria a figura mostrar um
desempenho que o sistema não tem — é o mesmo vício de buscar o threshold na janela em que
se reporta, que já medimos valer até 30 pp.

Também expõe `prepare()` e `draw_zoom()` para o `plot_zoom_todos_alarmes.py`, que percorre
todos os incidentes reusando o mesmo desenho e o mesmo ponto de operação.

Uso:
    PYTHONPATH=. python scripts/plot_zoom_janela.py --start 2025-01-10 --end 2025-01-19
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")
bl = _load("baseline_trivial_vs_ae")
fb = _load("plot_frenteB_series")

SENSOR = "TC382_03_A"
SETPOINT = 760.0
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
# janela GLOBAL de avaliação = interseção dos dois modelos (b2024 começa em jun/2024)
G0, G1 = pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")
TASK_AE = "1a15c26d994e44febb77f0bec8c2b378"      # b2024, o melhor braço

INK, INK_MUTED, GRID = fb.INK, fb.INK_MUTED, fb.GRID
SERIES, OFF_BAND = fb.SERIES, fb.OFF_BAND
ALERT, THR, HIT, MISS, FP = fb.ALERT, fb.THR, fb.HIT, fb.MISS, fb.FP
SET_C = "#b8792a"


def prepare(task_ae: str = TASK_AE) -> dict:
    """Carrega tudo e fixa o ponto de operação GLOBAL uma única vez."""
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    mae = ev.load_mae_series(Task.get_task(task_id=task_ae), [SENSOR])[SENSOR]
    mae = mae[(mae.index >= G0) & (mae.index < G1)]
    t_grid = tc03.reindex(mae.index, method="nearest")
    inc_glob = sw.incidents_on(running, tc03, G0, G1)

    # o ponto de operação custa ~15 min (varre half-life × 120 thresholds nos 2 braços);
    # cacheado para que refazer as figuras não exija recalibrar
    cache = "eval_predictive_out/.cache_zoom_oppoint.json"
    op = {}
    if os.path.exists(cache):
        import json
        op = json.load(open(cache))
        print(f"  ponto de operação em cache: {cache}")

    arms = {}
    for key, score, label in [
        ("ae", mae, "AUTOENCODER b2024 — health = rank da EWMA do erro de reconstrução"),
        ("temp", t_grid, "LIMIAR TRIVIAL — health = rank da EWMA da própria temperatura"),
    ]:
        if key in op:
            r = op[key]
        else:
            r = bl.best_over_hl(score.dropna(), inc_glob, running)
            op[key] = {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in r.items() if k in ("hl", "threshold_q", "recall_raw",
                                                      "fa_per_day")}
        hl, q = float(r["hl"]), float(r["threshold_q"])
        h = sw.ewma_on(score.dropna(), hl, running).rank(pct=True)     # rank GLOBAL
        alert = ev.apply_sticky(h, q, sw.STICKY)
        matched, fps = fb.classify_episodes(alert, inc_glob)
        arms[key] = dict(h=h, q=q, hl=hl, label=label, matched=matched, fps=fps,
                         rr=r["recall_raw"], fa=r["fa_per_day"])
        print(f"  {key:<5} hl={hl} q={q:.4f}  global raw={r['recall_raw']:.1%} "
              f"FA={r['fa_per_day']:.3f}")

    import json
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(op, open(cache, "w"), indent=1)

    df_al, _, cond, tag = ev._parse_alarm_df(ALARM_CSV)
    df_al = df_al[df_al[tag] == SENSOR]
    return dict(running=running, tc03=tc03, arms=arms, inc_glob=inc_glob,
                alarmes=df_al, cond=cond)


def lead_real(h: pd.Series, q: float, t_inc: pd.Timestamp) -> float | None:
    """Antecedência SEM o teto de 8 h da métrica: recua até o início do episódio contínuo
    de alerta que precede o incidente. A métrica oficial censura em 8 h e por isso relata
    ~7,9 h para qualquer aviso mais antigo que isso."""
    up = h[(h >= q) & (h.index <= t_inc)]
    if not len(up):
        return None
    t = up.index[-1]
    if (t_inc - t) > pd.Timedelta(hours=24):     # alerta velho demais: não é este episódio
        return None
    while True:
        ant = h[(h.index < t) & (h.index >= t - pd.Timedelta(hours=2))]
        if len(ant) and bool((ant >= q).all()):
            t = ant.index[0]
        else:
            break
    return (t_inc - t).total_seconds() / 3600.0


def draw_zoom(axes, A: pd.Timestamp, B: pd.Timestamp, ctx: dict,
              titulo: str = "", alvo: "pd.Timestamp | None" = None) -> dict:
    """Desenha os 4 painéis no intervalo [A, B) e devolve as estatísticas do recorte.

    ⚠️ `alvo` é o incidente ao qual a página se refere. As estatísticas devolvidas
    (`*_detectou`, `*_lead_h`) valem SÓ para ele. Sem isso, como as janelas de páginas
    vizinhas se sobrepõem, um incidente detectado numa página contaria de novo nas outras
    e o recall agregado sairia inflado — foi exatamente o que aconteceu na 1ª versão
    (98,3% contra os 86,2% auditados). Os episódios FP também são devolvidos identificados
    (`*_fps`), para que o agregador possa DEDUPLICAR em vez de somar páginas."""
    tc03, running = ctx["tc03"], ctx["running"]
    inc_zoom = [t for t in ctx["inc_glob"] if A <= t < B]
    w = (tc03.index >= A) & (tc03.index < B)
    temp, on_p = tc03[w], (running[w] > 0.5)
    stats = {"n_inc": len(inc_zoom), "frac_on": float(on_p.mean()) if len(on_p) else 0.0,
             "t_max": float(temp.max()) if len(temp) else float("nan")}

    ax = axes[0]
    fb.shade_off(ax, on_p)
    ax.plot(temp.index, temp.values, lw=0.9, color=INK, alpha=0.9, zorder=2)
    ax.axhline(SETPOINT, color=SET_C, lw=1.2, ls="--", zorder=3)
    ax.annotate(f"setpoint HI {SETPOINT:.0f} °C", xy=(0.004, 0.90), xycoords="axes fraction",
                fontsize=8, color=SET_C,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
    fb.style(ax)
    ax.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)
    if titulo:
        ax.set_title(titulo, fontsize=10.5, color=INK, loc="left")

    lane = axes[1]
    al = ctx["alarmes"]
    al = al[(al["_time"] >= A) & (al["_time"] < B)]
    for _, r in al.iterrows():
        c = str(r[ctx["cond"]]).upper()
        cor = MISS if c in ("HI", "HIHI") else ("#7a9b76" if c == "OK" else INK_MUTED)
        lane.axvline(r["_time"], color=cor, lw=1.8, alpha=0.95)
        lane.annotate(c, xy=(r["_time"], 0.5), xytext=(3, 0), textcoords="offset points",
                      fontsize=6.8, color=cor, va="center", rotation=90)
    lane.set_ylim(0, 1)
    lane.set_yticks([])
    lane.set_ylabel("alarme\nDCS", fontsize=8, color=MISS, rotation=0, ha="right",
                    va="center", labelpad=12)
    for s in ("top", "right", "left"):
        lane.spines[s].set_visible(False)
    lane.spines["bottom"].set_color(GRID)
    lane.tick_params(colors=INK_MUTED, labelsize=8, length=3)

    for ax, key in zip(axes[2:], ["ae", "temp"]):
        a = ctx["arms"][key]
        hz = a["h"][(a["h"].index >= A) & (a["h"].index < B)]
        fb.shade_off(ax, on_p)
        fps_aqui = []
        for s0, s1 in a["matched"]:
            if s1 >= A and s0 < B:
                ax.axvspan(max(s0, A), min(s1, B), color=ALERT, alpha=0.85, lw=0, zorder=1)
        for s0, s1 in a["fps"]:
            if s1 >= A and s0 < B:
                ax.axvspan(max(s0, A), min(s1, B), color=FP, alpha=0.5, lw=0, zorder=1)
                fps_aqui.append((s0, s1))
        n_fp = len(fps_aqui)
        ax.plot(hz.index, hz.values, lw=1.0, color=SERIES, zorder=2)
        ax.axhline(a["q"], color=THR, lw=1.2, ls="--", zorder=3)
        hits, misses = fb.raw_hits(a["h"], a["q"], inc_zoom)
        for t in hits:
            ax.axvline(t, color=HIT, lw=1.6, zorder=4)
        for t in misses:
            ax.axvline(t, color=MISS, lw=1.6, zorder=4)
        # estatística do ALVO da página (não de todos os incidentes da janela)
        det_alvo = (alvo in hits) if alvo is not None else None
        lead_alvo = lead_real(a["h"], a["q"], alvo) if (alvo is not None and det_alvo) else None
        fb.style(ax)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("health index", fontsize=9, color=INK)
        if alvo is not None:
            alvo_txt = ("detectou o alvo" if det_alvo else "PERDEU o alvo")
            lead_txt = f" · antecedência real {lead_alvo:.1f} h" if lead_alvo else ""
        else:
            alvo_txt, lead_txt = f"detectou {len(hits)}/{len(inc_zoom)}", ""
        ax.text(0.005, 0.96,
                f"{a['label']}   —   q={a['q']:.4f} · hl={a['hl']}h  ·  "
                f"{alvo_txt}{lead_txt}  ·  {len(inc_zoom)} incidente(s) e {n_fp} episódio(s) FP no recorte",
                transform=ax.transAxes, fontsize=8.2, color=INK, va="top",
                bbox=dict(facecolor="white", edgecolor=GRID,
                          boxstyle="round,pad=0.26", alpha=0.95))
        stats[f"{key}_detectou"] = det_alvo
        stats[f"{key}_lead_h"] = lead_alvo
        stats[f"{key}_n_fp"] = n_fp
        stats[f"{key}_fps"] = fps_aqui

    dias = (B - A).days
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, dias // 8)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].set_xlim(A, B)
    return stats


def legenda(ax) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Line2D([], [], color=MISS, lw=2, label="alarme HI/HIHI no DCS"),
        Line2D([], [], color=HIT, lw=1.6, label="incidente detectado (cruzamento em 8 h)"),
        Line2D([], [], color=THR, lw=1.2, ls="--", label="threshold global"),
        Patch(facecolor=ALERT, label="alerta que antecede incidente"),
        Patch(facecolor=FP, alpha=0.5, label="alerta falso positivo"),
        Patch(facecolor=OFF_BAND, alpha=0.6, label="equipamento OFF"),
    ], loc="lower right", fontsize=7.4, framealpha=0.95, ncol=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    out = args.out or f"eval_predictive_out/fig_zoom_{args.start}_{args.end}_{SENSOR}.png"

    ctx = prepare()
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.14, 1, 1]})
    fig.patch.set_facecolor("white")
    tit = (f"Recorte {A.strftime('%d/%m/%Y')} – {(B - pd.Timedelta(days=1)).strftime('%d/%m/%Y')}"
           f"  ·  {SENSOR}  ·  ponto de operação GLOBAL "
           f"({len(ctx['inc_glob'])} incidentes), não recalibrado aqui")
    st = draw_zoom(axes, A, B, ctx, tit)
    legenda(axes[-1])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Figura: {out}\n{st}")


if __name__ == "__main__":
    main()
