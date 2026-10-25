"""Dossiê caso a caso dos trips (aprofundamento descritivo do EXP 66; nada se adota). Figuras e números de um caso por vez, em hora local.

Fontes: a grade de 2 min (UTC, convertida para local), o export de 30 s do PI (já em hora local), o diário de alarmes (hora local até 2024, UTC desde 02/2025;
ver `anatomia_trips.DIARIO_LOCAL_ATE`). Saídas em data_local/sota/anatomia_trips/dossie/.
Uso: PYTHONPATH=src:experiments .venv/bin/python -m sota.dossie_trips --caso 2024-01
"""
from __future__ import annotations
import argparse, glob, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; RAIZ = HERE.parent.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(HERE.parent))
from nv import io as nvio
from nv.io import RAIZ as RAIZ_PKG

SAIDA = RAIZ / "data_local" / "sota" / "anatomia_trips" / "dossie"
EXPORT_30S = RAIZ_PKG / "estrutura_pra_prod" / "Cabiunas" / "dados" / "tagsselecionadas_30s"
P = "954005_624_"; VIZINHOS = ["TI_0303", "TI_0307"]; H, HH = 120.0, 125.0  # limites dos mancais do compressor (planilha de limites da planta)
# cores: um matiz categórico (slot 1) para a medida principal; contexto em tinta neutra
AZUL, TINTA, TINTA2, CINZA, GRADE, PARADA = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f1f0ed"

CASOS = {
    "2024-01": dict(
        titulo="Janeiro de 2024 — parada de 16/01 (registro) e trip de 17/01 07:56",
        ini="2024-01-03 00:00", fim="2024-01-18 18:00", base=("2024-01-03", "2024-01-12 12:00"),
        eventos=[("2024-01-04 17:14", "parada sem alarme\n(5,4 d)"), ("2024-01-14 16:00", "degrau\n+15 °C"), ("2024-01-16 06:10", "parada do\nregistro"),
                 ("2024-01-17 07:56", "TRIP\nmuito alto")],
        zooms=[("04/01", "2024-01-04 12:30", "2024-01-04 17:20"), ("14/01", "2024-01-14 15:00", "2024-01-14 19:50"), ("17/01 (trip)", "2024-01-17 07:10", "2024-01-17 08:00")],
    ),
}


def grade_local() -> pd.DataFrame:
    g = nvio.carrega_grade().copy(); g.columns = [c.replace(P, "") for c in g.columns]
    g.index = g.index.tz_convert(None) - pd.Timedelta(hours=3)
    return g


def export_30s(ini: str, fim: str, tags: list[str]) -> pd.DataFrame:
    """O export mensal de 30 s (hora local). Lê só as colunas pedidas."""
    import openpyxl
    a, b = pd.Timestamp(ini), pd.Timestamp(fim); partes = []
    for mes in pd.period_range(a, b, freq="M"):
        f = sorted(glob.glob(str(EXPORT_30S / str(mes.year) / f"*{mes.year}-{mes.month:02d}-01_*")))
        if not f:
            continue
        ws = openpyxl.load_workbook(f[0], read_only=True).worksheets[0]; it = ws.iter_rows(values_only=True); hdr = next(it)
        ic = {str(h).replace("bapiha02-", "").replace(P, ""): i for i, h in enumerate(hdr) if h}
        rows = []
        for r in it:
            t = pd.Timestamp(r[0])
            if t > b:
                break
            if t >= a:
                rows.append([t] + [r[ic[c]] for c in tags])
        partes.append(pd.DataFrame(rows, columns=["t"] + tags).set_index("t"))
    return pd.concat(partes).apply(pd.to_numeric, errors="coerce")


def _eixo(ax, titulo: str, unidade: str):
    ax.set_title(f"{titulo} ({unidade})", loc="left", fontsize=10, color=TINTA, pad=4)
    ax.grid(axis="y", color=GRADE, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(CINZA)
    ax.tick_params(colors=TINTA2, labelsize=8.5)


def _paradas(ax, op: pd.Series):
    """Sombreia os intervalos com a máquina parada."""
    parada = ~op; ini = None
    for t, v in parada.items():
        if v and ini is None:
            ini = t
        elif not v and ini is not None:
            ax.axvspan(ini, t, color=PARADA, lw=0, zorder=0); ini = None
    if ini is not None:
        ax.axvspan(ini, parada.index[-1], color=PARADA, lw=0, zorder=0)


def figura_mancal(c: dict, g: pd.DataFrame, destino: Path) -> dict:
    x = g.loc[c["ini"]:c["fim"]].copy(); op = x.RUNNING_A.fillna(0) > 0.5
    viz = x[VIZINHOS].median(axis=1); dif = (x.TI_0305 - viz)
    b0, b1 = c["base"]; ref = dif.where(op & (x.T5_AVG_A > 600)).loc[b0:b1].dropna()
    med, mad = float(ref.median()), float(1.4826 * (ref - ref.median()).abs().median())
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.0, 0.8], hspace=0.32))
    m = lambda s: s.where(op)  # só com a máquina operando (parada: leitura espúria da TI_0305)
    ax = axs[0]; _eixo(ax, "Mancal radial LNA do compressor (TI_0305) e mancais vizinhos (mediana de TI_0303 e TI_0307), máquina operando", "°C")
    _paradas(ax, op); ax.plot(m(viz).index, m(viz), color=CINZA, lw=1.1, label="vizinhos (mediana)"); ax.plot(m(x.TI_0305).index, m(x.TI_0305), color=AZUL, lw=1.4, label="TI_0305")
    for v in (H, HH):
        ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
    ax.text(x.index[-1], H - 1.5, "limites: alto 120, muito alto 125", fontsize=8, color=TINTA2, ha="right", va="top")
    ax.set_ylim(50, 135); ax.legend(loc="center left", bbox_to_anchor=(0.16, 0.55), frameon=False, fontsize=8.5, labelcolor=TINTA2)
    ax = axs[1]; _eixo(ax, f"Diferença TI_0305 − vizinhos; faixa = mediana ± 4 MAD de {pd.Timestamp(b0):%d/%m} a {pd.Timestamp(b1):%d/%m %H:%M}", "°C")
    _paradas(ax, op); ax.axhspan(med - 4 * mad, med + 4 * mad, color=GRADE, lw=0, zorder=0.5); ax.plot(m(dif).index, m(dif), color=AZUL, lw=1.4)
    ax.set_ylim(0, 45)
    ax = axs[2]; _eixo(ax, "Carga: temperatura média de exaustão T5; faixa clara = máquina parada", "°C")
    _paradas(ax, op); ax.plot(m(x.T5_AVG_A).index, m(x.T5_AVG_A), color=AZUL, lw=1.2); ax.set_ylim(450, 800)
    for t, rot in c["eventos"]:
        t = pd.Timestamp(t)
        for a in axs:
            a.axvline(t, color=TINTA, lw=0.8, ls=(0, (2, 2)), zorder=3)
        axs[0].text(t, 133, rot, fontsize=8, color=TINTA, ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1.2))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator()); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.suptitle(c["titulo"] + " (hora local)", x=0.01, ha="left", fontsize=12, color=TINTA, y=0.995)
    fig.savefig(destino, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    z = (dif.where(op) - med) / mad
    return dict(base_mediana=round(med, 2), base_mad=round(mad, 3), z_max=round(float(z.max()), 1), t_z_max=z.idxmax(),
                primeiro_4mad=z[z > 4].index.min(), horas_acima_4mad=round(float((z > 4).sum()) * 2 / 60, 1))


def figura_zooms(c: dict, destino: Path) -> pd.DataFrame:
    """Os degraus a 30 s, com a mesma escala vertical, para comparar a forma da subida."""
    fig, axs = plt.subplots(1, len(c["zooms"]), figsize=(12, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08)); taxas = []
    tudo = export_30s(min(z[1] for z in c["zooms"]), max(z[2] for z in c["zooms"]), ["TI_0305"] + VIZINHOS + ["RUNNING_A"])  # uma leitura só
    for ax, (nome, a, b) in zip(axs, c["zooms"]):
        x = tudo.loc[a:b]; _eixo(ax, nome, "°C")
        mins = (x.index - x.index[0]).total_seconds() / 60
        ax.plot(mins, x[VIZINHOS].median(axis=1), color=CINZA, lw=1.1); ax.plot(mins, x.TI_0305, color=AZUL, lw=1.4)
        for v in (H, HH):
            ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel(f"minutos desde {pd.Timestamp(a):%d/%m %H:%M}", fontsize=8.5, color=TINTA2)
        r = (x.TI_0305.diff().rolling(10).mean() * 2).where(x.RUNNING_A > 0.5)  # °C/min, média de 5 min, só com a máquina operando
        taxas.append(dict(janela=nome, taxa_max_c_min=round(float(r.max()), 2), subida_c=round(float(x.TI_0305.where(x.RUNNING_A > 0.5).max() - x.TI_0305.iloc[0]), 1)))
    axs[0].set_ylim(50, 135); axs[0].text(0, HH + 1, "muito alto 125", fontsize=8, color=TINTA2, va="bottom")
    fig.suptitle("TI_0305 (azul) e vizinhos (cinza) no export de 30 s, mesma escala", x=0.01, ha="left", fontsize=11, color=TINTA, y=1.02)
    fig.savefig(destino, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return pd.DataFrame(taxas)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--caso", required=True, choices=sorted(CASOS)); a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True); c = CASOS[a.caso]; g = grade_local()
    r = figura_mancal(c, g, SAIDA / f"caso_{a.caso}_trajetoria.png")
    print(f"diferença TI_0305 − vizinhos: base {r['base_mediana']} °C (MAD {r['base_mad']}); máximo {r['z_max']} MAD em {r['t_z_max']:%d/%m %H:%M}; "
          f"primeiro instante acima de 4 MAD {r['primeiro_4mad']:%d/%m %H:%M}; {r['horas_acima_4mad']} h acima de 4 MAD com a máquina operando", flush=True)
    print(figura_zooms(c, SAIDA / f"caso_{a.caso}_degraus_30s.png").to_string(index=False), flush=True)
    print(f"figuras em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
