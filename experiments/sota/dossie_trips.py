"""Dossiê caso a caso dos trips (aprofundamento descritivo do EXP 66; nada se adota). Figuras e números de um caso por vez, em hora local.

Fontes: a grade de 2 min (UTC, convertida para local), o export de 30 s do PI e o diário de alarmes. O export e o diário estão em hora local até 2024 e em UTC
desde 2025 (o export conferido contra a grade em 25/10: correlação 1,0 na defasagem certa; ver `anatomia_trips.DIARIO_LOCAL_ATE`); aqui tudo sai em hora local.
A partir de 25/10/2024 entra o replay do detector publicado sobre o cache de sinais (o mesmo do EXP 42). Saídas em data_local/sota/anatomia_trips/dossie/.
Uso: PYTHONPATH=src:experiments .venv/bin/python -m sota.dossie_trips --caso 2024-01
"""
from __future__ import annotations
import argparse, contextlib, glob, io, sys, warnings
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
EXPORT_LOCAL_ATE = pd.Timestamp("2025-01-01")  # arquivos mensais a partir daqui estão em UTC
num = lambda v: f"{v:g}".replace(".", ",")  # número com vírgula decimal nos rótulos
P = "954005_624_"; VIZINHOS = ["TI_0303", "TI_0307"]; H, HH = 120.0, 125.0  # limites dos mancais do compressor (planilha de limites da planta)
# cores: um matiz categórico (slot 1) para a medida principal; contexto em tinta neutra
AZUL, TINTA, TINTA2, CINZA, GRADE, PARADA = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f1f0ed"

CASOS = {
    "2024-01": dict(
        tipo="mancal", titulo="Janeiro de 2024 — parada de 16/01 (registro) e trip de 17/01 07:56",
        ini="2024-01-03 00:00", fim="2024-01-18 18:00", base=("2024-01-03", "2024-01-12 12:00"),
        eventos=[("2024-01-04 17:14", "parada sem alarme\n(5,4 d)"), ("2024-01-14 16:00", "degrau\n+15 °C"), ("2024-01-16 06:10", "parada do\nregistro"),
                 ("2024-01-17 07:56", "TRIP\nmuito alto")],
        zooms=[("04/01", "2024-01-04 12:30", "2024-01-04 17:20"), ("14/01", "2024-01-14 15:00", "2024-01-14 19:50"), ("17/01 (trip)", "2024-01-17 07:10", "2024-01-17 08:00")],
    ),
    "2025-02": dict(
        tipo="selagem", titulo="Fevereiro de 2025 — trip de 27/02 05:38 (selagem, registro), repetido às 14:33, e a parada de 20/02 06:13",
        ini="2025-02-14 00:00", fim="2025-03-02 00:00",
        eventos=[("2025-02-20 06:13", "parada abrupta\n+ falha do PDIT"), ("2025-02-27 05:38", "TRIP 05:38\ne 14:33"), ("2025-02-27 14:33", "")],
        zooms=[("20/02 (parada)", "2025-02-20 06:00", "2025-02-20 06:25"), ("27/02 (trip do registro)", "2025-02-27 03:30", "2025-02-27 05:50"),
               ("27/02 (trip de novo)", "2025-02-27 14:20", "2025-02-27 14:45")],
        zoom_tag="PDIT_0305", zoom_contexto=[], limites=(2.5, 6.0), zoom_ylim=[(-0.6, 0.9), (-0.5, 11.5), (-0.5, 11.5)],
    ),
    "2025-03": dict(
        tipo="mancal", titulo="Março de 2025 — trip de 17/03 15:16 (mancal LNA, registro) e de novo em 18/03 08:16",
        ini="2025-02-28 00:00", fim="2025-03-27 00:00", base=("2025-02-01", "2025-02-27 05:00"), carga=True,
        eventos=[("2025-03-13 11:13", "todos os mancais\n+6 a 8 °C"), ("2025-03-17 15:16", "TRIP 17/03\ne 18/03"), ("2025-03-18 08:16", "")],
        zooms=[("13/03 (modo comum)", "2025-03-13 11:00", "2025-03-13 11:50"), ("17/03 (trip)", "2025-03-17 14:20", "2025-03-17 15:17"),
               ("18/03 (religa e trip)", "2025-03-18 05:00", "2025-03-18 08:18")],
    ),
    "2025-04-07": dict(
        tipo="mancal", titulo="Abril de 2025 — trip de 07/04 18:18 (mancal LNA, registro)",
        ini="2025-03-18 12:00", fim="2025-04-09 00:00", base=("2025-02-01", "2025-02-27 05:00"), carga=True,
        eventos=[("2025-03-25 00:00", "deriva\nrecomeça"), ("2025-03-31 04:50", "óleo quente\n(parada)"), ("2025-04-02 02:18", "óleo quente\n(parada)"),
                 ("2025-04-04 09:00", "carga\nsobe"), ("2025-04-07 04:52", ""), ("2025-04-07 18:18", "TRIP")],
        zooms=[("31/03 (óleo quente)", "2025-03-31 04:20", "2025-03-31 05:00"), ("02/04 (óleo quente)", "2025-04-02 01:25", "2025-04-02 02:25"),
               ("07/04 (trip)", "2025-04-07 17:30", "2025-04-07 18:19")],
    ),
    "2025-04-11": dict(
        tipo="mancal", titulo="Abril de 2025 — trip de 11/04 14:02 (mancal LNA, registro)",
        ini="2025-04-08 06:00", fim="2025-04-16 00:00", base=("2025-02-01", "2025-02-27 05:00"), carga=True,
        eventos=[("2025-04-10 16:06", "falha de\naquisição"), ("2025-04-11 06:15", ""), ("2025-04-11 14:02", "rampa 06:15\n→ TRIP 14:02")],
        zooms=[("11/04 início da rampa", "2025-04-11 05:30", "2025-04-11 09:30"), ("11/04 platô e carga reduzida", "2025-04-11 09:30", "2025-04-11 12:45"),
               ("11/04 disparo final", "2025-04-11 12:45", "2025-04-11 14:03")],
    ),
    "2025-04-29": dict(
        tipo="mancal", titulo="Abril de 2025 — parada de 29/04 00:04 a 124,7 °C (mancal LNA, registro)",
        ini="2025-04-13 00:00", fim="2025-04-30 00:00", base=("2025-02-01", "2025-02-27 05:00"), carga=True,
        eventos=[("2025-04-25 06:34", "paradas e\nmanutenção"), ("2025-04-27 16:42", "parada\nabrupta"), ("2025-04-29 00:04", "PARADA\n124,7 °C")],
        zooms=[("27/04 parada abrupta", "2025-04-27 16:20", "2025-04-27 17:10"), ("28/04 subida lenta", "2025-04-28 12:00", "2025-04-28 21:30"),
               ("28/04 disparo e parada", "2025-04-28 21:30", "2025-04-29 00:06")],
    ),
    "2025-11-04": dict(
        tipo="mancal", titulo="Outubro e novembro de 2025 — excursão do mancal LNA sem trip e o trip de óleo de 04/11 03:22",
        ini="2025-09-20 00:00", fim="2025-11-12 00:00", base=("2025-02-01", "2025-02-27 05:00"), carga=True,
        eventos=[("2025-10-27 11:50", "disparo\nque parou"), ("2025-10-31 06:42", ""), ("2025-11-04 03:22", "manutenção 31/10\ne TRIP de óleo")],
        zooms=[("27/10 disparo que parou", "2025-10-27 11:00", "2025-10-27 14:00"), ("30–31/10 junto de 120 °C", "2025-10-30 12:00", "2025-10-31 06:45"),
               ("03/11 religamento até o trip", "2025-11-03 11:50", "2025-11-04 03:23")],
        oleo=dict(ini="2025-11-04 02:55", fim="2025-11-04 03:26", eventos=[("2025-11-04 03:14:30", "carga\ncai"), ("2025-11-04 03:17:30", "carga\ncai"),
                                                                            ("2025-11-04 03:22:18", "TRIP")]),
    ),
}


def grade_local() -> pd.DataFrame:
    g = nvio.carrega_grade().copy(); g.columns = [c.replace(P, "") for c in g.columns]
    g.index = g.index.tz_convert(None) - pd.Timedelta(hours=3)
    return g


def export_30s(ini: str, fim: str, tags: list[str]) -> pd.DataFrame:
    """O export mensal de 30 s, devolvido em hora local. Lê só as colunas pedidas."""
    import openpyxl
    a, b = pd.Timestamp(ini), pd.Timestamp(fim); partes = []
    for mes in pd.period_range(a, b + pd.Timedelta(hours=3), freq="M"):
        f = sorted(glob.glob(str(EXPORT_30S / str(mes.year) / f"*{mes.year}-{mes.month:02d}-01_*")))
        if not f:
            continue
        desloc = pd.Timedelta(hours=-3) if mes.start_time >= EXPORT_LOCAL_ATE else pd.Timedelta(0)
        ws = openpyxl.load_workbook(f[0], read_only=True).worksheets[0]; it = ws.iter_rows(values_only=True); hdr = next(it)
        ic = {str(h).replace("bapiha02-", "").replace(P, ""): i for i, h in enumerate(hdr) if h}
        rows = []
        for r in it:
            t = pd.Timestamp(r[0]) + desloc
            if t > b:
                break
            if t >= a:
                rows.append([t] + [r[ic[c]] for c in tags])
        partes.append(pd.DataFrame(rows, columns=["t"] + tags).set_index("t"))
    x = pd.concat(partes).sort_index()
    return x[~x.index.duplicated()].apply(pd.to_numeric, errors="coerce")


def detector_local() -> pd.DataFrame:
    """Replay do detector publicado sobre o cache de sinais (o mesmo do EXP 42), em hora local. O cache começa em 25/10/2024."""
    from nv.controle_sinais import _ci, PKG
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
        S = pd.read_parquet(RAIZ / "data_local" / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
        R = ci.prever({"detector": det}, proc)
    R.index = R.index.tz_convert(None) - pd.Timedelta(hours=3)
    return R


def _trechos(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Os intervalos em que a série booleana fica verdadeira."""
    d = s.astype(int).diff().fillna(s.iloc[0]); ini = list(s.index[d == 1]); fim = list(s.index[d == -1])
    if len(fim) < len(ini):
        fim.append(s.index[-1])
    return list(zip(ini, fim))


def patamares(g: pd.DataFrame, tag: str, ini: str, fim: str) -> pd.DataFrame:
    """A tag em cada trecho de operação da janela (de uma partida à parada seguinte)."""
    x = g.loc[ini:fim]; op = (x.RUNNING_A.fillna(0) > 0.5) & (x.T5_AVG_A > 300)
    return pd.DataFrame([dict(partida=a, parada=b, horas=round((b - a).total_seconds() / 3600, 1), mediana=round(float(x[tag].loc[a:b].median()), 3),
                              p99=round(float(x[tag].loc[a:b].quantile(0.99)), 3), maximo=round(float(x[tag].loc[a:b].max()), 3)) for a, b in _trechos(op)])


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
    for a, b in _trechos(~op):
        ax.axvspan(a, b, color=PARADA, lw=0, zorder=0)


def _eventos(c: dict, axs, y: float):
    for t, rot in c["eventos"]:
        t = pd.Timestamp(t)
        for a in axs:
            a.axvline(t, color=TINTA, lw=0.8, ls=(0, (2, 2)), zorder=3)
        if rot:
            axs[0].text(t, y, rot, fontsize=8, color=TINTA, ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1.2))


def _fecha(c: dict, fig, axs, destino: Path):
    axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, round(len(pd.date_range(c["ini"], c["fim"])) / 14))))  # ~14 rótulos por figura
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.suptitle(c["titulo"] + " (hora local)", x=0.01, ha="left", fontsize=12, color=TINTA, y=0.995)
    fig.savefig(destino, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)


def residuo_carga(g: pd.DataFrame, base: tuple[str, str]) -> tuple[pd.Series, np.ndarray, float]:
    """A diferença TI_0305 − vizinhos descontada a carga: reta dif × T5 ajustada na base (operando, T5 > 550, carga estável há 30 min).
    Devolve o resíduo (°C, só nos instantes estáveis), os coeficientes e o MAD do resíduo na base."""
    op = (g.RUNNING_A > 0.5) & (g.T5_AVG_A > 550) & (g.T5_AVG_A.diff().abs().rolling(15).max() < 5)
    dif = g.TI_0305 - g[VIZINHOS].median(axis=1); b = pd.DataFrame({"d": dif, "T5": g.T5_AVG_A})[op].loc[base[0]:base[1]].dropna()
    coef = np.polyfit(b.T5, b.d, 1); res = (dif - np.polyval(coef, g.T5_AVG_A)).where(op); rb = res.loc[base[0]:base[1]]
    return res - rb.median(), coef, float(1.4826 * (rb - rb.median()).abs().median())


JANELAS = {"7d": (pd.Timedelta(days=7), pd.Timedelta(days=1)), "24h": (pd.Timedelta(hours=24), pd.Timedelta(hours=2)),
           "2h": (pd.Timedelta(hours=2), pd.Timedelta(minutes=10)), "10min": (pd.Timedelta(minutes=10), pd.Timedelta(0))}


def varredura(g: pd.DataFrame, trip: str, base: tuple[str, str], partida_h: float = 2.0) -> pd.DataFrame:
    """Todas as tags analógicas contra a própria faixa normal, descontada a carga, nas janelas antes do trip (JANELAS: de T − a até T − b).
    Por tag: reta tag × T5 na base (operando, aquecida, carga estável); nas janelas, a mediana do resíduo em MAD da base (z) e a fração do tempo
    fora de [p0,5; p99,5] do resíduo na base (fora). O PI_0319 fica fora (alterna entre dois valores). Só conta a máquina operando e aquecida (as primeiras `partida_h` horas depois de cada partida
    ficam fora). O MAD tem piso de 1% da faixa p1–p99 da base, para tag quase constante não explodir. O T5 entra cru, como contexto de carga."""
    T = pd.Timestamp(trip); op = (g.RUNNING_A > 0.5) & (g.T5_AVG_A > 300)
    ultima = pd.Series(g.index.where(op & ~op.shift(1, fill_value=False)), index=g.index).ffill()  # instante da última partida
    quente = op & ((g.index.to_series() - ultima) > pd.Timedelta(hours=partida_h))
    bm = (quente & (g.T5_AVG_A > 550) & (g.T5_AVG_A.diff().abs().rolling(15).max() < 5)).loc[base[0]:base[1]]
    linhas = []
    for tag in [c for c in g.columns if c not in ("RUNNING_A", "HSX_6240001A", "PI_0319")]:  # PI_0319 (gás de partida) é sinal de dois estados
        b = g[tag].loc[base[0]:base[1]][bm].dropna()
        if len(b) < 100 or b.std() == 0:
            continue
        if tag == "T5_AVG_A":
            res = g[tag]
        else:
            coef = np.polyfit(g.T5_AVG_A.loc[b.index], b, 1); res = g[tag] - np.polyval(coef, g.T5_AVG_A)
        rb = res.loc[b.index]; med = rb.median(); mad = max(1.4826 * (rb - med).abs().median(), 0.01 * (rb.quantile(0.99) - rb.quantile(0.01)), 1e-9)
        lo, hi = rb.quantile(0.005), rb.quantile(0.995); r = res.where(quente); linha = dict(tag=tag)
        for nome, (a, z) in JANELAS.items():
            w = r.loc[T - a:T - z].dropna()
            linha[f"z {nome}"] = round(float((w.median() - med) / mad), 1) if len(w) else np.nan
            linha[f"fora {nome}"] = round(float(((w < lo) | (w > hi)).mean()), 2) if len(w) else np.nan
        linhas.append(linha)
    t = pd.DataFrame(linhas).set_index("tag")
    return t.reindex(t[[f"z {n}" for n in JANELAS]].abs().max(axis=1).sort_values(ascending=False).index)


def figura_mancal(c: dict, g: pd.DataFrame, destino: Path, R: pd.DataFrame | None = None) -> dict:
    x = g.loc[c["ini"]:c["fim"]].copy(); op = x.RUNNING_A.fillna(0) > 0.5
    viz = x[VIZINHOS].median(axis=1); b0, b1 = c["base"]
    if c.get("carga"):
        res, coef, mad = residuo_carga(g, c["base"]); dif, med = res.loc[x.index], 0.0
        tit2 = f"Diferença TI_0305 − vizinhos descontada a carga (reta {num(round(coef[1], 2))} + {num(round(coef[0], 4))}·T5 de {pd.Timestamp(b0):%d/%m} a {pd.Timestamp(b1):%d/%m}); faixa = ± 4 MAD"
    else:
        dif = (x.TI_0305 - viz); ref = dif.where(op & (x.T5_AVG_A > 600)).loc[b0:b1].dropna()
        med, mad = float(ref.median()), float(1.4826 * (ref - ref.median()).abs().median())
        tit2 = f"Diferença TI_0305 − vizinhos; faixa = mediana ± 4 MAD de {pd.Timestamp(b0):%d/%m} a {pd.Timestamp(b1):%d/%m %H:%M}"
    n = 3 if R is None else 4
    fig, axs = plt.subplots(n, 1, figsize=(12, 8.2 if n == 3 else 10.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.0, 0.8, 0.9][:n], hspace=0.32))
    m = lambda s: s.where(op)  # só com a máquina operando (parada: leitura espúria da TI_0305)
    ax = axs[0]; _eixo(ax, "Mancal radial LNA do compressor (TI_0305) e mancais vizinhos (mediana de TI_0303 e TI_0307), máquina operando", "°C")
    _paradas(ax, op); ax.plot(m(viz).index, m(viz), color=CINZA, lw=1.1, label="vizinhos (mediana)"); ax.plot(m(x.TI_0305).index, m(x.TI_0305), color=AZUL, lw=1.4, label="TI_0305")
    for v in (H, HH):
        ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
    ax.text(x.index[-1], H - 1.5, "limites: alto 120, muito alto 125", fontsize=8, color=TINTA2, ha="right", va="top")
    ax.set_ylim(50, 135); ax.legend(loc="center left", bbox_to_anchor=(0.16, 0.55), frameon=False, fontsize=8.5, labelcolor=TINTA2)
    ax = axs[1]; _eixo(ax, tit2, "°C")
    _paradas(ax, op); ax.axhspan(med - 4 * mad, med + 4 * mad, color=GRADE, lw=0, zorder=0.5); ax.plot(m(dif).index, m(dif), color=AZUL, lw=1.4)
    ax.set_ylim(*((-10, 30) if c.get("carga") else (0, 45)))
    ax = axs[2]; _eixo(ax, "Carga: temperatura média de exaustão T5; faixa clara = máquina parada", "°C")
    _paradas(ax, op); ax.plot(m(x.T5_AVG_A).index, m(x.T5_AVG_A), color=AZUL, lw=1.2); ax.set_ylim(400, 800)
    if R is not None:
        ax = axs[3]; _eixo(ax, "Detector publicado (replay): canais acesos no nível A e alarme emitido", "faixas"); _paradas(ax, op); _painel_detector(ax, R, c["ini"], c["fim"])
    _eventos(c, axs, 133); _fecha(c, fig, axs, destino)
    z = (dif.where(op) - med) / mad; z6 = z.dropna().rolling("6h").median()  # mediana de 6 h: ignora pico isolado de transiente
    return dict(base_mediana=round(med, 2), base_mad=round(mad, 3), z_max=round(float(z.max()), 1), t_z_max=z.idxmax(),
                primeiro_4mad=z[z > 4].index.min(), primeiro_4mad_6h=z6[z6 > 4].index.min(), horas_acima_4mad=round(float((z > 4).sum()) * 2 / 60, 1))


def _painel_detector(ax, R: pd.DataFrame, ini: str, fim: str):
    """Canais acesos no nível A e o alarme emitido, em faixas."""
    x = R.loc[ini:fim]; linhas = [("t", "a_t"), ("p", "a_p"), ("sp", "a_sp"), ("vb", "a_vb"), ("alarme", "is_anomaly")]
    for i, (nome, col) in enumerate(linhas):
        y = len(linhas) - 1 - i; on = x[col].astype(bool) & x.vigiado.astype(bool)
        segs = [(mdates.date2num(a), mdates.date2num(b) - mdates.date2num(a)) for a, b in _trechos(on)]
        ax.broken_barh(segs, (y - 0.32, 0.64), color=TINTA if nome == "alarme" else AZUL, lw=0)
    ax.set_yticks(range(len(linhas)), [n for n, _ in linhas][::-1]); ax.set_ylim(-0.6, len(linhas) - 0.4); ax.grid(False)


def figura_selagem(c: dict, g: pd.DataFrame, R: pd.DataFrame, destino: Path):
    x = g.loc[c["ini"]:c["fim"]].copy(); op = (x.RUNNING_A.fillna(0) > 0.5) & (x.T5_AVG_A > 300); lo, hi = c["limites"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.0, 0.7, 0.9], hspace=0.34))
    ax = axs[0]; _eixo(ax, "Vazamento do selo primário, vent LA (PDIT_0305), operando e parada; 10,62 = sinal saturado (~21 mA), −2,49 = malha aberta (~0 mA)", "kgf/cm²")
    _paradas(ax, op); ax.plot(x.index, x.PDIT_0305, color=AZUL, lw=1.2)
    for v in (lo, hi):
        ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
    ax.text(x.index[0], hi + 0.3, f"limites: alto {num(lo)}, muito alto {num(hi)}", fontsize=8, color=TINTA2, ha="left", va="bottom"); ax.set_ylim(-3.2, 12.5)
    ax = axs[1]; _eixo(ax, "PDIT_0305 com a máquina operando, escala do zero (os picos saem da escala)", "kgf/cm²")
    _paradas(ax, op); ax.plot(x.index, x.PDIT_0305.where(op), color=AZUL, lw=1.2); ax.set_ylim(-0.03, 0.13)
    ax = axs[2]; _eixo(ax, "Carga: temperatura média de exaustão T5; faixa clara = máquina parada", "°C")
    _paradas(ax, op); ax.plot(x.index, x.T5_AVG_A.where(op), color=AZUL, lw=1.2); ax.set_ylim(450, 800)
    ax = axs[3]; _eixo(ax, "Detector publicado (replay): canais acesos no nível A e alarme emitido", "faixas")
    _paradas(ax, op); _painel_detector(ax, R, c["ini"], c["fim"])
    _eventos(c, axs, 12.2); _fecha(c, fig, axs, destino)


def figura_oleo(o: dict, destino: Path):
    """O trip de óleo a 30 s: pressão do óleo do compressor, header da turbina, tanque e carga (limites da planilha da planta)."""
    tags = ["PI_0308", "PI_0339", "PI_0340", "TI_0325", "T5_AVG_A", "RUNNING_A"]; x = export_30s(o["ini"], o["fim"], tags)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 0.8, 0.8], hspace=0.38))
    paineis = [("Pressão do óleo lubrificante do compressor (PI_0308); limites: baixo 0,84, muito baixo 0,56", "kgf/cm²", ["PI_0308"], (0.84, 0.56), (-0.3, 1.8)),
               ("Header de óleo lubrificante da turbina (PI_0339 azul, PI_0340 cinza); limites: baixo 2,88, muito baixo 1,78", "kgf/cm²", ["PI_0339", "PI_0340"], (2.88, 1.78), (0, 3.8)),
               ("Temperatura do tanque de óleo (TI_0325)", "°C", ["TI_0325"], (), (55, 70)), ("Carga: temperatura média de exaustão T5", "°C", ["T5_AVG_A"], (), (100, 750))]
    op = x.RUNNING_A.fillna(0) > 0.5
    for ax, (tit, un, ts, lim, yl) in zip(axs, paineis):
        _eixo(ax, tit, un); _paradas(ax, op)
        for k, t in enumerate(ts):
            ax.plot(x.index, x[t], color=AZUL if k == 0 else CINZA, lw=1.4 if k == 0 else 1.1)
        for v in lim:
            ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylim(*yl)
    for t, rot in o["eventos"]:
        for a in axs:
            a.axvline(pd.Timestamp(t), color=TINTA, lw=0.8, ls=(0, (2, 2)), zorder=3)
        axs[0].text(pd.Timestamp(t), 1.75, rot, fontsize=8, color=TINTA, ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1.2))
    axs[-1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5))); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle(f"Trip de óleo de {pd.Timestamp(o['ini']):%d/%m/%Y} no export de 30 s (hora local); faixa clara = máquina parada", x=0.01, ha="left", fontsize=12, color=TINTA, y=0.995)
    fig.savefig(destino, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)


def figura_zooms(c: dict, destino: Path) -> pd.DataFrame:
    """Os trechos decisivos a 30 s. No mancal, mesma escala nos três para comparar a forma da subida."""
    tag, ctx, (lo, hi) = c.get("zoom_tag", "TI_0305"), c.get("zoom_contexto", VIZINHOS), c.get("limites", (H, HH))
    ylims = c.get("zoom_ylim", [(50, 135)] * len(c["zooms"])); unid = "°C" if tag.startswith("TI") else "kgf/cm²"
    fig, axs = plt.subplots(1, len(c["zooms"]), figsize=(12, 3.6), sharey=len(set(ylims)) == 1, gridspec_kw=dict(wspace=0.08 if len(set(ylims)) == 1 else 0.22)); taxas = []
    tudo = export_30s(min(z[1] for z in c["zooms"]), max(z[2] for z in c["zooms"]), [tag] + ctx + ["RUNNING_A"])  # uma leitura só
    for ax, (nome, a, b), yl in zip(axs, c["zooms"], ylims):
        x = tudo.loc[a:b]; _eixo(ax, nome, unid)
        mins = (x.index - x.index[0]).total_seconds() / 60; roda = x.RUNNING_A.to_numpy() >= 0.5
        for a_, b_ in _trechos(pd.Series(~roda, index=mins)):
            ax.axvspan(a_, b_, color=PARADA, lw=0, zorder=0)
        i0 = int(roda.argmax()); depois = ~roda[i0:]  # a primeira parada depois do primeiro instante operando
        if ctx:
            ax.plot(mins, x[ctx].median(axis=1), color=CINZA, lw=1.1)
        ax.plot(mins, x[tag], color=AZUL, lw=1.4)
        for v in (lo, hi):
            if yl[0] < v < yl[1]:
                ax.axhline(v, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylim(*yl); ax.set_xlabel(f"minutos desde {pd.Timestamp(a):%d/%m %H:%M}", fontsize=8.5, color=TINTA2)
        r = (x[tag].diff().rolling(10).mean() * 2).where(x.RUNNING_A > 0.5)  # por minuto, média de 5 min, só com a máquina operando
        taxas.append(dict(janela=nome, taxa_max_por_min=round(float(r.max()), 2), subida=round(float(x[tag][roda].max() - x[tag].iloc[i0]), 2),
                          parada=x.index[i0 + int(depois.argmax())] if depois.any() else "—"))
    if hi < ylims[0][1]:
        axs[0].text(0, hi + 0.01 * (ylims[0][1] - ylims[0][0]), f"muito alto {num(hi)}", fontsize=8, color=TINTA2, va="bottom")
    quem = f"{tag} (azul) e vizinhos (cinza)" if ctx else tag
    fig.suptitle(f"{quem} no export de 30 s; faixa clara = máquina parada", x=0.01, ha="left", fontsize=11, color=TINTA, y=1.02)
    fig.savefig(destino, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return pd.DataFrame(taxas)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--caso", required=True, choices=sorted(CASOS)); a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True); c = CASOS[a.caso]; g = grade_local()
    if c["tipo"] == "mancal":
        R = detector_local() if pd.Timestamp(c["ini"]) >= pd.Timestamp("2024-10-25") else None
        r = figura_mancal(c, g, SAIDA / f"caso_{a.caso}_trajetoria.png", R)
        print(f"diferença TI_0305 − vizinhos{' descontada a carga' if c.get('carga') else ''}: base {r['base_mediana']} °C (MAD {r['base_mad']}); máximo {r['z_max']} MAD "
              f"em {r['t_z_max']:%d/%m %H:%M}; primeiro instante acima de 4 MAD {r['primeiro_4mad']:%d/%m %H:%M} (mediana de 6 h: {r['primeiro_4mad_6h']:%d/%m %H:%M}); "
              f"{r['horas_acima_4mad']} h acima de 4 MAD com a máquina operando", flush=True)
    else:
        figura_selagem(c, g, detector_local(), SAIDA / f"caso_{a.caso}_trajetoria.png")
        print(f"{c['zoom_tag']} por trecho de operação:\n{patamares(g, c['zoom_tag'], c['ini'], c['fim']).to_string(index=False)}", flush=True)
    print(figura_zooms(c, SAIDA / f"caso_{a.caso}_degraus_30s.png").to_string(index=False), flush=True)
    if c.get("oleo"):
        figura_oleo(c["oleo"], SAIDA / f"caso_{a.caso}_oleo_30s.png")
    print(f"figuras em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
