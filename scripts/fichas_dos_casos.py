#!/usr/bin/env python3
"""Fichas por caso do TC-33003A: uma seção por trip e por falso positivo do detector publicado (desde 01/02/2025), com contexto (partida, carga, manutenção), o que o detector
viu (episódio, canais, família mais desviada, carga na notificação) e o diário de alarmes da planta. USO INTERNO: revela quais janelas são FP — não enviar à operação junto do
protocolo da etapa 1 (que é às cegas).

Uso:  PYTHONPATH=src .venv/bin/python scripts/fichas_dos_casos.py [saida.md]
"""
from __future__ import annotations
import contextlib, importlib.util, io, sys
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src"))
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.avalia.metricas import JANELA
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "experiments" / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E55 = _carrega("exp55", "55_modos_de_falha.py")
INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); LOCAL = pd.Timedelta(hours=-3); P = "954005_624_"
CANAL = {"t": "temperaturas (t)", "p": "pressões (p)", "sp": "spread do mancal (sp)", "vb": "vibração da turbina (vb)"}
RUIDO = ("PI_0319",)    # "Falha PI Gás Motor p/Partida": repete em toda partida


def loc(t):
    return (t + LOCAL).strftime("%d/%m/%Y %H:%M") if pd.notna(t) else "—"


def main():
    saida = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "FICHAS_CASOS_TC33003A.md"
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc = pd.read_parquet(RAIZ / "data_local" / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        res = ci.prever({"detector": det}, proc)
    al, vig = carrega_controle("atual")
    g = nvio.carrega_grade(); mk = nvio.mascaras(g); lig, est = mk["ligada"], mk["estavel"]
    part = lig.index[mk["partida"].to_numpy()]; d = lig.astype(int).diff(); paradas = lig.index[d == -1]
    hsx = (g["HSX_6240001A"].fillna(0) > 0.5); dh = hsx.astype(int).diff(); h_ini = hsx.index[dh == 1]; h_fim = hsx.index[dh == -1]
    manut = [(a, h_fim[h_fim > a][0] if (h_fim > a).any() else hsx.index[-1]) for a in h_ini]
    t5 = g["T5_AVG_A"].where(est); ref_ano = t5.loc[INICIO8:pd.Timestamp("2026-04-20", tz="UTC")].dropna().to_numpy()
    trips = [t for t in carrega_registro() if t >= INICIO8]; modo = E55.modos_dos_trips(trips)
    a = pd.read_csv(E55.ALARMES); a["t"] = pd.to_datetime(a["Data da Ocorrência"]).dt.tz_localize("UTC"); a = a[a["Status"].astype(str).str.startswith("ACT")].sort_values("t")
    eps = [(x, y) for x, y in ci._episodios(al.loc[INICIO8:])]
    tipo = {e: ("TP" if any(e[0] <= T and e[1] >= T - JANELA for T in trips) else "FP") for e in eps}
    aberturas = vig.index[np.flatnonzero(vig.to_numpy()[1:] & ~vig.to_numpy()[:-1]) + 1]

    def ultima_partida(t):
        p = part[part <= t]; return p[-1] if len(p) else pd.NaT

    def carga_causal(t):
        w = t5.loc[t:t + pd.Timedelta(hours=2)].dropna(); r = t5.loc[t - pd.Timedelta(days=30):t - pd.Timedelta(minutes=2)].dropna()
        return float((r < w.median()).mean()) if len(w) and len(r) else np.nan

    def canais(x, y):
        r = res.loc[x:y]; v = r["vigiado"].astype(bool)
        return ", ".join(f"{CANAL[c]} {r['a_' + c][v].mean():.0%}" for c in ("t", "p", "sp", "vb")) if v.any() else "sem vigilância"

    A1 = pd.read_csv(RAIZ / "data_local" / "exp55_episodios.csv", parse_dates=["inicio"])   # artefato do EXP 55 (ClearML 7399ed7e)

    def familia(x):
        """Família de maior PERCENTIL contra o nulo de 390 janelas (EXP 55); o desvio bruto em sigma não serve (o gás é bimodal)."""
        r = A1[A1.inicio == x]
        if not len(r):
            return "—", np.nan
        r = r.iloc[0]; return str(r.A1), float(r[f"pct_{r.A1}"])

    def diario(x, y, n=10):
        w = a[(a.t >= x) & (a.t <= y)]; w = w[~w["Tag Alarme"].astype(str).str.contains("|".join(RUIDO))]
        vistos, linhas = set(), []
        for _, r in w.iterrows():
            k = (str(r["Tag Alarme"]), str(r["Condição do Alarme"]))
            if k in vistos:
                continue
            vistos.add(k); linhas.append(f"{loc(r.t)} — {str(r['Tag Alarme']).replace(P, '')} {r['Condição do Alarme']}: {str(r['Descrição Alarme']).replace('TC_33003A', '').strip(' -')}")
        return linhas[:n], len(w)

    def manut_txt(x, y):
        dep = [(m0, (m1 - m0).total_seconds() / 3600) for m0, m1 in manut if x <= m0 <= y]
        return ", ".join(f"{loc(m0)} por {h:.0f} h" for m0, h in dep) or "nenhuma"

    L = ["# Fichas por caso — TC-33003A, fev/2025 a abr/2026", "",
         "**USO INTERNO.** Estas fichas dizem quais períodos são falso positivo. Não enviar à operação junto do protocolo da etapa 1, que é às cegas.", "",
         "Hora local (UTC−3). Detector publicado (v2) no replay causal; 8 trips, 18 episódios (9 de trip, 9 FP). Carga = percentil do T5 contra os 30 dias anteriores (EXP 61);",
         "família = a de maior percentil nas 2 h de notificação contra 390 janelas normais (EXP 55); canais = fração do tempo vigiado com o canal aceso no nível A.", ""]
    L += ["## Trips", ""]
    for i, T in enumerate(trips, 1):
        p = ultima_partida(T); w5 = t5.loc[T - pd.Timedelta(hours=48):T - pd.Timedelta(minutes=10)].dropna()
        ant = [m1 for m0, m1 in manut if m1 <= T]
        inter = [e for e in eps if e[0] <= T and e[1] >= T - JANELA]
        L += [f"### Trip {i} — {loc(T)} · modo: **{modo[T]}**", ""]
        L += [f"- **Contexto:** última partida {loc(p)} ({(T - p).total_seconds() / 3600:.1f} h antes); carga nas 48 h antes no percentil {float((ref_ano < w5.median()).mean()):.2f} do período;"
              f" último modo manutenção terminou {((T - ant[-1]).total_seconds() / 86400):.0f} d antes; manutenção nos 7 d seguintes: {manut_txt(T, T + pd.Timedelta(days=7))}."]
        if inter:
            for x, y in inter:
                f, pf = familia(x); ab = aberturas[aberturas <= x]; borda = len(ab) and (x - ab[-1]) <= pd.Timedelta(minutes=30)
                L += [f"- **Detector:** episódio de {loc(x)} a {loc(y)} ({(y - x).total_seconds() / 3600:.0f} h){' — nasce na borda da partida' if borda else ''};"
                      f" notificação {loc(x + pd.Timedelta(hours=2))}; antecedência até o trip {(T - max(x, T - JANELA)).total_seconds() / 3600:.1f} h;"
                      f" canais na janela: {canais(max(x, T - JANELA), min(y, T))}; família mais desviada na notificação: {f} (percentil {pf:.2f}); carga na notificação {carga_causal(x):.2f}."]
        else:
            L += ["- **Detector:** NÃO detectou (nenhum episódio nas 48 h antes)."]
        dl, n = diario(T - pd.Timedelta(hours=48), T + pd.Timedelta(minutes=5))
        L += [f"- **Diário da planta (48 h antes até o trip, sem o PI_0319 de partida; {n} ativações):**"] + [f"  - {x}" for x in dl] + [""]
    L += ["## Falsos positivos", ""]
    for i, (x, y) in enumerate([e for e in eps if tipo[e] == "FP"], 1):
        p = ultima_partida(x); ab = aberturas[aberturas <= x]; borda = len(ab) and (x - ab[-1]) <= pd.Timedelta(minutes=30)
        pp = paradas[paradas >= x]; termina = "numa parada" if len(pp) and pp[0] <= y + pd.Timedelta(minutes=10) else "com a máquina ligada"
        f, pf = familia(x)
        L += [f"### FP {i} — {loc(x)} · {(y - x).total_seconds() / 3600:.0f} h", ""]
        L += [f"- **Contexto:** partida {loc(p)}; alarme {(x - p).total_seconds() / 3600:.1f} h depois{' (BORDA: 30 min após a máscara abrir)' if borda else ''};"
              f" termina {termina} ({loc(y)}); manutenção até 72 h depois: {manut_txt(y, y + pd.Timedelta(hours=72))}.",
              f"- **Detector:** canais no episódio: {canais(x, y)}; família mais desviada na notificação: {f} (percentil {pf:.2f}); carga na notificação {carga_causal(x):.2f}"
              f"{' (BAIXA: a regra do EXP 62 a rebaixaria)' if carga_causal(x) < 0.5 else ''}."]
        dl, n = diario(x - pd.Timedelta(hours=6), y + pd.Timedelta(hours=1))
        L += [f"- **Diário da planta (6 h antes até o fim, sem o PI_0319 de partida; {n} ativações):**"] + ([f"  - {z}" for z in dl] or ["  - nenhuma"]) + [""]
    saida.write_text("\n".join(L), encoding="utf-8"); print(f"{saida} ({len(L)} linhas)")


if __name__ == "__main__":
    main()
