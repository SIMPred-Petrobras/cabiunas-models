#!/usr/bin/env python3
"""EXP 13b — QUAIS EPISÓDIOS SAÍRAM? Verificação por episódio dos dois candidatos do EXP 13 (escalada_idade_h = 192 e refratário adaptativo) e da vizinhança de cada um.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 13 achou duas células GANHO em 30 comparações (FP 9 -> 7 e horas fora 407 -> 261/280 a detecção igual). A Regra C da pesquisa v2 já mostrou que "melhorar" o FP pode ser horas encostadas numa parada, ou fusão/corte de episódio longo, e não menos alarme falso. Antes de chamar de ganho se compara episódio por episódio contra o controle.
O QUE SE FAZ. (a) Para o controle, E192 (escalada_idade_h = 192) e o refratário adaptativo, lista cada episódio (início, duração em h, trip associado ou FP) e a diferença contra o controle: episódios do controle que SUMIRAM, que ENCURTARAM (mais de 20% de queda de duração) e que NASCERAM. (b) Vizinhança: escalada_idade_h em {144, 192, 240, 288}; refratário adaptativo com base em {48, 72, 96} h e multiplicador da duração em {0,5; 1; 2}. (c) Por evento: início, banda e lead contra o controle (qual evento perde o início no adaptativo).
CRITÉRIOS (fixados antes). Um candidato é GENUÍNO se: (1) os episódios que somem são FP do controle de VERDADE (não intersectam janela) e a maior parte da redução de horas NÃO vem de um único episódio; (2) nenhum evento perde detecção e a perda de início, se existir, é de UM evento e se explica; (3) a vizinhança imediata (os dois valores vizinhos do eixo) não perde detecção. É ARTEFATO se a redução de horas vem de 1 a 2 episódios longos (> 100 h) ou se um valor vizinho perde detecção.
EXPECTATIVAS: E192 corta horas por UM episódio longo (> 100 h) ~65% (então artefato do tipo Regra C); o adaptativo apaga FP repetidos do mesmo regime (mais de um episódio) ~55%; o adaptativo perde o início do 04/11 ou do 11/04 (os eventos de borda) ~45%; a vizinhança do adaptativo é estável em detecção ~60%; alguma das duas é GENUÍNA ~30%.
LIMITES. 7 eventos; ainda é uma escolha feita olhando os eventos (30 comparações): mesmo GENUÍNA só entra como hipótese prospectiva no sombra.py, nunca no pacote.

Uso:  PYTHONPATH=src .venv/bin/python experiments/13b_quem_saiu.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E13 = _carrega("exp13", "13_politica_sobre_o_escore_do_v2.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2

INICIO = E4.INICIO; JAN = pd.Timedelta(hours=48)


def adaptativo(res, base_h, mult):
    al = res["is_anomaly"].astype(bool); out = np.zeros(len(al), bool); bloq = None; idx = al.index
    for a, b in episodios_de_serie(al):
        if bloq is not None and a <= bloq:
            continue
        out[idx.searchsorted(a):idx.searchsorted(b, side="right")] = True; bloq = b + pd.Timedelta(hours=base_h) + mult * (b - a)
    r = res.copy(); r["is_anomaly"] = pd.Series(out, index=idx); return r


def lista(res, reg):
    ini = INICIO; al = res["is_anomaly"].astype(bool).loc[ini:]; out = []
    for a, b in episodios_de_serie(al):
        tr = [t for t in reg if a <= t and b >= t - JAN]
        out.append(dict(ini=a, dur_h=(b - a).total_seconds() / 3600 + 2 / 60, trip=(str(tr[0])[:10] if tr else "FP")))
    return pd.DataFrame(out)


def compara(base, var):
    """episódios do controle que somem / encurtam / nascem na variante (sobreposição de intervalos)."""
    def iv(d): return [(r.ini, r.ini + pd.Timedelta(hours=r.dur_h)) for r in d.itertuples()]
    B, V = iv(base), iv(var); somem, encurtam = [], []
    for i, (a, b) in enumerate(B):
        ov = [(c, d) for c, d in V if c < b and d > a]
        if not ov:
            somem.append(base.iloc[i])
        elif sum((min(b, d) - max(a, c)).total_seconds() for c, d in ov) < 0.8 * (b - a).total_seconds():
            encurtam.append(base.iloc[i])
    nascem = [var.iloc[j] for j, (c, d) in enumerate(V) if not any(a < d and b > c for a, b in B)]
    return somem, encurtam, nascem


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    r0 = decide_v2(proc, det); b = lista(r0, reg); R0 = E6.mede(r0, reg)
    print(f"CONTROLE: {E6.linha(R0)}\n{b.assign(ini=b.ini.dt.strftime('%Y-%m-%d %H:%M')).round(1).to_string(index=False)}\n", flush=True)

    def relata(nome, res):
        v = lista(res, reg); R = E6.mede(res, reg); s, e, n = compara(b, v)
        print(f"== {nome}: {E6.linha(R)}")
        for rot, x in (("SOMEM", s), ("ENCURTAM", e), ("NASCEM", n)):
            for r in x:
                print(f"     {rot:8s} {r.ini:%Y-%m-%d %H:%M} dur {r.dur_h:6.1f} h  {r.trip}")
        h_som = sum(r.dur_h for r in s if r.trip == "FP"); h_long = sum(r.dur_h for r in s if r.trip == "FP" and r.dur_h > 100)
        print(f"     horas de FP removidas por episódios que somem: {h_som:.0f} h (das quais {h_long:.0f} h em episódios > 100 h) | por evento (início/banda/lead): " + "; ".join(f"{k[:10]} {int(x['inicio'])}{int(x['banda'])} {x['lead_h'] if x['lead_h'] is None else round(x['lead_h'], 1)}" for k, x in R.por_evento.items()), flush=True)
        return R
    print("controle por evento:", "; ".join(f"{k[:10]} {int(x['inicio'])}{int(x['banda'])} {x['lead_h'] if x['lead_h'] is None else round(x['lead_h'], 1)}" for k, x in R0.por_evento.items()), "\n")
    for v in (144, 192, 240, 288):
        d2 = copy.deepcopy(det); d2["escalada_idade_h"] = v; relata(f"escalada_idade_h = {v}", decide_v2(proc, d2))
    base = decide_v2(proc, det)
    for bh in (48, 72, 96):
        for m in (0.5, 1.0, 2.0):
            relata(f"refratário adaptativo base {bh} h + {m} x duração", adaptativo(base, bh, m))


if __name__ == "__main__":
    main()
