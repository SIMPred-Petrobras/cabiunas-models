#!/usr/bin/env python3
"""EXP 13b — QUAIS EPISÓDIOS SAÍRAM? Verificação por episódio dos dois candidatos do EXP 13 (escalada_idade_h = 192 e refratário adaptativo) e da vizinhança de cada um.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 13 achou duas células GANHO em 30 comparações (FP 9 -> 7 e horas fora 407 -> 261/280 a detecção igual). A Regra C da pesquisa v2 já mostrou que "melhorar" o FP pode ser horas encostadas numa parada, ou fusão/corte de episódio longo, e não menos alarme falso. Antes de chamar de ganho se compara episódio por episódio contra o controle.
O QUE SE FAZ. (a) Para o controle, E192 (escalada_idade_h = 192) e o refratário adaptativo, lista cada episódio (início, duração em h, trip associado ou FP) e a diferença contra o controle: episódios do controle que SUMIRAM, que ENCURTARAM (mais de 20% de queda de duração) e que NASCERAM. (b) Vizinhança: escalada_idade_h em {144, 192, 240, 288}; refratário adaptativo com base em {48, 72, 96} h e multiplicador da duração em {0,5; 1; 2}. (c) Por evento: início, banda e lead contra o controle (qual evento perde o início no adaptativo).
CRITÉRIOS (fixados antes). Um candidato é GENUÍNO se: (1) os episódios que somem são FP do controle de VERDADE (não intersectam janela) e a maior parte da redução de horas NÃO vem de um único episódio; (2) nenhum evento perde detecção e a perda de início, se existir, é de UM evento e se explica; (3) a vizinhança imediata (os dois valores vizinhos do eixo) não perde detecção. É ARTEFATO se a redução de horas vem de 1 a 2 episódios longos (> 100 h) ou se um valor vizinho perde detecção.
EXPECTATIVAS: E192 corta horas por UM episódio longo (> 100 h) ~65% (então artefato do tipo Regra C); o adaptativo apaga FP repetidos do mesmo regime (mais de um episódio) ~55%; o adaptativo perde o início do 04/11 ou do 11/04 (os eventos de borda) ~45%; a vizinhança do adaptativo é estável em detecção ~60%; alguma das duas é GENUÍNA ~30%.
LIMITES. 7 eventos; ainda é uma escolha feita olhando os eventos (30 comparações): mesmo GENUÍNA só entra como hipótese prospectiva no sombra.py, nunca no pacote.

RESULTADO (15/10/2026) -- NENHUM DOS DOIS É GENUÍNO (ambos ARTEFATO pelo critério fixado).
  CONTROLE (16 episódios, 9 FP): os dois FP longos são 2025-10-15 (135 h) e 2025-10-26 (112 h), seguidos um do outro; o episódio de 647,7 h (29/01 a 26/02/2026) é o que 'detecta' o trip de 26/02/2026 por fusão; o início desse evento vem de um episódio SEPARADO de 20,3 h em 25/02.
  escalada_idade_h: 144 = igual ao controle em FP e piora a carga (520 h fora); 192, 240 e 288 dão o MESMO resultado (platô): somem 2 FP (26/10, 112 h; 09/01/2026, 34 h) = 146 h, das quais 112 h (77%) num único episódio > 100 h; nenhum evento muda (início/banda/lead idênticos). Critério: a maior parte da redução de horas vem de UM episódio longo -> ARTEFATO do tipo Regra C (e o vizinho 144 não ajuda). Os 2 FP a menos são reais, mas é uma amostra de 2 episódios.
  REFRATÁRIO ADAPTATIVO: as variantes que preservam 6/7 (base 48 ou 72 h com 0,5 ou 1 x duração) PERDEM O INÍCIO do evento de 26/02/2026 (o episódio de 647,7 h arma um silêncio de 72 h + 648 h que apaga o episódio de 25/02, o único que nasce na janela); a variante 'FP 7, 280 h' (72 h + 1 x) remove os FP de 26/10 (112 h) e 08/04 e o episódio de 25/02. Com base 96 h ou 2 x duração a DETECÇÃO CAI (5/7 e 4/7; perdem 17/03, 11/04, 07/04): a vizinhança imediata NÃO é estável. Mesmo driver de horas (o episódio de 112 h) e um mecanismo que pune justamente o trip que vem depois de um episódio longo.
  LEITURA. (1) As duas 'vitórias' do EXP 13 são o MESMO episódio de 112,4 h (26/10/2025), logo depois de outro de 135 h (15/10), no período ago-out/2025 já conhecido como atípico (RUNNING_A ligada sem combustão): um regime, não duas descobertas. (2) A política publicada (refratário 72 h, escalada 96 h) é ponto de borda em 8 de 9 botões e o único modo de cortar FP sem perder detecção encontrado depende de um episódio. (3) Não se adota nada; E192 poderia entrar como hipótese prospectiva no sombra (corta horas se o regime de ago-out se repetir), mas a sombra não aceita mudar a configuração congelada.
PLACAR: ACERTEI: E192 corta horas por um episódio longo (~65%); o adaptativo apaga FP repetidos do mesmo regime (~55%, 2 FP) mas junto apaga um início verdadeiro. ERREI: perde o início do 04/11 ou do 11/04 (~45%: o que perde nas variantes próximas é o de 26/02/2026; o 11/04 só nas largas); a vizinhança do adaptativo estável em detecção (~60%: base 96 h perde 1); alguma GENUÍNA (~30%: nenhuma).

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
