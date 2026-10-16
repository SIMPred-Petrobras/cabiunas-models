#!/usr/bin/env python3
"""EXP 30 — AS CONCLUSÕES MAIS FORTES RESISTEM AO 8.º EVENTO? (vizinhança dos k, consenso e E192 refeitos nos 8 eventos)

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. EXP 24 (a vizinhança dos k), 25/26 (o consenso) e 13/13b (o `escalada_idade_h = 192`) foram feitos nos 7 eventos do recorte do harness. Com o replay causal estendido (EXP 29: sinais_v2_8ev.parquet, reprodução exata do controle com 8 eventos: 7/8, 9 FP, 0,807 FP/mês, 407 h), repete-se o essencial nos 8, avaliado a partir de 01/02/2025, com os mesmos critérios e sementes. O 27/02/2025 é detectado pelo publicado, então o 8.º evento pode REFORÇAR ou ENFRAQUECER as conclusões.
O QUE SE FAZ (caminho exato; os sinais novos; `ci.prever` como caixa-preta; `INICIO` = 2025-02-01; os 8 eventos):
  A VIZINHANÇA: 100 sorteios dos 8 k em +-10% (semente 20261015) e 100 em +-20%: fração BOA (>= 7/8, FP <= 11, horas fora <= 550) e estatísticas (o EXP 24 deu, nos 7 eventos, 10% BONS em +-10% e 2% em +-20%).
  B CONSENSO: o conjunto 1 (60 configurações em +-10%, semente 20261016) em q = 0,5 a 0,8, e as individuais do conjunto 2 (semente 20261017): o consenso domina a fronteira? (critério do EXP 25b).
  C E192: `escalada_idade_h = 192` sozinho (o publicado tem 96).
CRITÉRIOS (os de antes, com os 8 eventos). BOM = detecção >= 7/8, FP <= 11 e horas fora <= 550. A conclusão do EXP 24 'o ponto é topo de pico' SE MANTÉM se a fração BOA em +-10% é <= 25%. O consenso SE MANTÉM como ponto de custo melhor se, em algum q, tem detecção >= 6/8, FP <= 8 e horas fora <= 260 E domina as individuais do conjunto 2 (FP <= o menor delas com detecção >= a sua e horas <= 0,9 x as menores). O E192 SE MANTÉM se dá >= 7/8 com FP <= 7 e <= 280 h.
EXPECTATIVAS: o publicado é 7/8, 9 FP, 407 h (reprodução ~95%); fração BOA em +-10% <= 25% ~70%; em +-20% <= 10% ~80%; o consenso se mantém ~55% (o evento a mais pode ser perdido junto com o de 29/04, o que daria 5/8); o consenso domina ~40%; o E192 se mantém (7/8, FP 7, ~260 h) ~65%.
INTERPRETAÇÃO PREVISTA. Se tudo se mantém, o 8.º evento não muda a leitura (pico, remoção de dois episódios, dominância do E192) e a amostra é de 8. Se o consenso ou o E192 perdem o 27/02/2025, isso mostra que o corte tem custo de detecção maior do que se via nos 7.
LIMITES. Mesmos 8 eventos e o mesmo mecanismo de remoção de dois FP (EXP 27); a vizinhança conjunta é amostra; nada se adota sem sombra.

RESULTADO (16/10/2026) -- TODAS AS CONCLUSÕES SE MANTÊM COM OS 8 EVENTOS (o 8.º evento, detectado, não muda nada).
  PUBLICADO (8 eventos): 7/8, início 6, 9 FP, 0,807 FP/mês, 407 h fora, 1.457 h totais.
  C E192 (escalada_idade_h = 192): 7/8, início 5, 7 FP, 0,628 FP/mês, 261 h fora, 1.312 h totais: SE MANTÉM (>= 7/8, FP <= 7, <= 280 h).
  A VIZINHANÇA CONJUNTA: +-10%: 10% BONS (>= 7/8, FP <= 11, <= 550 h); detecção mediana 7/8 (contagem 5/8: 10, 6/8: 34, 7/8: 47, 8/8: 9); FP mediano 12 (9 a 18); horas fora mediana 552. +-20%: 2% BONS; detecção mediana 6/8 (4/8: 1, 5/8: 21, 6/8: 38, 7/8: 26, 8/8: 14); FP mediano 14 (10 a 22); horas fora mediana 738. Os 10% e 2% são IDÊNTICOS aos dos 7 eventos: 'o ponto é o topo de um pico' SE MANTÉM (fração BOA <= 25%).
  B CONSENSO (60 configurações em +-10%): q = 0,5: 6/8, 10 FP, 0,897 FP/mês, 412 h (não domina); q = 0,6: 6/8, 8 FP, 0,717 FP/mês, 237 h (DOMINA); q = 0,7: 6/8, 7 FP, 0,628 FP/mês, 234 h (DOMINA); q = 0,8: 6/8, 6 FP, 0,538 FP/mês, 227 h (DOMINA). As individuais do conjunto 2: 7 de 60 BONS, detecção mediana 6/8, FP mediano 12; das 50 com detecção >= 6, o menor FP é 9 e as menores horas 384. Os eventos perdidos pelo consenso em q = 0,7 e 0,8: 29/04/2025 e 04/11/2025 (este já não era detectado); o 27/02/2025 é DETECTADO. O consenso SE MANTÉM.
LEITURA. O 27/02/2025 é detectado em todas as variantes (publicado, E192, consenso), logo as taxas de FP, as horas e a dominância não mudam; o que muda é só o denominador (7/8 em vez de 6/7, 6/8 em vez de 5/7) e o FP/mês (a exposição é maior: 0,807 em vez de 0,893 no publicado; 0,628 no E192 e no consenso q = 0,7; 0,538 no consenso q = 0,8). Valem as ressalvas dos EXP 26 e 27: o corte é a remoção de dois a três episódios longos (15/10/2025, 09/01/2026 e, no E192, 26/10/2025; p do bootstrap mensal do FP 0,127, os meses são os mesmos), um deles (26/10/2025) termina numa parada de 77 h seguida do trip de 04/11, e o consenso custa a detecção frágil de 29/04/2025 enquanto o E192 não custa nenhum evento.
HIPÓTESES PROSPECTIVAS REGISTRADAS (especificadas aqui, antes de qualquer dado novo; a sombra não recebe mudança de configuração: as regras são aplicadas OFFLINE sobre os sinais que ela grava): H-E192 = o detector.json vigente com `escalada_idade_h` = 192 (o resto igual); predição: em dado novo, FP/mês <= o do publicado e a detecção igual ou melhor; refuta se o FP/mês for maior ou se a detecção cair >= 1 evento. H-CONSENSO (src/nv/consenso.py; q = 0,7): predição: FP/mês menor que o do publicado e detecção igual ou 1 evento a menos. Prior de replicação ~25% para cada uma; nenhuma se adota na produção sem a prova prospectiva.
PLACAR: ACERTEI: o publicado reproduz (~95%); a fração BOA em +-10% <= 25% (~70%) e em +-20% <= 10% (~80%); o consenso se mantém (~55%) e domina (~40%); o E192 se mantém (~65%). ERREI: que o evento a mais pudesse ser perdido junto com o de 29/04 (nenhuma das variantes o perdeu).

Uso:  PYTHONPATH=src .venv/bin/python experiments/30_conclusoes_com_oito_eventos.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG

INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); GL = E25.GL; CHAVES = E25.CHAVES; ROOT = Path(__file__).resolve().parents[1] / "data_local"
BOM8 = lambda r: r.det >= 7 and r.fp <= 11 and r.horas_alarme_fora <= 550


def tarefa(a):
    return a[0], E25.alarme(a[1])


def sorteios(semente, amp, n):
    rng = np.random.default_rng(semente); return [{k: float(rng.uniform(1 - amp, 1 + amp)) for k in CHAVES} for _ in range(n)]


def main():
    E25.INICIO = INICIO8; ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    ped = [(("pub",), {}), (("e192",), None)] + [(("v10", i), m) for i, m in enumerate(sorteios(20261015, 0.10, 100))] + [(("v20", i), m) for i, m in enumerate(sorteios(20261015 + 7, 0.20, 100))] + [(("c1", i), m) for i, m in enumerate(sorteios(20261016, 0.10, 60))] + [(("c2", i), m) for i, m in enumerate(sorteios(20261017, 0.10, 60))]
    ped = [x for x in ped if x[1] is not None]
    with Pool(4) as p:
        R = dict(p.map(tarefa, ped, chunksize=2))
    # E192: botão de política, fora do dicionário de k
    import copy
    d2 = copy.deepcopy(det); d2["escalada_idade_h"] = 192
    from nv.plugin import decide_v2
    e192 = decide_v2(proc, d2)["is_anomaly"].astype(bool).to_numpy()
    lin = lambda r: f"{r.det}/{r.n_eventos} | início {r.inicio} | FP {r.fp} | FP/mês {r.fp_mes:.3f} | fora {r.horas_alarme_fora:.0f} h | totais {r.horas_alarme:.0f} h"
    pub = E25.mede(R[("pub",)], idx); print(f"PUBLICADO (8 eventos): {lin(pub)}", flush=True)
    print("\nC — E192 (escalada_idade_h = 192): " + lin(E25.mede(e192, idx)), flush=True)
    print("\nA — VIZINHANÇA CONJUNTA", flush=True)
    for tag, rot in (("v10", "±10%"), ("v20", "±20%")):
        rs = [E25.mede(R[(tag, i)], idx) for i in range(100)]; d = [r.det for r in rs]; f = [r.fp for r in rs]; o = [r.horas_alarme_fora for r in rs]
        print(f"   {rot}: BONS {sum(BOM8(r) for r in rs)}% | detecção mediana {np.median(d):.0f}/8 (mín {min(d)}, máx {max(d)}; contagem {np.bincount(d, minlength=9).tolist()}) | FP mediano {np.median(f):.0f} (mín {min(f)}, máx {max(f)}) | horas fora mediana {np.median(o):.0f}", flush=True)
    print("\nB — CONSENSO (60 configurações em ±10%) contra as individuais do conjunto 2", flush=True)
    M = np.mean([R[("c1", i)] for i in range(60)], axis=0); indiv = [E25.mede(R[("c2", i)], idx) for i in range(60)]; dom = []
    print(f"   individuais (conjunto 2): BONS {sum(BOM8(r) for r in indiv)} de 60 | detecção mediana {np.median([r.det for r in indiv]):.0f}/8 | FP mediano {np.median([r.fp for r in indiv]):.0f}", flush=True)
    for q in (0.5, 0.6, 0.7, 0.8):
        r = E25.mede(M >= q, idx); pool_ = [x for x in indiv if x.det >= r.det]; mfp = min(x.fp for x in pool_) if pool_ else None; mfo = min(x.horas_alarme_fora for x in pool_) if pool_ else None
        d_ = bool(pool_) and r.fp <= mfp and r.horas_alarme_fora <= 0.9 * mfo; dom.append(d_)
        print(f"   q = {q}: {lin(r)} | individuais com detecção >= {r.det}: {len(pool_)} (menor FP {mfp}, menores horas {mfo if mfo is None else round(mfo)}) | DOMINA: {d_}", flush=True)
        if q in (0.7, 0.8):
            print("      por evento (detectado):", {k[:10]: x['det'] for k, x in r.por_evento.items()}, flush=True)


if __name__ == "__main__":
    main()
