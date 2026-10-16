#!/usr/bin/env python3
"""EXP 33 — OS ESCORES POR FAMÍLIA ISOLADOS (sem o voto), POLÍTICA SIMPLES E CUSTO IGUAL POR BISSEÇÃO, COM 8 EVENTOS.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 32/32b mostrou que trocar um canal dentro do voto e recalibrar a escala não permite comparar escores (o custo igual não é alcançável: as alternativas carregam episódios persistentes que nenhuma escala elimina). A pergunta do usuário ('o PCA é simples demais?') pede uma comparação justa dos ESCORES, que só se faz isolando-os: cada escore sozinho, com a mesma política fixa (a do EXP 04/05: EWMA de 1 h só sobre instantes vigiados, episódio = corrida acima do limiar com duração >= 2 h, refratário de 72 h) e o limiar por bisseção (monótona) para o MESMO custo: as horas de alarme fora das janelas de trip = 407 h (a do publicado). A detecção a esse custo mede o que o escore sabe, sem o desenho do voto.
O QUE SE FAZ. Os 10 escores por família em cache (data_local/escores_fiel8, desde 01/02/2025: fiel = PCA(0,95) do bundle, pca3, ridge, gbm, ocsvm, iforest, knn, lof, gmm, ecod), canais t (14 temperaturas) e p (12 pressões), n de treino 20.000 e 19.000; a política fixa; o limiar por `calibra_limiar` para 407 h fora; os 8 eventos (INICIO = 01/02/2025). Reporta-se, por célula: detecção, início, FP, FP/mês, horas totais. Mais um escore COMBINADO por variante: o máximo, entre t e p, do escore de cada família dividido pelo seu p99 vigiado do período anterior ao mês (causal: o p99 dos últimos 60 dias vigiados antes de cada mês, para os dois escores 'fiel' apenas), para ver se o par t+p isolado também fecha a lacuna do EXP 05 (a consolidação do alarme).
CRITÉRIOS (fixados antes). Referência = o escore fiel do mesmo canal e da mesma variante. GANHO = detecção >= referência + 1 nas DUAS variantes de n; EQUIVALENTE = a +-1 em ambas; PIOR = o resto. Piso de ruído: ±2 eventos com 8 eventos (EXP 32).
EXPECTATIVAS: a referência fiel isolada a 407 h tem 4 a 5 de 8 em t e em p ~65% (no EXP 05 o canal t sozinho dava 5/7); algum escore alternativo dá GANHO em algum canal ~12% (os de resíduo por sensor ~7%; os de densidade ~5%); o ocsvm é o melhor das densidades ~25%; o escore combinado t+p isolado dá <= 5/8 ~75% (a lacuna da consolidação do alarme é de ~4 eventos).
INTERPRETAÇÃO PREVISTA. Se nenhum dá GANHO nesta comparação justa, a resposta à hipótese do usuário é: em escores por família, o PCA com o máximo por sensor é o melhor ou empata com os alternativos testados, e a melhoria tem de vir de outra parte (a fusão dos canais ou os dados). Se algum der GANHO, vale o placebo e uma fusão com ele.
LIMITES. 8 eventos (±2 de ruído), um único tamanho de treino por variante, o custo medido com os rótulos (as horas fora das janelas), uma política simples que não é a de produção; nada se adota sem sombra.

RESULTADO (16/10/2026) -- NO CANAL t O PCA FIEL ISOLADO É CLARAMENTE O MELHOR; NO p TODOS SÃO FRACOS E O 'GANHO' FORMAL DE DOIS ESCORES É UM EVENTO, DENTRO DO RUÍDO.
  Política simples fixa, limiar por bisseção para 407 h de alarme fora das janelas, 8 eventos (detecção | FP | FP/mês), n = 20.000 e 19.000. Canal t: fiel 7/8, 18 FP (1,61/mês) e 6/8, 22 FP (1,97) | pca3 3/8, 12 FP e 2/8, 15 | ridge 1/8 e 1/8 (11 FP) | gbm 2/8, 18 FP e 1/8, 15 | ocsvm 1/8 e 1/8 (12 e 10 FP) | iforest 3/8, 26 FP e 3/8, 20 | knn 2/8 e 1/8 (12 e 10 FP) | lof 4/8, 34 FP e 3/8, 23 | gmm 1/8 e 1/8 | ecod 3/8, 20 FP e 2/8, 15. Todos PIORES que a referência (7 e 6). Canal p: fiel 2/8, 19 FP e 2/8, 20 | pca3 1/8 e 1/8 | ridge 2/8 e 2/8 | gbm 2/8 e 2/8 | ocsvm 3/8, 21 FP e 3/8, 27 | iforest 2/8 e 1/8 (35 e 36 FP) | knn 1/8 e 2/8 | lof 1/8 e 0/8 | gmm 1/8 e 2/8 | ecod 3/8, 32 FP e 3/8, 32. Pelo critério escrito, o ocsvm e o ecod no canal p dão GANHO (3 contra 2 nas duas variantes) e os demais EQUIVALENTE ou PIOR.
  COMBINADO t+p (fiel; máximo dos dois escores normalizados pelo p99 vigiado dos 60 dias anteriores): 3/8, 19 FP (n = 20.000) e 2/8, 17 FP: muito pior que o t sozinho (7/8 e 6/8) e que o publicado com o voto (7/8, 9 FP).
LEITURA. (1) A hipótese 'o PCA é simples demais' NÃO se sustenta em escores POR FAMÍLIA: no canal t, o resíduo do PCA fiel isolado dá 7/8 e 6/8 a 407 h, e as nove alternativas (resíduo por sensor, kernel, florestas, mistura, vizinhança, ECOD) dão 1 a 4 de 8; a diferença é de 3 a 6 eventos, muito acima do ruído (±2). (2) O 'GANHO' formal do ocsvm e do ecod no p é de UM evento (3 contra 2/8) num canal em que TODOS têm 0 a 3 de 8, com FP de 21 a 32 contra 19-20 da referência: está abaixo do piso de ruído de ±2 declarado no pré-registro, é uma de 18 comparações (9 escores x 2 canais; esperam-se ganhos formais por acaso) e não é um ganho. (3) A consolidação do alarme é onde está o valor: o t isolado já dá 7/8 a 407 h, mas com 18 FP; o voto de produção dá o mesmo 7/8 com 9 FP (metade). Um máximo ingênuo de t e p é muito pior (3/8): fusão ingênua não presta; a de produção (voto + CUSUM + refratário + duração) é boa. (4) Logo a próxima pergunta não é o escore, é a FUSÃO: uma fusão baseada em princípio (probabilidades de cauda combinadas e uma estatística sequencial) faz melhor que o voto 3 de 4, a custo igual?
PLACAR: ACERTEI: o combinado t+p isolado <= 5/8 (~75%); algum GANHO formal (~12%: dois no canal p, mas dentro do ruído). ERREI: a referência fiel isolada de 4 a 5 de 8 em t e p (~65%: t deu 7 e 6, p deu 2); o ocsvm o melhor das densidades (~25%: em p sim, em t o pior com 1/8).

Uso:  PYTHONPATH=src .venv/bin/python experiments/33_escores_isolados_a_custo_igual.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E32 = _carrega("exp32", "32_escores_alternativos_a_custo_igual.py"); E6 = E32.E6
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.politica import ewma_vigiado, episodios, alarme_de, calibra_limiar

ROOT = E32.ROOT; INICIO8 = E32.INICIO8; ESC = E32.ESC; NS = E32.NS; ALVO = 407.0


def avalia_a_custo(s, vig, reg):
    ini = INICIO8 - pd.Timedelta(days=10); s = s.loc[ini:]; v = vig.loc[ini:]
    e = ewma_vigiado(s, v); thr = calibra_limiar(e, v, reg, ALVO, inicio=INICIO8); al = alarme_de(episodios(e, thr), e.index)
    return avalia(al, v, reg, inicio_periodo=INICIO8)


def main():
    g = nvio.carrega_grade(); reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c.reindex(g.index).fillna(False)
    print("ESCORES ISOLADOS, política simples, limiar por bisseção para 407 h fora (detecção de 8 | FP | FP/mês | h totais)", flush=True)
    tab = {}
    for c in ("t", "p"):
        for nm in ESC:
            lin = []
            for n in NS:
                s = pd.read_parquet(ROOT / "escores_fiel8" / f"{c}_{nm}_{n}.parquet")["s"]; r = avalia_a_custo(s, vig, reg); tab[(c, nm, n)] = r
                lin.append(f"n={n}: {r.det}/8, início {r.inicio}, FP {r.fp}, {r.fp_mes:.2f}/mês, {r.horas_alarme:.0f} h")
            print(f"   {c} <- {nm:7s} | " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO contra a referência fiel do mesmo canal", flush=True)
    for c in ("t", "p"):
        ref = [tab[(c, "fiel", n)].det for n in NS]; print(f"   [{c}] referência fiel isolada a 407 h: detecção {ref}", flush=True)
        for nm in ESC[1:]:
            d = [tab[(c, nm, n)].det for n in NS]; cl = "GANHO" if all(x >= y + 1 for x, y in zip(d, ref)) else "EQUIVALENTE" if all(abs(x - y) <= 1 for x, y in zip(d, ref)) else "PIOR"
            print(f"      {nm:7s}: detecção {d} -> {cl}", flush=True)
    # combinado t+p (so o fiel): max dos dois escores normalizados pelo p99 vigiado dos 60 dias anteriores a cada mes
    print("\nCOMBINADO t+p (fiel): máximo dos dois escores, cada um dividido pelo p99 vigiado dos 60 dias anteriores ao mês", flush=True)
    for n in NS:
        st = pd.read_parquet(ROOT / "escores_fiel8" / f"t_fiel_{n}.parquet")["s"]; sp = pd.read_parquet(ROOT / "escores_fiel8" / f"p_fiel_{n}.parquet")["s"]; out = []
        for c, s in (("t", st), ("p", sp)):
            z = pd.Series(np.nan, index=s.index)
            for m0 in pd.date_range(s.dropna().index[0].normalize().replace(day=1), s.index[-1], freq="MS", tz="UTC"):
                ant = s.loc[m0 - pd.Timedelta(days=60):m0][vig.loc[m0 - pd.Timedelta(days=60):m0].to_numpy()].dropna(); m1 = m0 + pd.offsets.MonthBegin(1)
                if len(ant) > 500:
                    z.loc[m0:m1 - pd.Timedelta("2min")] = s.loc[m0:m1 - pd.Timedelta("2min")] / float(ant.quantile(0.99))
            out.append(z)
        comb = pd.concat(out, axis=1).max(axis=1, skipna=False); r = avalia_a_custo(comb, vig, reg)
        print(f"   n={n}: {r.det}/8 | início {r.inicio} | FP {r.fp} | {r.fp_mes:.2f}/mês | totais {r.horas_alarme:.0f} h", flush=True)
    print("   (para comparar, o publicado completo, com o voto: 7/8, início 6, 9 FP, 0,81/mês, 1.457 h totais)", flush=True)


if __name__ == "__main__":
    main()
