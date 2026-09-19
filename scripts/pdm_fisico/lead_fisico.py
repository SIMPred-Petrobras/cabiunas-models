#!/usr/bin/env python3
"""Quanto tempo antes do trip o SINAL comeca a se mover? -- o teto do detector.

POR QUE ESTA PERGUNTA, E POR QUE AGORA. Depois de ~40 alavancas refutadas e da
licao de que com 8 eventos e 4 FP toda regra encontrada cabe no erro amostral
([[piso-de-forca-candidata-nao-aplicada]]), tentar mais uma otimizacao as cegas
seria repetir o erro. A pergunta util nao e "que ajuste fazer" -- e "quanto ainda
ha para ganhar".

O detector tem um LEAD DO ALARME: quando o episodio nasce. O que ninguem mediu e
o LEAD FISICO: quando o sinal comeca a se desviar do normal, independentemente de
qualquer limiar, voto ou pos-processamento.

  * se o lead fisico for MUITO MAIOR que o do alarme, existe informacao na serie
    que a camada de decisao esta jogando fora -- e vale continuar mexendo nela;
  * se forem parecidos, o detector ja extrai o que ha, e nenhum ajuste de regra
    vai adiantar. O teto seria dos SINAIS, nao das REGRAS.

COMO MEDIR SEM CIRCULARIDADE. Nao se pode usar o limiar do detector, senao a
resposta vem por construcao. Aqui o criterio e independente: para cada canal, a
referencia e a distribuicao do PROPRIO canal nas 30 dias anteriores a janela de
busca (excluindo a janela), e o desvio conta quando o sinal passa do percentil 95
dessa referencia e FICA acima por 2 h seguidas. E um criterio frouxo de proposito:
mede quando a serie muda, nao quando o detector decide.

Os tres eventos que o v2 perde na banda sao o foco:
  27/02  o alarme nasce 1,6 h antes -- abaixo do tau_min de 4 h
  17/03  alarme de pe, nao nasce na janela
  29/04  precursor comeca ~51,8 h antes e perde a janela de 48 h por 3,8 h

RESULTADO: A MEDIDA NAO SOBREVIVE AO CONTROLE NEGATIVO.

Nos 8 trips o sinal parecia se mover MUITO antes do alarme:

    trip          lead fisico   lead do alarme   "sobra"
    27/02/2025       144,4h          1,6h        142,8h
    17/03/2025       168,0h          --             --
    11/04/2025       168,0h          4,1h        163,9h
    ...
    mediana          162,2h                      141,9h

Cento e quarenta horas de informacao aparentemente jogada fora pela camada de
decisao. Mas o controle negativo -- a MESMA medida em 200 instantes aleatorios em
regime, a pelo menos 14 dias de qualquer trip -- desmente:

                     trips          nulo
    cruzam            8/8       140/200 (70%)
    mediana         162,2h         135,0h

E **34,3% das janelas SEM TRIP tambem mostram "desvio ha 162 h ou mais"**. Por
canal a razao observado/nulo fica entre 1,16x e 1,92x, fraca demais para sustentar
qualquer coisa.

O QUE ISSO SIGNIFICA, e responde a pergunta que motivou o script: nao ha
informacao de INICIO sendo desperdicada. O sinal fica alto boa parte do tempo --
os canais tem duty de 24% a 52% ([[o-detector-esta-numa-fronteira]]) -- e e
exatamente por isso que o detector precisa de voto e pos-processamento para ser
seletivo. A "sobra" media a mesma coisa que o duty alto, por outro caminho.

CONSEQUENCIA PRATICA: o teto nao esta na camada de decisao. Mexer em limiar, voto,
refratario ou duracao nao vai destravar 140 h de antecedencia que nao existem. O
teto e dos SINAIS, e isso fecha mais uma linha de investigacao.

Uso:  PYTHONPATH=. python lead_fisico.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import cru, mask, idx, alvo
from publica_clearml import HL, TMIN_BANDA
from margem_no_v2 import decide, voto_v2

SIN = ["t", "p", "sp", "vb"]
BUSCA = pd.Timedelta(days=7)        # ate onde procurar o inicio do desvio
REF = pd.Timedelta(days=30)         # referencia: 30 dias antes da janela de busca
SUST = 60                           # 2 h em passos de 2 min
JAN48 = pd.Timedelta(hours=48)

EW = {c: cru[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean() for c in SIN}


def inicio_desvio(c: str, t: pd.Timestamp) -> float:
    """Horas antes de `t` em que o canal `c` passa do p95 da propria referencia
    e fica acima por 2 h. NaN se nao cruzar dentro da janela de busca."""
    s = EW[c].where(mask)
    ref = s.loc[t - BUSCA - REF: t - BUSCA].dropna()
    if len(ref) < 500:
        return float("nan")
    lim = float(np.percentile(ref, 95))
    w = s.loc[t - BUSCA: t]
    acima = (w > lim).fillna(False)
    corrido = acima.rolling(SUST, min_periods=SUST).sum() >= SUST
    hit = corrido[corrido]
    if hit.empty:
        return float("nan")
    # volta ao inicio do trecho sustentado
    p = hit.index[0] - pd.Timedelta(minutes=2 * (SUST - 1))
    return (t - p).total_seconds() / 3600


if __name__ == "__main__":
    fin = decide(voto_v2(None, ""))
    eps = AV.episodios(fin)
    print(f"{'trip':<18}{'t':>9}{'p':>9}{'sp':>9}{'vb':>9}   "
          f"{'FISICO':>9}{'ALARME':>9}   sobra")
    print("-" * 86)
    linhas = []
    for t in alvo:
        leads = {c: inicio_desvio(c, t) for c in SIN}
        fis = np.nanmax([v for v in leads.values()] + [np.nan])
        nasc = [a for a, _ in eps if t - JAN48 <= a <= t]
        alm = (t - max(nasc)).total_seconds() / 3600 if nasc else float("nan")
        sobra = fis - alm if np.isfinite(fis) and np.isfinite(alm) else float("nan")
        na_banda = np.isfinite(alm) and alm >= TMIN_BANDA
        marca = "" if na_banda else "   <-- fora da banda"
        print(f"{t:%Y-%m-%d %H:%M}" +
              "".join(f"{leads[c]:>9.1f}" if np.isfinite(leads[c]) else f"{'--':>9}"
                      for c in SIN) +
              f"   {fis:>8.1f}h" +
              (f"{alm:>8.1f}h" if np.isfinite(alm) else f"{'--':>9}") +
              (f"{sobra:>8.1f}h" if np.isfinite(sobra) else f"{'--':>9}") + marca)
        linhas.append((t, fis, alm, sobra, na_banda))

    print("-" * 86)
    fs = [f for _, f, _, _, _ in linhas if np.isfinite(f)]
    sb = [s for _, _, _, s, _ in linhas if np.isfinite(s)]
    print(f"  lead FISICO  : mediana {np.median(fs):.1f}h   faixa {min(fs):.1f} a {max(fs):.1f}h")
    print(f"  sobra (fisico - alarme): mediana {np.median(sb):.1f}h  "
          f"faixa {min(sb):.1f} a {max(sb):.1f}h")
    print(f"\n  eventos em que o sinal se move mas o alarme nao nasce na banda:")
    for t, f, a, s, ok in linhas:
        if not ok and np.isfinite(f):
            print(f"     {t:%Y-%m-%d}  sinal se move {f:.1f}h antes, "
                  f"alarme {'nasce ' + format(a, '.1f') + 'h antes' if np.isfinite(a) else 'nao nasce'}")
