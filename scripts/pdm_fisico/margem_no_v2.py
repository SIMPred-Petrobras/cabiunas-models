#!/usr/bin/env python3
"""A MARGEM AO SETPOINT sobre o v2 -- a medida que faltava.

POR QUE ESTE SCRIPT. A margem consumida ate o setpoint de trip foi a primeira
alavanca a deslocar a fronteira depois de ~25 refutadas: o TI_0305 a 20% da
margem cobre 5/8 na banda acionavel com duty de 4,5% (6,18x, p=0,0006). Mas foi
medida como 5o canal do **v1** -- voto >=2, um nivel so, que dava banda 3/8. O
ponto publicado hoje e o v2 (gatilho de dois niveis), que ja da 5/8. A margem
nunca foi testada sobre ele, e a pergunta em aberto e se ela ainda acrescenta.

POR QUE ELA E O COMPLEMENTO CERTO, E NAO MAIS UM CANAL. O diagnostico de fundo
do nosso detector e duty alto demais: os quatro canais ficam acesos 24-52% do
tempo, o voto satura, os episodios fundem e o nascimento sai da janela de 48 h.
A margem tem o perfil oposto -- 4,5% de duty -- e mede contra outra referencia:
o LIMITE DE PROTECAO PROJETADO, nao a normalidade aprendida. No MESMO sensor, a
referencia de engenharia cobre 5/8 onde o nosso `sp` cobre 2/8 com 41,75% de duty.

RESULTADO: NEGATIVO. Nenhuma das 36 formas (6 x 3 limiares x 2 tipicos) melhora o
v2; o melhor caso e empate exato (`Bgate`, a margem e redundante como portao) e o
resto custa FP ou perde deteccao. Mais duas frentes testadas e tambem negativas,
documentadas ao pe deste arquivo: uniao NAS SAIDAS e inversao da ordem do
pos-processamento.

O DIAGNOSTICO -- e sutil, porque os sinais SAO complementares:

    trip          v2 na banda    margem>=20% na banda
    2025-04-29         --               SIM
    2026-02-26        SIM                --

Cada um pega um evento que o outro nao pega, entao a uniao "deveria" dar 6/8. Nao
da, e o motivo esta no pos-processamento. A margem crua pega o 29/04 nascendo em
28/04 22:00 (lead 5,1 h, 304 min de duracao). Mas:

    margem + duracao so        -> pega  (o episodio de 304 min sobrevive)
    margem + refratario so     -> pega  (sobra o de 27/04, 34 min)
    margem + refratario + dur  -> NAO pega

O refratario admite o episodio de 34 min, abre 72 h de bloqueio e mata o de 304 min
que vinha depois; ai a duracao mata o de 34 min. Nao sobra nada. **A complementaridade
existe nos sinais e e destruida pela camada de decisao** -- exatamente o padrao da
fronteira ([[o-detector-esta-numa-fronteira]]).

SEIS FORMAS DE INCORPORAR, porque "somar um canal ao voto" nao e a unica:
  A5_3   nivel A vira >=3 de 5          (margem dilui o quorum)
  A5_4   nivel A vira >=4 de 5          (mantem a exigencia proporcional)
  Bgate  portao do nivel B vira sp|vb|mg
  B5     nivel B vira >=2 de 5
  C      NIVEL C: a margem sozinha dispara  (so faz sentido com duty baixo)
  C2     nivel C exigindo mais um canal qualquer junto

TIPICO CAUSAL. O `tipico` da margem e a mediana do sensor em operacao. Medi-la na
serie inteira vaza o futuro. Aqui ela e EXPANDING (so com o passado) -- e a forma
que vai para producao, porque o bundle so pode carregar o que ja aconteceu. A
versao global entra como controle, para separar "a margem funciona" de "a margem
funciona porque viu o futuro".

Uso:  PYTHONPATH=. python margem_no_v2.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, mask, idx, alvo, op, sel, EW, cru
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K_LO, VOTO_LO,
                             VOTO_HI, REFRAT_V2, DUR_MIN, ESC_IDADE, ESC_ABS,
                             ESC_DUR, HL, TMIN_BANDA)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

JAN48 = pd.Timedelta(hours=48)
KH = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
N_GAP = int(pd.Timedelta(hours=AV.GAP_EP_H) / pd.Timedelta("2min")) + 1
paradas = paradas_reais_2h()
part = op & ~op.shift(fill_value=False)
reset = ((~mask) | part).to_numpy()
LIM = pd.read_csv("limites_alarme.csv")
SENSOR = "954005_624_TI_0305"          # mancal radial LNA: tipico 70,25 -> trip 125


def margem(col: str, causal: bool) -> pd.Series:
    """Fracao da margem de protecao consumida. 0 = tipico, 1 = no limite."""
    r = LIM[LIM.col == col].iloc[0]
    s = g[col].astype("float64").where(mask)
    lim = r["HH"] if pd.notna(r["HH"]) else r["H"]
    if causal:
        # expanding: em cada instante, so o passado. Minimo de 1 semana de
        # operacao para a mediana significar alguma coisa.
        tip = s.expanding(min_periods=30 * 24 * 7).median()
    else:
        tip = pd.Series(float(s.median()), index=idx)
    return ((s - tip) / (lim - tip)).clip(lower=0)


def canal(c: str, k: float) -> pd.Series:
    """Degrau sustentado OU CUSUM, igual aos quatro originais."""
    thr = BASE[c] * k
    E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    return (deg | cu) & mask


A = {c: canal(c, K_LO[c]) for c in SIN}
B = {c: canal(c, KH[c]) for c in SIN}
FORCA = pd.concat([EW[c].where(mask) / (BASE[c] * KH[c]) for c in SIN], axis=1).max(axis=1)


def decide(voto: pd.Series) -> pd.Series:
    """Escalada por idade + refratario com furo + duracao. Identico ao v2."""
    f = FORCA.fillna(0.0).to_numpy()
    v = voto.to_numpy().copy()
    n_id = int(ESC_IDADE * 30)
    dentro, ini, ja = False, 0, False
    for i in range(len(v)):
        if not v[i]:
            dentro, ja = False, False
            continue
        ac = f[i] > ESC_ABS
        if not dentro:
            dentro, ini, ja = True, i, ac
            continue
        if ac and not ja and (i - ini) >= n_id:
            v[max(ini + 1, i - N_GAP):i] = False
            ini = i
        ja = ac
    voto = pd.Series(v, index=idx)
    al = pd.Series(False, index=idx)
    bloq = ini_b = None
    fortes = []
    for a, b in AV.episodios(voto):
        forte = float(FORCA.loc[a:b].max()) > ESC_ABS
        velho = ini_b is not None and (a - ini_b).total_seconds() / 3600 >= ESC_IDADE
        if bloq is not None and a <= bloq and not (forte and velho):
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=REFRAT_V2)
        ini_b = a
        if forte:
            fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        d = (b - a).total_seconds() / 60 + 2
        if (any(x >= a and y <= b for x, y in fortes) and d >= ESC_DUR) or d >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def mede(fin: pd.Series) -> dict:
    eps = AV.episodios(fin)
    det = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    hfp = sum((b - a).total_seconds() / 3600 for a, b, k, _ in cls if k == "FP")
    mes = det["horas_op"] / 730.0
    leads = [(t - max([a for a, _ in eps if t - JAN48 <= a <= t])).total_seconds() / 3600
             for t in alvo if any(t - JAN48 <= a <= t for a, _ in eps)]
    return dict(
        banda=sum(1 for t in alvo if any(t - JAN48 <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                         for a, _ in eps)),
        inicio=len(leads), det=det["det"], fp=nfp / mes, h=hfp / mes,
        eps=len(eps), lead=float(np.mean(leads)) if leads else float("nan"))


def voto_v2(mg: pd.Series | None, forma: str) -> pd.Series:
    """O voto do v2, com a margem entrando de uma das seis formas."""
    nA = sum(A[c].astype(int) for c in SIN)
    nB = sum(B[c].astype(int) for c in SIN)
    vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
    vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"])
    if mg is None:
        return vA | vB
    m = mg & mask
    if forma == "A5_3":
        vA = pd.Series((nA + m.astype(int)) >= 3, index=idx) & mask
    elif forma == "A5_4":
        vA = pd.Series((nA + m.astype(int)) >= 4, index=idx) & mask
    elif forma == "Bgate":
        vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"] | m)
    elif forma == "B5":
        vB = (pd.Series((nB + m.astype(int)) >= VOTO_HI, index=idx) & mask
              & (B["sp"] | B["vb"] | m))
    elif forma == "C":
        return vA | vB | m
    elif forma == "C2":
        return vA | vB | (m & (pd.Series(nA >= 1, index=idx) & mask))
    return vA | vB


if __name__ == "__main__":
    base = mede(decide(voto_v2(None, "")))
    print(f"{'':>34}{'banda':>7}{'inicio':>8}{'det':>6}{'FP/mes':>9}"
          f"{'h/mes':>8}{'lead':>8}{'eps':>6}")
    print("-" * 92)
    print(f"{'v2 publicado (controle)':>34}{base['banda']:>5}/8{base['inicio']:>6}/8"
          f"{base['det']:>4}/8{base['fp']:>9.3f}{base['h']:>8.1f}"
          f"{base['lead']:>7.1f}h{base['eps']:>6}")

    for causal in (True, False):
        m_raw = margem(SENSOR, causal)
        rot = "tipico CAUSAL (vai para producao)" if causal else "tipico global (controle, vaza)"
        print(f"\n  ---- {rot} ----")
        print(f"       duty do sensor por limiar: " + "  ".join(
            f"{th:.0%}={100*float(((m_raw >= th) & mask).sum())/int(mask.sum()):.1f}%"
            for th in (0.15, 0.20, 0.25, 0.30)))
        for th in (0.15, 0.20, 0.25):
            mg = (m_raw >= th).fillna(False)
            for forma in ("A5_3", "A5_4", "Bgate", "B5", "C", "C2"):
                r = mede(decide(voto_v2(mg, forma)))
                d = []
                if r["banda"] > base["banda"]: d.append(f"+{r['banda']-base['banda']} banda")
                if r["banda"] < base["banda"]: d.append(f"{r['banda']-base['banda']} banda")
                if r["det"] < base["det"]: d.append(f"PERDE det")
                marca = "   <<< " + ", ".join(d) if d else ""
                print(f"{f'margem>={th:.0%}  {forma}':>34}{r['banda']:>5}/8{r['inicio']:>6}/8"
                      f"{r['det']:>4}/8{r['fp']:>9.3f}{r['h']:>8.1f}"
                      f"{r['lead']:>7.1f}h{r['eps']:>6}{marca}")


# ─────────────────────────────────────────────────────────────────────────────
# AS OUTRAS DUAS FRENTES, tambem negativas. Ficam aqui para nao serem refeitas.
#
# 2) UNIAO NAS SAIDAS (cada detector com seu proprio pos-processamento, e so
#    depois a uniao dos episodios). A ideia era escapar da fusao no voto. Melhor
#    caso medido -- margem>=25%, refratario 72 h, duracao 120 min:
#
#        v2 sozinho              banda 5/8  inicio 6/8  0,344 FP/mes   6,6 h/mes
#        v2 U margem             banda 5/8  inicio 7/8  0,431 FP/mes   7,0 h/mes
#
#    Compra +1 na regua de inicio por +25% de FP, e NAO move a banda -- que e a
#    regua que decide acionabilidade. As 12 combinacoes ficaram todas em 5/8.
#
# 3) INVERTER A ORDEM DO POS-PROCESSAMENTO (duracao antes do refratario). Parecia
#    corrigir o mecanismo descrito acima: so alarme de verdade consumiria o
#    refratario. Medido, e muito pior:
#
#        ordem atual (refratario -> duracao)   0,344 FP/mes    6,6 h/mes
#        ordem nova  (duracao -> refratario)   0,861 FP/mes   49,6 h/mes
#
#    Nao era bug. O episodio curto que consome o refratario funciona como
#    AMORTECEDOR: ele bloqueia os episodios seguintes e e por isso que o custo e
#    baixo. Tirar o amortecedor solta tudo que vinha depois -- 7,5x as horas de
#    alarme falso, sem um unico evento a mais. A ordem fica como esta.
