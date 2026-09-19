#!/usr/bin/env python3
"""A Regra C precisa de um TETO DE DURACAO -- e o que muda quando tem.

O PROBLEMA. A Regra C nao conta como falso positivo o episodio seguido de parada
real em ate 48 h: o detector viu algo que a operacao tambem viu, e puni-lo seria
injusto. Mas a regra diz QUANDO perdoar e nao diz POR QUANTO TEMPO -- nao ha
limite de duracao.

Descoberto ao testar canais de dinamica ([[a-regra-c-nao-limita-duracao]]): um
canal novo parecia baixar o custo de 0,344 para 0,258 FP/mes, e o que acontecia
era a troca de um falso positivo de 7,4 h por um episodio de 139,83 h que a regra
perdoava inteiro. A conta melhorava porque as 140 horas nao eram contadas; a sala
de controle veria seis dias de alarme ininterrupto.

Isso e a segunda regua nossa a premiar o comportamento errado -- a primeira foi a
de inicio, creditando o 29/04 porque o detector viu MENOS.

O QUE ESTE SCRIPT FAZ. Poe um teto: episodio que passa de `teto_h` deixa de ser
perdoado e volta a contar como falso positivo, com todas as suas horas. Depois
remede o que ja foi decidido com a regua antiga -- v1 contra v2, e o ponto de
deploy contra o ponto otimo -- para ver se alguma conclusao vira.

A pergunta nao e "o numero piora" (vai piorar, porque para de esconder horas).
E "a ORDEM entre as opcoes muda?". Se nao muda, as decisoes ficam de pe e so o
numero publicado precisa ser corrigido.

RESULTADO. A regua esconde 492 h em 7 episodios neutros, dos quais quatro passam
de 48 h (153,8 / 135,2 / 112,4 / 52,4). Com teto, o custo do v2 vai de 6,6 para
45,6 h/mes -- SETE VEZES o que vinhamos publicando.

Mas AS DECISOES FICAM DE PE: o v2 bate o v1 por dois eventos na banda em todos os
tetos (5/8 contra 3/8), e o ponto de deploy segue empatado com o otimo. A ordem
entre as opcoes nao muda; so o numero publicado estava errado.

A SAIDA MELHOR QUE UM TETO ARBITRARIO: a metrica de CARGA -- horas de FP mais
horas de neutro, tudo que fica aceso sem preceder trip. Ela e INVARIANTE ao teto
(48,9 h/mes para o v2 com qualquer valor, inclusive sem teto), porque nao depende
de onde se corta o perdao. E foi ela que desmascarou o canal `vol 3h t`, que
parecia baixar o custo de 6,6 para 5,9 e na verdade sobe a carga de 48,9 para 60,3.

RECOMENDACAO: nao mexer na Regra C -- ela esta certa no que se propoe, que e nao
punir acerto. Reportar as duas coisas lado a lado:

    acuracia          0,344 FP/mes   -- quantas vezes alarma sem nada acontecer
    carga operacional  48,9 h/mes    -- o que o operador ve aceso

Nenhum dos dois sozinho conta a historia, e o segundo nunca foi reportado.

Uso:  PYTHONPATH=. python regra_c_com_teto.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import TMIN_BANDA, K_LO, K_LO_PONTO_OTIMO, reproduz
from plota_estilo_francisco import paradas_reais_2h

JAN = pd.Timedelta(hours=48)
paradas = paradas_reais_2h()


def classifica(eps, teto_h: float | None):
    """Regra C, opcionalmente com teto de duracao no perdao.

    teto_h=None reproduz a regra atual. Com teto, um episodio que seria NEUTRO mas
    dura mais que `teto_h` volta a contar como FP -- inteiro, com todas as horas.
    Perdoar duracao ilimitada e o furo que esta correcao fecha."""
    jw = [(t - JAN, t) for t in alvo]
    out = []
    for a, b in eps:
        dur = (b - a).total_seconds() / 3600 + 2 / 60
        if [t for t, (t0, t1) in zip(alvo, jw) if a <= t1 and b >= t0]:
            out.append((a, b, "TP", dur)); continue
        perto = len(paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]) > 0
        if perto and (teto_h is None or dur <= teto_h):
            out.append((a, b, "NEUTRO", dur))
        else:
            out.append((a, b, "FP", dur))
    return out


def mede(fin, teto_h):
    eps = AV.episodios(fin)
    det = AV.avalia(fin, alvo, mask)
    cls = classifica(eps, teto_h)
    fp = [(a, b, d) for a, b, k, d in cls if k == "FP"]
    ne = [(a, b, d) for a, b, k, d in cls if k == "NEUTRO"]
    mes = det["horas_op"] / 730.0
    ban = sum(1 for t in alvo if any(t - JAN <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    ini = sum(1 for t in alvo if any(t - JAN <= a <= t for a, _ in eps))
    return dict(banda=ban, inicio=ini, det=det["det"],
                fp=len(fp) / mes, h=sum(d for *_, d in fp) / mes,
                n_ne=len(ne), h_ne=sum(d for *_, d in ne) / mes,
                eps=len(eps), mes=mes)


def linha(nome, fin, teto):
    r = mede(fin, teto)
    print(f"{nome:<26}{r['banda']:>5}/8{r['inicio']:>6}/8{r['det']:>4}/8"
          f"{r['fp']:>9.3f}{r['h']:>8.1f}{r['n_ne']:>7}{r['h_ne']:>9.1f}")
    return r


if __name__ == "__main__":
    print("Quanto a regua esconde hoje: episodios NEUTROS por duracao\n")
    al_v2, *_ = reproduz(v2=True)
    cls = classifica(AV.episodios(al_v2), None)
    ne = sorted([(d, a, b) for a, b, k, d in cls if k == "NEUTRO"], reverse=True)
    print(f"{'duracao':>10}   inicio do episodio")
    for d, a, b in ne:
        marca = "   <-- acima de 48 h" if d > 48 else ("   <- acima de 24 h" if d > 24 else "")
        print(f"{d:>9.1f}h   {a:%Y-%m-%d %H:%M}{marca}")
    tot = sum(d for d, _, _ in ne)
    print(f"\n{len(ne)} episodios neutros, {tot:.0f} h no total -- "
          f"{sum(1 for d,_,_ in ne if d > 48)} passam de 48 h")

    print("\n" + "=" * 96)
    print("O QUE MUDA NAS DECISOES JA TOMADAS")
    print("=" * 96)
    al_v1, *_ = reproduz(v2=False)
    import publica_clearml as P
    k_guardado = dict(P.K_LO)
    P.K_LO = dict(K_LO_PONTO_OTIMO)
    al_otimo, *_ = reproduz(v2=True)
    P.K_LO = k_guardado

    for teto in (None, 72, 48, 24):
        rot = "SEM teto (regua atual)" if teto is None else f"teto de {teto} h"
        print(f"\n  ---- {rot} ----")
        print(f"{'':<26}{'banda':>7}{'inicio':>8}{'det':>6}{'FP/mes':>9}"
              f"{'h/mes':>8}{'neutros':>7}{'h neutras':>9}")
        a = linha("v1 (um nivel)", al_v1, teto)
        b = linha("v2 ponto otimo", al_otimo, teto)
        c = linha("v2 ponto de deploy", al_v2, teto)
        melhor = min([("v1", a), ("otimo", b), ("deploy", c)],
                     key=lambda x: (-x[1]["banda"], x[1]["fp"], x[1]["h"]))[0]
        print(f"{'':<26}-> melhor por banda, depois custo: {melhor}")
