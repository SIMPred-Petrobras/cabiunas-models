#!/usr/bin/env python3
"""O gargalo e o CRITERIO de selecao, nao o tamanho do espaco. Testando criterios.

custo_da_busca.py mostrou que buscar mais NAO piora -- a curva sobe e satura, e o desvio
entre sorteios cai de 1,94 para 0,53. Mas a busca sobre as 1.512 configuracoes converge
para LOEO 5/8, enquanto o ponto montado por raciocinio (k=1,7 k_vib=2,2 R=48h D=60min) da
8/8. Logo o problema nao e sobreajuste da busca: e que o criterio "maxima deteccao sob
orcamento de FP" tem teto de 5/8.

A hipotese: o ponto bom tem uma propriedade que esse criterio nao enxerga -- ele esta num
PLATO. Foi platô que aprovou o refratario (det=8 em R=0,12,24,36,48h) e que reprovou o
piso de 1,6 degC, a faixa R=120h e o voto entre sondas (todos com o resultado vivendo num
unico valor da grade). Um criterio que exija estabilidade local deveria preferir
configuracoes que generalizam, que e exatamente o que o LOEO mede.

Criterios comparados, todos sob o mesmo orcamento de custo:
  C1  deteccao maxima                                    (o de antes, teto 5/8)
  C2  deteccao ROBUSTA = pior caso entre a config e seus vizinhos de grade
  C3  deteccao maxima, desempate por tamanho do plato
  C4  deteccao maxima, desempate por lead
  C5  deteccao robusta, desempate por lead

Vizinho = difere um passo em UM eixo da grade (kb, kv, R, D, teto) -> ate 10 vizinhos.
C2/C3/C5 avaliam a vizinhanca do candidato, o que uma busca cuidadosa faria de qualquer
forma; custa ~10 avaliacoes extras por candidato, nao muda a ordem de grandeza.
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from auto_reset import trunca
import reduz_fp as RF

CACHE = "criterio_busca_cache.npz"
T0 = pd.Timestamp("2024-02-01 00:00", tz="UTC")
JAN = pd.Timedelta(hours=48)
KB = [1.2, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0]
KV = [1.6, 2.2, 2.8, 3.5]
RS = [0, 12, 24, 48, 72, 120]
DS = [0, 60, 120]
TE = [0, 12, 24]
EIXOS = [KB, KV, RS, DS, TE]
ORC = 2.5
RNG = np.random.default_rng(7)


def constroi():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df) & (idx >= T0)
    alvo = list(todas[todas >= T0])
    meses = mask.sum() * 2 / 60 / 730.0
    out = roda(BRACO, df, todas)
    DETV, LEAD, EPS = [], [], []
    for kb in KB:
        for kv in KV:
            base = alerta_2k(out, mask, kb, kv) & mask
            for Rh in RS:
                aR = RF.refratario(base, Rh) if Rh else base
                for D in DS:
                    aD = RF.dur_min(aR, D) if D else aR
                    for te in TE:
                        al = trunca(aD, te) if te else aD
                        eps = A.episodios(al)
                        d, l = [], []
                        for t in alvo:
                            ini = [a for a, b in eps if a <= t and b >= t - JAN]
                            d.append(bool(ini))
                            l.append((t - min(ini)).total_seconds()/3600 if ini else 0.0)
                        DETV.append(d); LEAD.append(l); EPS.append(eps)
        print(f"  k_base={kb} reduzido ({len(DETV)} configs)", flush=True)
    DETV = np.array(DETV); LEAD = np.array(LEAD)
    FP = np.empty((len(alvo), len(EPS)))
    for i in range(len(alvo)):
        jw = [(t - JAN, t) for j, t in enumerate(alvo) if j != i]
        for k, eps in enumerate(EPS):
            FP[i, k] = sum(1 for a, b in eps
                           if not any(a <= t1 and b >= t0 for t0, t1 in jw)) / meses
    np.savez_compressed(CACHE, DETV=DETV, LEAD=LEAD, FP=FP, n_alvo=len(alvo))
    return DETV, LEAD, FP, len(alvo)


def main():
    if os.path.exists(CACHE):
        z = np.load(CACHE); DETV, LEAD, FP, n_alvo = z["DETV"], z["LEAD"], z["FP"], int(z["n_alvo"])
        print(f"cache {CACHE} reaproveitado", flush=True)
    else:
        DETV, LEAD, FP, n_alvo = constroi()
    N = DETV.shape[0]
    print(f"configuracoes: {N}   eventos no alvo: {n_alvo}\n", flush=True)

    # --- estrutura de vizinhanca na grade
    dims = [len(e) for e in EIXOS]
    coords = list(itertools.product(*[range(d) for d in dims]))
    pos = {c: i for i, c in enumerate(coords)}
    VIZ = []
    for c in coords:
        v = []
        for eixo in range(5):
            for passo in (-1, 1):
                cc = list(c); cc[eixo] += passo
                if 0 <= cc[eixo] < dims[eixo]:
                    v.append(pos[tuple(cc)])
        VIZ.append(np.array(v))

    def loeo(sub, criterio):
        ac = 0
        for i in range(n_alvo):
            outros = [j for j in range(n_alvo) if j != i]
            det_tr = DETV[:, outros].sum(axis=1)        # sobre TODAS, para os vizinhos
            lead_tr = LEAD[np.ix_(np.arange(N), outros)].mean(axis=1)
            fp = FP[i]
            ok = np.zeros(N, bool); ok[sub] = True
            ok &= (fp <= ORC)
            if not ok.any():
                continue
            if criterio == "C1":
                pont = det_tr * 1000.0 - fp
            elif criterio == "C2":
                rob = np.array([min(det_tr[k], det_tr[VIZ[k]].min()) for k in range(N)])
                pont = rob * 1000.0 - fp
            elif criterio == "C3":
                plat = np.array([(det_tr[VIZ[k]] >= det_tr[k]).sum() for k in range(N)])
                pont = det_tr * 1000.0 + plat * 10.0 - fp
            elif criterio == "C4":
                pont = det_tr * 1000.0 + lead_tr
            else:  # C5
                rob = np.array([min(det_tr[k], det_tr[VIZ[k]].min()) for k in range(N)])
                pont = rob * 1000.0 + lead_tr
            pont = np.where(ok, pont, -np.inf)
            ac += bool(DETV[int(np.argmax(pont)), i])
        return ac

    NOMES = {"C1": "detecção máxima", "C2": "detecção robusta (pior vizinho)",
             "C3": "detecção, desempate por platô", "C4": "detecção, desempate por lead",
             "C5": "detecção robusta + lead"}
    print("=" * 96)
    print(f"LOEO ANINHADO POR CRITERIO E POR TAMANHO DO ESPACO (orcamento <= {ORC} FP/mes)")
    print("=" * 96)
    print(f"{'criterio':34s} " + " ".join(f"{s:>7}" for s in [30, 100, 300, 1000, N]))
    todos = np.arange(N)
    tab = []
    for cr, nome in NOMES.items():
        linha = []
        for S in [30, 100, 300, 1000, N]:
            n = 1 if S >= N else 20
            v = [loeo(np.sort(RNG.choice(todos, S, replace=False)) if S < N else todos, cr)
                 for _ in range(n)]
            linha.append(np.mean(v))
        print(f"{nome:34s} " + " ".join(f"{x:7.2f}" for x in linha), flush=True)
        tab.append(dict(criterio=cr, nome=nome, **{f"S{s}": x for s, x in zip([30,100,300,1000,N], linha)}))
    pd.DataFrame(tab).to_csv("criterio_busca.csv", index=False)

    print(f"\n  (o alvo tem {n_alvo} eventos; o ponto montado por raciocinio da LOEO {n_alvo}/{n_alvo})")
    i_ref = pos[(KB.index(1.7), KV.index(2.2), RS.index(48), DS.index(60), TE.index(0))]
    print(f"\n  a busca no espaco completo escolhe, por criterio:")
    for cr in NOMES:
        outros = list(range(1, n_alvo))
        det_tr = DETV[:, outros].sum(axis=1); fp = FP[0]
        ok = fp <= ORC
        if cr == "C1": pont = det_tr*1000.0 - fp
        elif cr == "C2":
            rob = np.array([min(det_tr[k], det_tr[VIZ[k]].min()) for k in range(N)]); pont = rob*1000.0 - fp
        elif cr == "C3":
            plat = np.array([(det_tr[VIZ[k]] >= det_tr[k]).sum() for k in range(N)]); pont = det_tr*1000.0 + plat*10.0 - fp
        elif cr == "C4": pont = det_tr*1000.0 + LEAD[:, outros].mean(axis=1)
        else:
            rob = np.array([min(det_tr[k], det_tr[VIZ[k]].min()) for k in range(N)]); pont = rob*1000.0 + LEAD[:, outros].mean(axis=1)
        k = int(np.argmax(np.where(ok, pont, -np.inf)))
        c = coords[k]
        print(f"    {cr}: k_base={KB[c[0]]} k_vib={KV[c[1]]} R={RS[c[2]]}h D={DS[c[3]]}min "
              f"teto={TE[c[4]]}h   (o recomendado e o indice {i_ref}, escolhido {k==i_ref})")


if __name__ == "__main__":
    main()
