#!/usr/bin/env python3
"""O GATE DE NAO-DECAIMENTO: separa a cauda do transiente de religamento da degradacao.

DE ONDE VEM A PERGUNTA (autopsia_fp.csv, medido, nao suposto).
Dos 23 episodios do ponto de operacao, 9 dos 15 falsos positivos comecam em
dist_partida = 6,4667 h -- o MESMO valor, ate a 13a casa. Nao e coincidencia: e
(180 + 14) x 2 min, ou seja, exatamente 6 h de blackout mais os 14 passos que faltam
para fechar o SUSTAIN de 30 min. Sao alarmes que disparam no PRIMEIRO instante em que
a mascara permite. E os tres episodios mais longos de falso positivo (153,7 h, 135,1 h
e 52,4 h -- que sozinhos respondem por metade das horas de alarme falso) estao nesse
grupo.

O MECANISMO. A EWMA e calculada sobre o sinal CRU, sem mascara -- ela atravessa o
blackout carregando o pico do religamento. Com meia-vida de 1 h (t, p) e 30 min
(sp, vb), seis horas derrubam o pico por um fator de 2^6 a 2^12, mas um transiente de
partida chega a duas ordens de grandeza acima do limiar. O que sobra ainda cruza. O
blackout nao apaga o transiente: ele apenas adia o instante em que o resto dele e visto.

POR QUE NAO E "ALONGAR O BLACKOUT". Tres DETECCOES nascem na mesma borda
(21/02/2025, 03/11/2025, 29/01/2026): sao religamentos que ja voltaram degradados --
um modo de falha real, e o unico caminho para o trip de 04/11. Empurrar a borda mata
os 9 falsos positivos e as 3 deteccoes junto. Isso ja foi varrido em blackout_curto.py
e fechado; nao e por ai.

A HIPOTESE NOVA. As duas populacoes tem GEOMETRIA oposta no instante da borda:
  - cauda de transiente: a EWMA esta CAINDO (relaxa monotonicamente desde o pico);
  - religamento degradado: a EWMA esta parada ou SUBINDO (a degradacao alimenta o sinal).
Entao o discriminante nao e nivel nem tempo -- e DERIVADA. Um sinal so pode acusar
enquanto nao estiver em decaimento ativo:

    decai_c(t) = E_c(t) < E_c(t - W) * (1 - delta)        # caiu de verdade em W

Bloqueia SO o decaimento ativo. Platô nao e bloqueado (E(t) ~ E(t-W)), subida nao e
bloqueada. Por isso o gate nao pode, por construcao, matar uma degradacao sustentada:
para escapar dele basta o sinal parar de cair.

O GATE VALE PARA ARMAR, NAO PARA MANTER -- e isso nao e detalhe. A primeira versao
aplicava ~decai continuamente sobre ON e o resultado foi o oposto do procurado: FP subiu
de 1,12 para 1,9-4,9 por mes em toda a grade. A causa e mecanica: a 2 min de passo uma
cauda em relaxamento nao cai de forma monotona, ela oscila, entao ~decai liga e desliga
e PARTE um episodio longo em varios curtos. Como o custo e contado por EPISODIO, cortar
um alarme em cinco pedacos multiplica o custo por cinco enquanto reduz as horas.
A forma certa e uma trava:

    arma_c   = (degrau_c | cusum_c) & mascara & ~decai_c    # so aqui pode COMECAR
    mantem_c = (degrau_c | cusum_c) & mascara               # daqui em diante SEGUE
    ON_c     = primeira ocorrencia de arma_c em diante, dentro de cada corrida de mantem_c

Assim o gate atrasa ou impede o inicio e nunca fragmenta o que ja comecou.

O gate e global, nao so pos-partida: qualquer cauda em relaxamento e tratada igual,
venha de religamento ou de manobra. Nao ha parametro dependente de partida.

ARMADILHA DE PARETO. Apertar o gate reduz a cobertura, o que mexe em deteccao E custo
ao mesmo tempo. Comparar a k fixo mediria sensibilidade, nao o gate. Aqui (kb, kv) sao
varridos DENTRO de cada (W, delta) e a leitura e (a) melhor deteccao dentro do orcamento
de FP e (b) deteccao a custo igualado com a linha de base.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import (GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM,
                             CARGA, REFRAT_H, DUR_MIN, T0, ORC_FP)
from blackout_curto import cusum

KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
KV = [1.8, 2.2, 2.8]
JANELAS = ["30min", "1h", "2h", "4h"]      # W do teste de decaimento
DELTAS = [0.0, 0.05, 0.10, 0.20]           # queda relativa exigida para chamar de decaimento
H_FP_BASE = 39.0                           # custo da linha de base, para leitura a custo igualado

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
mask = (estavel & ~blk) & sel
reset = ((~mask) | part).to_numpy()

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)
# EWMA sobre o sinal CRU -- e ela que atravessa o blackout, e e a sua derivada que o
# gate le. Calcular sobre o sinal mascarado apagaria justamente o que queremos medir.
EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}


def pos(voto):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def partes(kb, kv):
    """degrau e CUSUM por sinal, sem gate -- o que nao depende de (W, delta)."""
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


def decaimento(w, delta):
    """~decai_c: o sinal NAO caiu mais que delta ao longo de W."""
    n = int(pd.Timedelta(w) / pd.Timedelta(GRID))
    return {c: ~(EW[c] < EW[c].shift(n) * (1 - delta)).fillna(False) for c in SIN}


def trava(mantem, ok, modo):
    """Aplica o gate `ok` a `mantem` sem fragmentar, em dois graus de rigor.

    'trava': ON comeca na primeira amostra da corrida em que ok e verdadeiro e segue ate
             o fim dela. O gate ATRASA o inicio.
    'porta': a corrida inteira e admitida ou rejeitada pelo valor de ok na sua PRIMEIRA
             amostra. O gate DECIDE a excursao pela geometria com que ela nasceu.

    'trava' e permissivo demais quando a corrida e longa: numa corrida de 150 h basta uma
    amostra sem queda para armar, e uma cauda de transiente sempre tem uma. 'porta' e o
    teste honesto da hipotese -- esta excursao nasceu de um relaxamento ou de uma subida?
    """
    grp = (~mantem).cumsum()
    if modo == "trava":
        on = (ok & mantem).astype(int).groupby(grp).cummax().astype(bool)
    else:
        prim = mantem & ~mantem.shift(fill_value=False)      # 1a amostra de cada corrida
        on = (ok & prim).astype(int).groupby(grp).cummax().astype(bool)
    return on & mantem


def mede(ON, exige_mancal=False):
    v = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    if exige_mancal:
        v = v & (ON["sp"] | ON["vb"])
    m = AV.avalia(pos(v), alvo, mask)
    perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
    return m, perd


if __name__ == "__main__":
    P = {(kb, kv): partes(kb, kv) for kb in KB for kv in KV}

    b, _ = mede(P[(1.7, 2.2)])
    print(f"controle A: sem gate, k=1,7/2,2 -> {b['det']}/8, {b['fp_mes']:.2f} FP/mes, "
          f"{b['h_fp_mes']:.1f} h/mes, lead {b['lead_med']:.1f} h  "
          f"(esperado 8/8, 1,12, 39,0, 29,0)", flush=True)

    # controle B: com o gate desligado (ND todo verdadeiro) a trava tem que devolver
    # exatamente ON = mantem, isto e, a linha de base bit a bit. Se este controle
    # falhar, a trava esta mudando o detector por si mesma e nada abaixo vale.
    livre = pd.Series(True, index=idx)
    for modo in ("trava", "porta"):
        ON0 = {c: trava(P[(1.7, 2.2)][c], livre, modo) for c in SIN}
        assert all((ON0[c] == P[(1.7, 2.2)][c]).all() for c in SIN), f"{modo} nao e neutro"
    print("controle B: trava e porta com gate desligado == linha de base  OK\n", flush=True)

    lin = []
    for w in JANELAS:
        for dl in DELTAS:
            ND = decaimento(w, dl)
            cob = float((~ND["t"] & mask).sum() / max(mask.sum(), 1))
            for (kb, kv), pr in P.items():
              for modo in ("trava", "porta"):
                ON = {c: trava(pr[c], ND[c], modo) for c in SIN}
                for mg in (False, True):
                    m, perd = mede(ON, exige_mancal=mg)
                    lin.append(dict(W=w, delta=dl, modo=modo, kb=kb, kv=kv, mancal=mg,
                                    det=m["det"], eps=m["episodios"],
                                    fp_mes=round(m["fp_mes"], 3),
                                    h_fp_mes=round(m["h_fp_mes"], 1),
                                    lead=round(m["lead_med"], 2) if m["det"] else np.nan,
                                    perdidos=",".join(perd)))
            print(f"  W={w:>6s} delta={dl:.2f}  ({cob*100:4.1f}% do tempo de operacao "
                  f"em decaimento de t)", flush=True)

    d = pd.DataFrame(lin)
    d.to_csv("nao_decaimento.csv", index=False)

    print("\n" + "=" * 104)
    print(f"POR (W, delta): melhor deteccao com FP <= {ORC_FP}/mes, e a custo igualado "
          f"({H_FP_BASE:.0f} h/mes)")
    print("=" * 104)
    print(f"{'W':>7s} {'delta':>6s} {'modo':>6s} {'manc':>5s} | {'melhor det':>16s} {'fp':>6s} "
          f"{'h/mes':>7s} {'lead':>6s} | {'det a 39h':>12s} {'fp':>6s}  perdidos")
    for w in JANELAS:
        for dl in DELTAS:
            for modo in ("trava", "porta"):
              for mg in (False, True):
                s = d[(d.W == w) & (d.delta == dl) & (d.modo == modo) & (d.mancal == mg)]
                a = s[s.fp_mes <= ORC_FP]
                a = (a.sort_values(["det", "h_fp_mes"], ascending=[False, True]).iloc[0]
                     if len(a) else None)
                c = s.iloc[(s.h_fp_mes - H_FP_BASE).abs().argmin()]
                ta = f"{a.det}/8 (k {a.kb}/{a.kv})" if a is not None else " -- "
                fa = f"{a.fp_mes:.2f}" if a is not None else " -- "
                ha = f"{a.h_fp_mes:.1f}" if a is not None else " -- "
                la = f"{a.lead:.1f}" if a is not None and a.det else " -- "
                print(f"{w:>7s} {dl:6.2f} {modo:>6s} {str(mg):>5s} | {ta:>16s} {fa:>6s} {ha:>7s} "
                      f"{la:>6s} | {c.det:>3d}/8 {c.fp_mes:>6.2f}  {a.perdidos if a is not None else ''}")
    print("\n-> nao_decaimento.csv")
