#!/usr/bin/env python3
"""Ordem dos sinais medida ANTES do episodio -- correcao de um erro de desenho meu.

O teste anterior (ordem.py) mediu o intervalo entre o 1o e o 2o sinal DENTRO do episodio
e achou 0,00 h em todos os 21. Nao e resultado sobre a maquina: a regra e `voto >= 2`,
entao o episodio COMECA no instante em que o segundo sinal cruza. Eu media o intervalo
numa janela que a propria regra recorta no ponto de coincidencia.

A medida correta olha o PRE-EPISODIO: para cada sinal, quando comecou a corrida contigua
de ativacao que desagua no episodio. Ai a ordem e os intervalos sao reais.

O diagnostico bruto insinuava algo: nas deteccoes o primeiro a disparar e `p` ou `t`
metade a metade; nos falsos positivos o `p` lidera em 8 de 13. Se `p` liderando for
marcador de transiente, e usavel como condicao.

Mede, por episodio: quem liderou, ha quanto tempo cada sinal estava ativo quando o
episodio confirmou (fase solo), e o intervalo entre o 1o e o 2o.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
import reduz_fp as RF, cusum_cru as CC

T0 = CC.T0; SIN = CC.SIN


def inicio_corrida(s, t):
    """Inicio da corrida contigua de True em `s` que cobre `t` (ou None)."""
    if t not in s.index or not bool(s.loc[t]):
        return None
    v = s.loc[:t]
    falso = v[~v.fillna(False)]
    return v.index[0] if not len(falso) else s.index[s.index.get_loc(falso.index[-1]) + 1]


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    op = df["in_operation"].astype(bool); part = op & ~op.shift(fill_value=False)
    reset = (~mask) | part
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in CC.HL.items()}

    def cus(z, h, carry=0.25):
        x = (z - 0.75).fillna(0.0).to_numpy(); r = reset.to_numpy()
        S = np.empty(len(x)); acc = 0.0
        for i in range(len(x)):
            acc = acc*carry if r[i] else max(0.0, acc + x[i]); S[i] = acc
        return S > h
    ON = {c: (DET._sustained(E[c], CC.BASE[c]*CC.K[c]) |
              pd.Series(cus((E[c]/(CC.BASE[c]*CC.K[c])).clip(upper=20), 80), index=idx)) & mask
          for c in SIN}
    ref = RF.dur_min(RF.refratario(
        pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask, 48), 120)
    eps = A.episodios(ref & sel)
    jw = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    print(f"episodios: {len(eps)}\n", flush=True)

    rows = []
    for a, b in eps:
        tp = "DET" if any(a <= t1 and b >= t0 for t0, t1 in jw) else "FP"
        ini = {}
        for c in SIN:
            i0 = inicio_corrida(ON[c], a)
            if i0 is not None:
                ini[c] = (a - i0).total_seconds()/3600      # ha quanto tempo estava ativo
        if len(ini) < 2:
            continue
        o = sorted(ini.items(), key=lambda kv: -kv[1])       # o que esta ativo ha mais tempo lidera
        rows.append(dict(ini=a, tipo=tp, lider=o[0][0], solo=o[0][1]-o[1][1],
                         ativo_lider=o[0][1], segundo=o[1][0], n=len(ini)))
    D = pd.DataFrame(rows); D.to_csv("pre_episodio.csv", index=False)

    print("=" * 92); print("1. QUEM LIDERA, e ha quanto tempo estava ativo"); print("=" * 92)
    for tp in ["DET", "FP"]:
        g = D[D.tipo == tp]
        print(f"\n  {tp} (n={len(g)}):  lider = {g.lider.value_counts().to_dict()}")
        print(f"     fase SOLO do lider (h antes do 2o confirmar): "
              f"mediana {g.solo.median():.2f}  p25 {g.solo.quantile(.25):.2f}  "
              f"p75 {g.solo.quantile(.75):.2f}  max {g.solo.max():.1f}")
        print(f"     tempo ativo do lider no instante da confirmacao: "
              f"mediana {g.ativo_lider.median():.2f} h  max {g.ativo_lider.max():.1f} h")

    print("\n" + "=" * 92); print("2. a fase solo separa?"); print("=" * 92)
    print(f"{'exigir solo >=':>16} {'DET':>10} {'FP':>10}")
    nd0 = int((D.tipo == "DET").sum()); nf0 = int((D.tipo == "FP").sum())
    for s in [0, 0.5, 1, 2, 4, 8, 16, 24]:
        nd = int(((D.tipo == "DET") & (D.solo >= s)).sum())
        nf = int(((D.tipo == "FP") & (D.solo >= s)).sum())
        print(f"{'>= '+str(s)+' h':>16} {str(nd)+'/'+str(nd0):>10} {str(nf)+'/'+str(nf0):>10}")

    print("\n" + "=" * 92); print("3. o lider ser `p` marca transiente?"); print("=" * 92)
    for lid in SIN:
        nd = int(((D.tipo == "DET") & (D.lider == lid)).sum())
        nf = int(((D.tipo == "FP") & (D.lider == lid)).sum())
        raz = (nd/max(nd0,1)) / max(nf/max(nf0,1), 1e-9)
        print(f"  lider={lid:>3}: {nd}/{nd0} das deteccoes, {nf}/{nf0} dos falsos positivos"
              f"   razao {raz:.2f}")

    print("\n" + "=" * 92); print("4. regras derivadas"); print("=" * 92)
    def aplica(cond):
        novo = pd.Series(False, index=idx)
        for _, r in D.iterrows():
            if cond(r):
                a = r["ini"]
                fim = [b for x, b in eps if x == a]
                if fim: novo.loc[a:fim[0]] = True
        return novo
    print(f"{'regra':>34} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}  perde")
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    x0 = A.avalia(ref[sel], alvo, m2)
    print(f"{'referencia':>34} {x0['det']:3d}/8 {x0['episodios']:5d} {x0['fp_mes']:7.2f} "
          f"{x0['h_fp_mes']:7.1f} {x0['lead_med']:6.1f}  —")
    for rot, cond in [("solo >= 1 h", lambda r: r["solo"] >= 1),
                      ("solo >= 2 h", lambda r: r["solo"] >= 2),
                      ("solo >= 4 h", lambda r: r["solo"] >= 4),
                      ("lider != p", lambda r: r["lider"] != "p"),
                      ("lider != p OU solo >= 2 h", lambda r: r["lider"] != "p" or r["solo"] >= 2),
                      ("lider = vb ou sp", lambda r: r["lider"] in ("vb", "sp"))]:
        al = aplica(cond)
        x = A.avalia(al[sel], alvo, m2)
        perd = sorted(set(alvo_s) - set(x["detectados"]))
        marca = "  <<<" if x["det"] == 8 and x["fp_mes"] < x0["fp_mes"] else ""
        print(f"{rot:>34} {x['det']:3d}/8 {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f}  "
              f"{', '.join(p[5:] for p in perd) if perd else '—'}{marca}")


if __name__ == "__main__":
    main()
