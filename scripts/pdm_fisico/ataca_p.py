#!/usr/bin/env python3
"""Ataca a fonte dos falsos positivos: o sinal `p` no transiente pos-partida.

Achado que motiva (autopsia_fp.py), sobre os 23 episodios do ponto atual:
    `p` aparece em 15 de 15 falsos positivos -- TODOS -- e em so 5 de 8 deteccoes;
    `t+vb` aparece em 3 deteccoes e em NENHUM falso positivo;
    11 de 15 FP estao a <=30 h de uma partida, mediana 6,5 h -- o instante exato em que
    o blackout de 6 h expira.

Leitura: o erro de reconstrucao PCA das 12 tags de pressao ainda nao assentou quando o
blackout acaba, arrasta o `vb` junto e os dois fecham o voto >=2. E o mesmo mecanismo que
ja tinha refutado o escape por magnitude (`p` a 22x do limiar logo apos partidas) -- eu
identifiquei e nao voltei para atacar.

Tres intervencoes, todas a custo igualado e sob o protocolo completo:

  A  BLACKOUT POR SINAL   o `p` so ganha direito de falar N horas apos a partida; os
     outros tres seguem com 6 h. Nao custa deteccao por construcao -- so adia um sinal.
  B  VOTO PONDERADO       `p` vale w < 1. Assim `p+vb` (6 FP, 2 deteccoes) nao fecha o
     voto, mas `p+t+vb` e `t+vb` fecham. Custa deteccao, e a grade diz quanto.
  C  BASELINE POR CAMPANHA  `p` normalizado pela propria mediana no inicio da campanha,
     em vez do ajuste mensal -- ataca a causa (referencia velha) e nao o sintoma.

Referencia: 8/8, 1,29 FP/mes, 42,6 h/mes, lead 29,0 h.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
import reduz_fp as RF, cusum_cru as CC

T0 = CC.T0
SIN = CC.SIN
PAS = pd.Timedelta("2min")
BL_P = [6, 12, 18, 24, 36, 48]          # horas de blackout so para o `p`
WS = [0.4, 0.5, 0.6, 0.8, 1.0]          # peso do `p` no voto
CAMP_H = [0, 6, 12, 24]                 # horas iniciais da campanha usadas como baseline do p


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    op = df["in_operation"].astype(bool)
    part = op & ~op.shift(fill_value=False)
    out = roda(BRACO, df, todas)
    gid = part.cumsum()
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos, {int(part[sel].sum())} partidas\n", flush=True)

    def blackout(h):
        n = int(pd.Timedelta(hours=h) / PAS)
        return part.rolling(n, min_periods=1).max().astype(bool)

    def canais(out_c, mask_c, c, kappa=0.75, hc=80):
        E = out_c.ewm(halflife=pd.Timedelta(CC.HL[c]), times=idx).mean().where(mask_c)
        thr = CC.BASE[c] * CC.K[c]
        ew = DET._sustained(E, thr)
        reset_c = (~mask_c) | part
        Z = (E / thr).clip(upper=20)
        cu = pd.Series(CC.cusum_bool(Z, kappa, hc, reset_c), index=idx)
        return (ew | cu) & mask_c

    base_on = {c: canais(out[c], mask, c) for c in SIN}
    ref = RF.dur_min(RF.refratario(
        pd.Series(sum(base_on[c].astype(int) for c in SIN) >= 2, index=idx) & mask, 48), 60)
    xr = A.avalia(ref[sel], alvo, m2)
    print(f"REFERENCIA: {xr['det']}/8  {xr['episodios']} eps  {xr['fp_mes']:.2f} FP/mes  "
          f"{xr['h_fp_mes']:.1f} h/mes  lead {xr['lead_med']:.1f} h\n", flush=True)

    L = []
    def mede(al, rot, **kw):
        y = A.avalia(al[sel], alvo, m2)
        L.append(dict(rot=rot, det=y["det"], eps=y["episodios"], fp=y["fp_mes"],
                      hm=y["h_fp_mes"], lead=y["lead_med"], quais=",".join(y["detectados"]), **kw))

    # --- A: blackout so para o `p`
    for h in BL_P:
        mp = mask & ~blackout(h)
        on = dict(base_on); on["p"] = canais(out["p"], mp, "p")
        al = RF.dur_min(RF.refratario(
            pd.Series(sum(on[c].astype(int) for c in SIN) >= 2, index=idx) & mask, 48), 60)
        mede(al, "A_blackout_p", par=f"{h}h")
    print("  A varrido", flush=True)

    # --- B: voto ponderado
    for w in WS:
        s = (base_on["p"].astype(float)*w + sum(base_on[c].astype(float) for c in SIN if c != "p"))
        al = RF.dur_min(RF.refratario(pd.Series(s >= 2.0, index=idx) & mask, 48), 60)
        mede(al, "B_voto_ponderado", par=f"w={w}")
    print("  B varrido", flush=True)

    # --- C: baseline do `p` por campanha
    p_raw = out["p"]
    for hh in CAMP_H:
        if hh == 0:
            mede(ref, "C_baseline_campanha", par="sem (referencia)"); continue
        n = int(pd.Timedelta(hours=hh) / PAS)
        ini = part.rolling(n, min_periods=1).max().astype(bool) & mask   # inicio de campanha
        med = p_raw.where(ini).groupby(gid).transform("median")
        pc = (p_raw - med.fillna(0.0)).clip(lower=0)
        on = dict(base_on); on["p"] = canais(pc, mask, "p")
        al = RF.dur_min(RF.refratario(
            pd.Series(sum(on[c].astype(int) for c in SIN) >= 2, index=idx) & mask, 48), 60)
        mede(al, "C_baseline_campanha", par=f"{hh}h iniciais")
    print("  C varrido", flush=True)

    # --- A+B combinados nos melhores
    for h in [12, 24]:
        for w in [0.5, 0.8]:
            mp = mask & ~blackout(h)
            on = dict(base_on); on["p"] = canais(out["p"], mp, "p")
            s = (on["p"].astype(float)*w + sum(on[c].astype(float) for c in SIN if c != "p"))
            al = RF.dur_min(RF.refratario(pd.Series(s >= 2.0, index=idx) & mask, 48), 60)
            mede(al, "A+B", par=f"bl={h}h w={w}")
    T = pd.DataFrame(L); T.to_csv("ataca_p.csv", index=False)

    print("\n" + "=" * 96)
    print("RESULTADO (referencia: 8/8, 1,29 FP/mes, 42,6 h/mes, lead 29,0 h)")
    print("=" * 96)
    print(f"{'intervencao':>20} {'parametro':>18} {'det':>6} {'eps':>5} {'FP/mes':>7} "
          f"{'h/mes':>7} {'lead':>6}  perde")
    bq = set(xr["detectados"])
    for _, r in T.iterrows():
        q = set(str(r.quais).split(","))
        perd = sorted(bq - q)
        marca = "  <<<" if (r.det == 8 and r.fp < xr["fp_mes"]) else ""
        print(f"{r.rot:>20} {r.par:>18} {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} "
              f"{r.hm:7.1f} {r.lead:6.1f}  {', '.join(x[5:] for x in perd) if perd else '—'}{marca}")


if __name__ == "__main__":
    main()
