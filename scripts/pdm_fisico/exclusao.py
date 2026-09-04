#!/usr/bin/env python3
"""Janela de exclusao da referencia: `excl_dias` (usa rotulo) x `guarda_h` (causal).

A ideia. `rolante.z_rolante` apaga [falha - excl_dias, falha + 2d] da referencia para que
a degradacao conhecida nao vire o proprio normal. Se a degradacao comeca antes dos 7 dias
atuais, ela contamina a referencia e o detector fica cego justamente para o caso lento --
o mesmo mecanismo de passa-alta descrito no docstring do rolante.py. Aumentar o intervalo
e a correcao obvia.

O problema. `excl_dias` usa o ROTULO. No instante t antes da falha f, a janela [f-7d, t]
ja esta no passado; apaga-la exige saber que f vem. Ja medimos que a referencia rolante
vaza no LOEO (9/9 -> 8/9). Aumentar excl_dias aumenta o vazamento, e o ganho aparece
retrospectivamente sem existir em producao.

A alternativa causal e a BANDA DE GUARDA (`guarda_h`, hoje 24 h): apaga as ultimas N horas
da referencia sem consultar rotulo nenhum. Ataca o mesmo mecanismo -- impedir que a
degradacao recente seja absorvida -- e e reproduzivel em producao.

Este script mede os dois lados:
  A) sweep de excl_dias (7..30) -> quanto o desempenho retrospectivo infla;
  B) sweep de guarda_h (24..240) -> o que sobra quando so se usa o que e causal;
  C) LOEO HONESTO: para o evento retirado, a referencia e recalculada SEM ele na lista de
     exclusao. E a medida direta do vazamento -- a diferenca entre (A) e (C) e o quanto do
     ganho de excl_dias e rotulo, nao sinal.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
import reduz_fp as RF

EXCLS = [7.0, 14.0, 21.0, 30.0]
GUARDAS = [24.0, 48.0, 72.0, 120.0, 240.0]


def vib(df, stable, falhas, excl, guarda):
    V = df[C.VIBRATION_TAGS].where(stable)
    return RO.z_rolante(V, stable, falhas, horas_base=400, guarda_h=guarda,
                        phi=0.0, excl_dias=excl).max(axis=1)


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df); stable = df["stable"].astype(bool)
    out = roda(BRACO, df, falhas)          # t, p, sp nao dependem destes parametros

    def mede(vb_s, rot, **kw):
        o = out.copy(); o["vb"] = vb_s
        al = alerta_2k(o, mask, K_BASE, K_VIB)
        alr = RF.refratario(al, 24)        # o ponto recomendado, para ver se o ganho soma
        x = A.avalia(al, falhas, mask); x.update(A.permuta(al, mask, x["det"], len(falhas)))
        xr = A.avalia(alr, falhas, mask)
        xt = A.avalia(trunca(alr, 12), falhas, mask)
        print(f"{rot:>26} {x['det']:4d}/9 {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f} {x['p']:8.4f} | "
              f"{xr['det']:3d}/9 {xr['h_fp_mes']:6.1f} | {xt['det']:3d}/9 {xt['h_fp_mes']:6.1f}",
              flush=True)
        return dict(rot=rot, det=x["det"], eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                    lead=x["lead_med"], p=x["p"], r24_det=xr["det"], r24_h=xr["h_fp_mes"],
                    r24t12_det=xt["det"], r24t12_h=xt["h_fp_mes"],
                    quais=",".join(x["detectados"]), **kw)

    cab = (f"{'config':>26} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} "
           f"{'p':>8} | {'+refr24':>10} | {'+r24+teto':>11}")
    L = []
    print("A) sweep de excl_dias (usa rotulo) -- guarda fixa em 24 h\n" + cab, flush=True)
    for e in EXCLS:
        L.append(mede(vib(df, stable, falhas, e, 24.0), f"excl={e:.0f}d guarda=24h",
                      excl=e, guarda=24.0))
    print("\nB) sweep de guarda_h (causal) -- excl fixa em 7 d\n" + cab, flush=True)
    for g in GUARDAS:
        L.append(mede(vib(df, stable, falhas, 7.0, g), f"excl=7d guarda={g:.0f}h",
                      excl=7.0, guarda=g))
    pd.DataFrame(L).to_csv("exclusao.csv", index=False)

    print("\n" + "=" * 96)
    print("C) LOEO HONESTO -- para o evento retirado, a referencia NAO o exclui")
    print("=" * 96)
    print(f"{'config':>26} {'retrospectivo':>14} {'LOEO honesto':>13}  eventos perdidos no LOEO")
    for e in EXCLS:
        det_loeo, perdidos = 0, []
        for t in falhas:
            resto = falhas[falhas != t]
            vb_s = vib(df, stable, resto, e, 24.0)     # referencia cega ao evento testado
            o = out.copy(); o["vb"] = vb_s
            al = alerta_2k(o, mask, K_BASE, K_VIB)
            ok = bool(al.loc[t - pd.Timedelta(hours=48):t].fillna(False).any())
            det_loeo += ok
            if not ok: perdidos.append(f"{t:%d/%m}")
        r = [x for x in L if x["excl"] == e and x["guarda"] == 24.0][0]
        print(f"{'excl='+str(int(e))+'d':>26} {str(r['det'])+'/9':>14} {str(det_loeo)+'/9':>13}"
              f"  {','.join(perdidos)}", flush=True)


if __name__ == "__main__":
    main()
