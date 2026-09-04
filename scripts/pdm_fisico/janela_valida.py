#!/usr/bin/env python3
"""Reavalia tudo na JANELA EM QUE O DETECTOR EXISTE. Alvo passa a ter 8 eventos.

Decisao. 2024-01-16 sai do alvo. modos.py mostrou que ali o detector estava desligado por
falta de historico, nao errado: `t`, `p` e `sp` tem ZERO pontos validos na janela de 48 h
(o fit walk-forward de janeiro/2024 usa `stable & idx < 2024-01-01`, que e vazio) e havia
159 h de operacao quente contra as 400 h + 24 h de guarda que a referencia rolante exige.
Com um sinal so, o voto >=2 e insatisfazivel por construcao.

Mas remover so o evento do numerador seria contabilidade seletiva. O correto e recortar a
JANELA DE AVALIACAO: o detector nao pode ser creditado nem culpado por um periodo em que
nao produz escore. Isso corrige tres coisas de uma vez -- o denominador de eventos, o
denominador de FP (episodios naquele periodo) e os meses de operacao usados na taxa.

DUAS LISTAS DIFERENTES, e a distincao importa:
  - ALVO de avaliacao: 8 eventos (2024-01-16 fora).
  - EXCLUSAO da referencia rolante: os 9, inclusive 2024-01-16. Aquilo foi uma falha real
    e a degradacao em torno dela nao deve entrar no baseline. O papel das duas listas e
    distinto: uma diz "o que o detector deveria prever", a outra diz "que dado esta
    contaminado". Confundi-las seria vazamento numa direcao ou cegueira na outra.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
import reduz_fp as RF, ablacao_sp as AS

R_REFRAT, D_MIN = 48, 60


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    out = roda(BRACO, df, todas)          # referencia exclui as 9 -- de proposito

    # --- quando o detector passa a existir: os 4 sinais validos ao mesmo tempo
    val = out[["t", "p", "sp", "vb"]].notna().all(axis=1) & mask
    t0 = val[val].index[0]
    print("=" * 96)
    print("QUANDO O DETECTOR PASSA A EXISTIR")
    print("=" * 96)
    for c in ["t", "p", "sp", "vb"]:
        pv = out[c].notna()
        print(f"  {c:>3}: primeiro valor valido em {pv[pv].index[0]:%Y-%m-%d %H:%M}")
    print(f"\n  os quatro simultaneamente, dentro da mascara: {t0:%Y-%m-%d %H:%M}")
    print(f"  a serie comeca em {idx[0]:%Y-%m-%d %H:%M} -> {(t0-idx[0]).days} dias de aquecimento")

    alvo = todas[todas >= t0].reset_index(drop=True)
    fora = todas[todas < t0]
    print(f"\n  eventos no alvo: {len(alvo)}   fora da janela: {len(fora)} "
          f"({', '.join(f'{t:%d/%m/%Y}' for t in fora)})")

    dentro = pd.Series(idx >= t0, index=idx)
    m2 = mask & dentro
    print(f"  horas pontuaveis: {mask.sum()*2/60:.0f} h na serie toda -> "
          f"{m2.sum()*2/60:.0f} h na janela valida ({m2.sum()*2/60/730:.1f} meses de operacao)")

    al_full = alerta_2k(out, mask, K_BASE, K_VIB)
    pontos = [("base (k=1,7)", al_full),
              ("+ refrat 48 h + dur 60 min", RF.dur_min(RF.refratario(al_full, R_REFRAT), D_MIN)),
              ("+ refrat 48 h + dur 60 + teto 12 h",
               trunca(RF.dur_min(RF.refratario(al_full, R_REFRAT), D_MIN), 12))]

    camps = [(a, b, h) for a, b, h in AS.campanhas(df, mask, idx) if a >= t0]
    jw = [(t - pd.Timedelta(hours=48), t) for t in alvo]

    def duty_camp(al):
        y = []
        for a, b, h in camps:
            sel = (idx >= a) & (idx <= b)
            eps = A.episodios(al & sel)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0_ for t0_, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    print("\n" + "=" * 96)
    print(f"REAVALIACAO NA JANELA VALIDA -- alvo de {len(alvo)} eventos, {len(camps)} campanhas")
    print("=" * 96)
    print(f"{'configuracao':>36} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'duty':>7} "
          f"{'lead':>6} {'p':>8} | {'rho':>7} {'p_rho':>7}")
    for rot, al in pontos:
        a2 = al & dentro
        x = A.avalia(a2[dentro.values], alvo, m2[dentro.values])
        x.update(A.permuta(a2[dentro.values], m2[dentro.values], x["det"], len(alvo)))
        y = duty_camp(a2); r = stats.spearmanr(np.arange(len(y)), y)
        print(f"{rot:>36} {x['det']:3d}/{len(alvo)} {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {100*x['h_fp_mes']/730:6.2f}% {x['lead_med']:6.1f} "
              f"{x['p']:8.4f} | {r.statistic:+7.3f} {r.pvalue:7.4f}", flush=True)

    print("\n  intervalo de confianca de Wilson:")
    for k, n in [(8, 9), (8, 8)]:
        z = 1.959964; p = k/n
        d = 1 + z*z/n
        c = (p + z*z/(2*n))/d
        hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
        print(f"    {k}/{n} = {100*p:5.1f}%  ->  [{100*max(0,c-hw):5.1f}%, {100*min(1,c+hw):5.1f}%]"
              f"   largura {100*(min(1,c+hw)-max(0,c-hw)):.1f} pp")

    print("\n" + "=" * 96); print("LOEO na janela valida (8 dobras)"); print("=" * 96)
    fam = {}
    for kb in [1.2, 1.4, 1.7, 2.0, 2.4, 2.8]:
        a = alerta_2k(out, mask, kb, K_VIB)
        for Rh in [0, 24, 48, 72]:
            fam[(kb, Rh)] = RF.dur_min(RF.refratario(a, Rh), D_MIN) & dentro
    for teto_fp in [2.5, 3.5]:
        ac = 0
        for t in alvo:
            resto = [x for x in alvo if x != t]; m = None
            for key, al in fam.items():
                x = A.avalia(al[dentro.values], resto, m2[dentro.values])
                if x["fp_mes"] <= teto_fp and (m is None or (x["det"], -x["fp_mes"]) > m[1]):
                    m = (key, (x["det"], -x["fp_mes"]))
            if m is None: continue
            ac += bool(fam[m[0]].loc[t-pd.Timedelta(hours=48):t].fillna(False).any())
        print(f"  orcamento <= {teto_fp} FP/mes: LOEO {ac}/{len(alvo)}", flush=True)


if __name__ == "__main__":
    main()
