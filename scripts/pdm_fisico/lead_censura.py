#!/usr/bin/env python3
"""O lead de 29,0 h e CENSURADO pela propria regua. Quanto ele vale de verdade.

O PROBLEMA. A regua conta deteccao se houver alarme em [evento - 48 h, evento), e mede
o lead a partir do PRIMEIRO alarme DENTRO dessa janela. Se o alarme ja estava ligado
quando a janela abriu, o lead medido e exatamente 48,0 h -- nao porque foi esse o aviso,
mas porque a regua nao olha para tras disso. Quatro dos oito eventos batem em 48,0 h
cravado. O "lead medio de 29,0 h" e, portanto, um LIMITE INFERIOR, e o numero que a
operacao usa para decidir se da tempo de agir esta subestimado por construcao.

Isto nao e um ajuste do detector: o detector nao muda uma linha. E medir direito o que
ja existe. Depois de tres ataques ao custo terem batido no mesmo piso (1,033 FP/mes em
720 pontos de pos-processamento, geometria refutada em nao_decaimento.py, percentil
refutado em autocalibra.py), o lead e o unico eixo da entrega que ainda estava mal medido.

O QUE ESTE SCRIPT MEDE.
  1. Descensura por episodio: para cada evento, acha o episodio de alarme que a regua
     creditou e reporta o INICIO dele. E a resposta operacional -- "a que horas acendeu
     a luz que pegou esta falha?" -- e nao depende de nenhuma janela.
  2. Varredura da janela da regua (24 h a 168 h) para o nosso detector: como deteccao,
     lead e custo se movem quando a janela abre. Custo cai ao alargar (episodios que
     eram falso positivo passam a cair dentro de janela), entao os dois numeros tem que
     ser lidos juntos -- alargar a janela nao e de graca, e um afrouxamento da regua.
  3. A mesma varredura na serie do detector PCA do Francisco/Lara, quando disponivel,
     para que a comparacao continue simetrica: mudar a regua para um lado so seria
     trocar de regua no meio do jogo.

CONTROLE. Em janela = 48 h o resultado tem que ser o publicado (8/8, 1,03 FP/mes com a
porta de mancal, lead 29,0 h).
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import (GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM,
                             REFRAT_H, DUR_MIN, T0)
from blackout_curto import cusum
from pos_processamento import partes, pos, mask, idx, alvo, EW

JANELAS_H = [24, 48, 72, 96, 120, 168]
KB, KV = 1.7, 2.2
SERIE_ELES = "series_v6_para_plots.csv.gz"


def alerta_ponto_de_operacao():
    """O alarme final no ponto de operacao adotado: voto >=2 com porta de mancal."""
    ON = partes(KB, KV)
    n_sin = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(n_sin >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, n_sin, REFRAT_H, DUR_MIN, False)


def descensura(al, eventos, janela_h=48.0):
    """Para cada evento creditado em `janela_h`, o inicio do episodio que o creditou."""
    eps = AV.episodios(al)
    out = []
    for t in eventos:
        t0 = t - pd.Timedelta(hours=janela_h)
        # o episodio creditado e o que tem alguma amostra em [t0, t)
        cand = [(a, b) for a, b in eps if a < t and b >= t0]
        if not cand:
            out.append(dict(evento=t, creditado=False)); continue
        a, b = cand[-1]
        out.append(dict(evento=t, creditado=True, ini_episodio=a,
                        lead_regua=min((t - max(a, t0)).total_seconds() / 3600, janela_h),
                        lead_real=(t - a).total_seconds() / 3600,
                        censurado=a <= t0,
                        dur_episodio=(b - a).total_seconds() / 3600))
    return pd.DataFrame(out)


def varre(al, eventos, quente):
    lin = []
    for jh in JANELAS_H:
        m = AV.avalia(al, eventos, quente, janela_h=float(jh))
        d = descensura(al, eventos, float(jh))
        cens = int(d.get("censurado", pd.Series(dtype=bool)).fillna(False).sum())
        lin.append(dict(janela_h=jh, det=m["det"], fp_mes=round(m["fp_mes"], 3),
                        h_fp_mes=round(m["h_fp_mes"], 1),
                        lead_med=round(m["lead_med"], 2), lead_min=round(m["lead_min"], 2),
                        censurados=cens))
    return pd.DataFrame(lin)


if __name__ == "__main__":
    al = alerta_ponto_de_operacao()
    b = AV.avalia(al, alvo, mask)
    print(f"controle: janela 48 h -> {b['det']}/8, {b['fp_mes']:.2f} FP/mes, "
          f"{b['h_fp_mes']:.1f} h/mes, lead {b['lead_med']:.1f} h  "
          f"(esperado 8/8, 1,03, 38,7, 29,0)\n", flush=True)

    print("=" * 96)
    print("1. DESCENSURA POR EPISODIO -- a que horas acendeu a luz que pegou cada falha")
    print("=" * 96)
    d = descensura(al, alvo, 48.0)
    d["evento"] = d.evento.dt.strftime("%d/%m/%Y %H:%M")
    d["ini_episodio"] = pd.to_datetime(d.ini_episodio).dt.strftime("%d/%m/%Y %H:%M")
    print(f"{'evento':>17} {'inicio do alarme':>17} {'lead da regua':>14} "
          f"{'lead real':>10} {'censurado':>10}")
    for _, r in d.iterrows():
        print(f"{r.evento:>17} {r.ini_episodio:>17} {r.lead_regua:13.1f}h "
              f"{r.lead_real:9.1f}h {'SIM' if r.censurado else '-':>10}")
    print(f"\n  lead medio pela regua : {d.lead_regua.mean():6.1f} h   "
          f"(mediana {d.lead_regua.median():.1f} h, minimo {d.lead_regua.min():.1f} h)")
    print(f"  lead medio descensurado: {d.lead_real.mean():6.1f} h   "
          f"(mediana {d.lead_real.median():.1f} h, minimo {d.lead_real.min():.1f} h)")
    print(f"  eventos censurados em 48 h: {int(d.censurado.sum())}/8")
    d.to_csv("lead_censura_eventos.csv", index=False)

    print("\n" + "=" * 96)
    print("2. VARREDURA DA JANELA DA REGUA -- nosso detector (alargar a regua NAO e de graca)")
    print("=" * 96)
    v = varre(al, alvo, mask); v["quem"] = "nosso"
    print(v[["janela_h", "det", "fp_mes", "h_fp_mes", "lead_med", "lead_min",
             "censurados"]].to_string(index=False))

    import os
    if os.path.exists(SERIE_ELES):
        print("\n" + "=" * 96)
        print("3. A MESMA VARREDURA NA SERIE DO DETECTOR PCA (simetria da regua)")
        print("=" * 96)
        s = pd.read_csv(SERIE_ELES, parse_dates=["timestamp"]).set_index("timestamp")
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        ale = s["alerta"].resample("2min").max().fillna(False).astype(bool).reindex(idx).fillna(False)
        ve = varre(ale, alvo, mask); ve["quem"] = "pca"
        print(ve[["janela_h", "det", "fp_mes", "h_fp_mes", "lead_med", "lead_min",
                  "censurados"]].to_string(index=False))
        v = pd.concat([v, ve])
    else:
        print(f"\n(3) serie do detector PCA ausente ({SERIE_ELES}); varredura simetrica "
              f"nao feita -- baixar de novo a task a669fffabef9442197c054d974d43ad4 para refazer")

    v.to_csv("lead_censura.csv", index=False)
    print("\n-> lead_censura_eventos.csv, lead_censura.csv")
