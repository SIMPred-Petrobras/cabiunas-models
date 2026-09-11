#!/usr/bin/env python3
"""Fechar o episodio quando o sinal decai, E ABRIR UM NOVO quando ele sobe.

Corrige `decaimento_pico.py`, cujo corte marcava "morto" e nunca ressuscitava
enquanto o voto seguisse de pe -- o que silenciava uma subida real posterior
dentro da mesma corrida.

Motivacao (anatomia_travado.py): o episodio de 670 h antes de 26/02/2026 sao
DOIS eventos reais -- p a 35x o limiar em 29/01, e p a 2,75x em 24/02 -- colados
por tres semanas em que 86% do tempo nada passa de 1,5x. Quem segura o alarme no
meio e o CUSUM, que acumulou na primeira excursao. Fechar o vao e abrir episodio
novo na segunda subida transforma um alarme de 28 dias em dois alarmes curtos --
e o segundo NASCE dentro da janela de 48 h, contando pelas duas reguas.
"""
import numpy as np, pandas as pd


def corta_rearma(voto: np.ndarray, forca: np.ndarray, frac: float) -> np.ndarray:
    """Dentro de uma corrida de `voto`: acompanha o pico; quando a forca cai
    abaixo de frac x pico, SILENCIA e zera o pico; quando a forca volta a subir
    acima do nivel de silenciamento, RELIGA como episodio novo."""
    v = np.asarray(voto, dtype=bool)
    if frac <= 0 or not v.any():
        return v
    f = np.where(np.isfinite(forca), forca, 0.0)
    out = np.zeros(len(v), dtype=bool)
    pico = 0.0; calado = False; piso = 0.0
    for i in range(len(v)):
        if not v[i]:
            pico = 0.0; calado = False; piso = 0.0
            continue
        x = f[i]
        if calado:
            if x > piso:              # subiu de novo -> episodio NOVO
                calado = False; pico = x; out[i] = True
            continue
        pico = max(pico, x)
        if pico > 0 and x < frac * pico:
            calado = True; piso = x    # silencia; religa se voltar acima daqui
            continue
        out[i] = True
    return out


if __name__ == "__main__":
    import sys
    import avalia as AV
    from pos_processamento import partes, EW, pos, mask, idx, alvo
    from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
    from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c
    from regra_inicio_varredura import avalia_inicio

    K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
    FORCA = pd.concat([EW[c].where(mask) / (BASE[c] * K[c]) for c in SIN],
                      axis=1).max(axis=1).to_numpy()
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v0 = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    paradas = paradas_reais_2h(); meses = float(mask.sum()) * 2 / 60.0 / 730.0
    JAN = pd.Timedelta(hours=48)

    print("CORTE COM RELIGAMENTO -- fecha no decaimento, reabre na subida")
    print("=" * 104)
    print(f"{'frac':>6} | {'det':>5} {'FP':>4} {'FP/mes':>8} {'h/mes':>8} {'lead':>7} "
          f"| {'det_ini':>8} {'lead_ini':>9} | {'ep 26/02':>9} {'ep 17/03':>9}")
    print("-" * 104)
    for frac in [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
        v = pd.Series(corta_rearma(v0.to_numpy(), FORCA, frac), index=idx) if frac else v0
        al = pos(v, ns, REFRAT_H, DUR_MIN, False)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        hfc = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP") / meses
        def d_de(t):
            c = [(a, b) for a, b in eps if a <= t and b >= t - JAN]
            return f"{(c[0][1]-c[0][0]).total_seconds()/3600:.0f}h" if c else "--"
        print(f"{frac:6.2f} | {m['det']:4d}/8 {nfp:4d} {nfp/meses:8.3f} {hfc:8.1f} "
              f"{m['lead_med']:6.1f}h | {mi['det']:6d}/8 "
              f"{(f'{mi[chr(108)+chr(101)+chr(97)+chr(100)+chr(95)+chr(109)+chr(101)+chr(100)]:.1f}h' if mi['det'] else '--'):>9} "
              f"| {d_de(alvo.iloc[-1]):>9} {d_de(alvo.iloc[1]):>9}")
    print("-" * 104)
    print("  frac = 0 e o ponto de producao (8/8, 6 FP, 0,517, 7,1, 29,0 · det_ini 4/8)")
