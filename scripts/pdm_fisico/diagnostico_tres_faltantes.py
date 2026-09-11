#!/usr/bin/env python3
"""O que os tres eventos que falham na regua de inicio fazem nas 48 h finais?

Sob a regua de inicio (adotada como correta em 04/09/2026) ficamos em 5/8 com o
religamento. Falham 27/02/2025 (ep 144 h), 17/03/2025 (ep 195 h) e 29/04/2025
(ep de 17 h que terminou 34,7 h antes do trip).

Medir antes de propor. Tres desfechos possiveis, cada um aponta para uma tecnica
diferente:
  A. ha ACELERACAO na janela final (nivel ja alto, mas subindo mais rapido)
     -> canal de VARIACAO (derivada / CUSUM dos incrementos) criaria inicio ali
  B. ha excursao pequena e SUB-LIMIAR na janela final
     -> ESCORVA: apos um precursor, baixar o limiar por tempo limitado
  C. nao ha nada na janela final
     -> nenhuma das duas resolve; o evento e inalcancavel por essa via
"""
import numpy as np, pandas as pd
from pos_processamento import EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from plota_estilo_francisco import KB, KV

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
RAZ = {c: EW[c].where(mask) / (BASE[c] * K[c]) for c in SIN}
FALHAM = ["2025-02-27", "2025-03-17", "2025-04-29"]
PASSAM = ["2025-04-07", "2025-11-04", "2025-12-09", "2026-02-26"]

def perfil(t, rot):
    print(f"\n{rot}  {t:%d/%m/%Y %H:%M}")
    print("-" * 88)
    print(f"{'canal':>6} {'[-96,-48]h':>12} {'[-48,-24]h':>12} {'[-24,0]h':>11} "
          f"{'pico final':>11} {'aceleracao':>12}")
    for c in SIN:
        a = float(RAZ[c].loc[t-pd.Timedelta("96h"):t-pd.Timedelta("48h")].mean())
        b = float(RAZ[c].loc[t-pd.Timedelta("48h"):t-pd.Timedelta("24h")].mean())
        d = float(RAZ[c].loc[t-pd.Timedelta("24h"):t].mean())
        pico = float(RAZ[c].loc[t-pd.Timedelta("48h"):t].max())
        # aceleracao = inclinacao nas 48 h finais menos a das 48 h anteriores
        def incl(t0, t1):
            s = RAZ[c].loc[t0:t1].dropna()
            if len(s) < 20: return np.nan
            x = (s.index - s.index[0]).total_seconds().to_numpy()/3600
            return float(np.polyfit(x, s.to_numpy(), 1)[0]) * 24
        ac = incl(t-pd.Timedelta("48h"), t) - incl(t-pd.Timedelta("96h"), t-pd.Timedelta("48h"))
        flag = ""
        if pico >= 1.0: flag = "  ACESO"
        elif pico >= 0.7: flag = "  sub-limiar (>=0,7)"
        print(f"{c:>6} {a:12.2f} {b:12.2f} {d:11.2f} {pico:11.2f} {ac:+11.2f}/d{flag}")

print("OS TRES QUE FALHAM NA REGUA DE INICIO")
print("=" * 88)
for s in FALHAM:
    perfil(pd.Timestamp(s, tz="UTC") + (alvo[alvo.dt.strftime('%Y-%m-%d') == s].iloc[0]
                                        - pd.Timestamp(s, tz="UTC")), "FALHA")
print("\n\nOS QUE PASSAM, PARA CONTRASTE")
print("=" * 88)
for s in PASSAM[:2]:
    perfil(alvo[alvo.dt.strftime('%Y-%m-%d') == s].iloc[0], "passa")
