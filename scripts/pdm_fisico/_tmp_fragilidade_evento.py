"""Fragilidade por evento: de todas as configuracoes (kb,kv) dentro do orcamento
de FP, que fracao detecta CADA evento? E o numero honesto para a reuniao --
distingue "o detector ve o evento" de "o ponto de operacao foi calibrado nele".

Comparacao direta com o Francisco, que declara 04/11/2025 e 09/12/2025
inalcancaveis por qualquer das 51.840 configuracoes dele."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import avalia as AV

KBS = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4, 3.0]
KVS = [1.8, 2.2, 2.8, 3.5]
TETO = 1.0   # o mesmo teto do Francisco: 1 FP/mes
paradas = paradas_reais_2h()
JAN = pd.Timedelta(hours=48)
jw = [(t - JAN, t) for t in alvo]

linhas = []
for kb in KBS:
    for kv in KVS:
        ON = partes(kb, kv)
        ns = sum(ON[c].astype(int) for c in SIN)
        v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
        al = pos(v, ns, REFRAT_H, DUR_MIN, False)
        eps = AV.episodios(al)
        if not eps: continue
        m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
        cl = classifica_regra_c(eps, paradas)
        n_fp = sum(1 for a,b,c,l in cl if c == "FP")
        h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
        pega = {t: any(a <= t1 and b >= t0 for a, b in eps)
                for t, (t0, t1) in zip(alvo, jw)}
        linhas.append(dict(kb=kb, kv=kv, det=sum(pega.values()),
                           fp_mes=n_fp/meses, h_mes=h_fp/meses,
                           **{t.strftime("%d/%m/%Y"): pega[t] for t in alvo}))
T = pd.DataFrame(linhas)
cols_ev = [t.strftime("%d/%m/%Y") for t in alvo]
dentro = T[T.fp_mes <= TETO]
print(f"configuracoes varridas: {len(T)}   dentro do teto de {TETO} FP/mes: {len(dentro)}\n")
print("FRAGILIDADE POR EVENTO (fracao das configs no teto que detectam cada um)")
print("=" * 74)
print(f"{'evento':>12} {'no teto':>10} {'todas':>10}   avaliacao")
for c in cols_ev:
    f_teto = 100*dentro[c].mean() if len(dentro) else float('nan')
    f_all = 100*T[c].mean()
    tag = "ROBUSTO" if f_teto >= 50 else ("fragil" if f_teto > 0 else "NUNCA")
    marca = "  <<< Francisco diz inalcancavel" if c in ("04/11/2025","09/12/2025") else ""
    print(f"{c:>12} {f_teto:9.0f}% {f_all:9.0f}%   {tag}{marca}")
print(f"\nmelhor ponto dentro do teto (max det, depois min h/mes):")
b = dentro.sort_values(["det","h_mes"], ascending=[False,True]).head(3)
print(b[["kb","kv","det","fp_mes","h_mes"]].to_string(index=False))
