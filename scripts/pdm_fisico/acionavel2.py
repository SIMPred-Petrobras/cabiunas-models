"""Antecedencia REAL, sem o teto de 48 h da regua do projeto.

Em vez de perguntar 'houve alerta na janela de 48 h', pergunta-se: qual o
episodio de alerta que chega ao trip, e quando ele comecou? Episodio = alertas
separados por menos de 2 h contam como um so (mesma regra do resto do projeto).
So conta se o episodio termina a menos de 2 h do trip -- senao e um episodio
anterior que ja tinha cessado, e nao 'avisou' daquele evento.

Reporta tambem a fracao de tempo em cada patamar nos 7 dias anteriores, que
diz se o aviso foi continuo ou intermitente.
"""
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src")
import numpy as np, pandas as pd
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO

K_BASE, K_VIB = 1.7, 2.2
df = canonico()
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
falhas = ftodas[ftodas >= "2025-01-01"].reset_index(drop=True)
mask = mascara_pontuacao(df); idx = mask.index
out = roda(BRACO, df, ftodas)
def ew(c, hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
S = {"t": DET._sustained(ew("t","1h"), DET.THR_FAM*K_BASE),
     "p": DET._sustained(ew("p","1h"), DET.THR_FAM*K_BASE),
     "sp": DET._sustained(ew("sp","30min"), DET.THR_SPREAD*K_BASE),
     "vb": DET._sustained(ew("vb","30min"), 3.0*K_VIB)}
n = sum(s.astype(int) for s in S.values())
niveis = {"atencao": (n>=1)&mask, "confirmado": (n>=2)&mask}

def antecedencia(al, ev, tol_h=2.0):
    eps = A.episodios(al)
    cand = [(a,b) for a,b in eps if a < ev and (ev-b).total_seconds()/3600 <= tol_h]
    if not cand: return np.nan, np.nan
    a, b = max(cand, key=lambda x: x[0])
    return (ev-a).total_seconds()/3600, (b-a).total_seconds()/3600+2/60

linhas = []
for ev in falhas:
    r = {"evento": ev.strftime("%Y-%m-%d %H:%M")}
    for nome, al in niveis.items():
        lead, dur = antecedencia(al, ev)
        r[f"{nome}_lead_h"] = lead
        w = al[(al.index>=ev-pd.Timedelta(days=7)) & (al.index<ev)]
        m7 = mask[(mask.index>=ev-pd.Timedelta(days=7)) & (mask.index<ev)]
        r[f"{nome}_duty7d"] = 100*w.sum()/max(m7.sum(),1)
    linhas.append(r)
R = pd.DataFrame(linhas); R.to_csv("acionavel2.csv", index=False)

print(f"{'evento':>17} | {'ATENCAO':>22} | {'CONFIRMADO':>22}")
print(f"{'':>17} | {'inicio antes':>13} {'%7d':>8} | {'inicio antes':>13} {'%7d':>8}")
print("-"*70)
for _, r in R.iterrows():
    def fmt(l, d):
        return (f"{'nao visto':>13} {'':>8}" if np.isnan(l)
                else f"{l:10.1f} h {d:7.0f}%")
    print(f"{r.evento:>17} | {fmt(r.atencao_lead_h, r.atencao_duty7d)} | "
          f"{fmt(r.confirmado_lead_h, r.confirmado_duty7d)}")
print()
for c in ["atencao_lead_h","confirmado_lead_h"]:
    v = R[c].dropna()
    print(f"{c:22s} mediana {v.median():6.1f} h | min {v.min():5.1f} | max {v.max():6.1f} | n={len(v)}/8")
