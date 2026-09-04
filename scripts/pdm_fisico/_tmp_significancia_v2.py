"""v2: restringe alarmes E paradas a MESMA janela (a da grade), senao a taxa
de base fica subestimada e a significancia inflada."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from math import exp, factorial
from verdade import carrega_alarmes
from plota_estilo_francisco import paradas_reais_2h
from pos_processamento import idx

T_INI, T_FIM = idx.min(), idx.max()
alarmes = carrega_alarmes(0)
alarmes = alarmes[(alarmes.ts >= T_INI) & (alarmes.ts <= T_FIM)]
paradas = paradas_reais_2h()
n_par = len(paradas)
span_h = (T_FIM - T_INI).total_seconds() / 3600
JAN_H = 1.5

def p_ge(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return max(0.0, 1.0 - sum(exp(-lam) * lam**i / factorial(i) for i in range(k)))

print(f"janela da grade: {T_INI:%d/%m/%Y} .. {T_FIM:%d/%m/%Y}  ({span_h/24/30:.1f} meses)")
print(f"alarmes ACT nessa janela: {len(alarmes)}   paradas reais >=2h: {n_par}\n")
linhas = []
for tag, grp in alarmes.groupby("Tag Alarme"):
    ts = pd.DatetimeIndex(grp.ts)
    obs = sum(1 for q in paradas.ini
              if ((ts >= q - pd.Timedelta(hours=1)) & (ts <= q + pd.Timedelta(minutes=30))).any())
    esp = n_par * (1 - exp(-(len(grp) / span_h) * JAN_H))
    linhas.append(dict(tag=tag, nivel=bool(grp.nivel.iloc[0]), n=len(grp), obs=obs,
                       esperado=esp, p=p_ge(obs, esp), desc=grp["Descrição Alarme"].iloc[0]))
T = pd.DataFrame(linhas)
T = T[T.obs > 0].sort_values("p")
print(f"{'tag':18s} {'niv':>4} {'n':>5} {'obs':>4} {'esper':>7} {'p':>9}  descricao")
for r in T.itertuples():
    sig = "  ***" if r.p < 0.01 else ("  *" if r.p < 0.05 else "  (ns)")
    print(f"{r.tag:18s} {str(r.nivel)[:1]:>4} {r.n:5d} {r.obs:4d} {r.esperado:7.2f} {r.p:9.4f}  {r.desc[:42]}{sig}")

print("\n" + "="*95)
print("FOCO: PAL_6240339 (Pressao Bx. Header Oleo Lub.) -- as 4 paradas que acompanha")
print("="*95)
sub = alarmes[alarmes["Tag Alarme"] == "PAL_6240339"]
ts = pd.DatetimeIndex(sub.ts)
for q, dur in zip(paradas.ini, paradas.dur_h):
    m = (ts >= q - pd.Timedelta(hours=1)) & (ts <= q + pd.Timedelta(minutes=30))
    if m.any():
        dts = [(t - q).total_seconds()/60 for t in ts[m]]
        print(f"  parada {q:%d/%m/%Y %H:%M} ({dur:6.1f}h)  alarme a {min(dts,key=abs):+7.1f} min da queda")
print(f"\n  ativacoes totais de PAL_6240339 na janela: {len(sub)}")
print(f"  quantas NAO acompanham parada: {len(sub)} ativacoes -> so 4 paradas tocadas")
