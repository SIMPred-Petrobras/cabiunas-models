#!/usr/bin/env python3
"""A nossa regua aplicada ao detector PCA walk-forward -- a simetria que faltava.

Ate aqui so tinhamos convertido o NOSSO resultado para a regua deles. O caminho
inverso precisa da serie de alertas deles, que nao esta no nosso repositorio --
mas esta publicada no ClearML, na tarefa `pca-walkforward::monitoramento_sistema
_v6b_series_para_plots` (artefato series_v6_para_plots.csv.gz, 30 s, 2024-03 a
2026-04).

DUAS RESSALVAS QUE PRECISAM VIR ANTES DO NUMERO
1. A v6 NAO e o ponto de operacao documentado. Os parametros da tarefa dizem
   threshold_x_p99=2,0 · ewma 1h · excl 3d; a documentacao diz q99,9 · ewma 2h ·
   excl 0d. E a propria tabela de hit/miss da v6 marca 2 de 8, nao 6 de 8. Entao
   isto mede a REPRODUCAO do Diego, nao o detector do documento.
2. A coluna `scored` do export esta com 1 em todas as linhas, inclusive em
   712.825 amostras com is_running=0. Nao serve de mascara. Usamos a nossa.

O que a nossa regua acrescenta e que a deles nao tem: o teste de permutacao.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import reproduz, T0

C = "/home/thallys/.clearml/cache/storage_manager/global/"
SER = C + "7e171427127d7ce9781138b35fba2599.series_v6_para_plots.csv.gz"
COLS = ["alert_and_producao", "temp_alert", "mancal_alert", "selagem_alert",
        "pressao_alert", "vote_count", "is_running"]

fin, mask, alvo, ON, idx, sel = reproduz()
s = pd.read_csv(SER, parse_dates=["timestamp"], usecols=["timestamp"] + COLS)
s["timestamp"] = s["timestamp"].dt.tz_localize("UTC")
s = s.set_index("timestamp").sort_index()

# 30 s -> 2 min pela nossa grade: um bloco esta em alerta se qualquer amostra estava
r = s.resample("2min").max().reindex(idx)
voto2 = (r["vote_count"] >= 2).fillna(False)
seu_op = r["is_running"].fillna(0).astype(bool)

print(f"\ncobertura do export na nossa grade: {r['is_running'].notna().sum()/len(idx)*100:.1f}%")
print(f"horas de operacao -- nossa mascara {mask.sum()*2/60:,.0f} h · "
      f"is_running deles {(seu_op & sel).sum()*2/60:,.0f} h\n")

def mede(nome, al, m):
    al = (al.fillna(False) & m)
    x = AV.avalia(al, alvo, m)
    pm = AV.permuta(al, m, x["det"], x["n_ev"])
    perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(x["detectados"]))
    print(f"{nome:34s} {x['det']:>3d}/8 {x['episodios']:>5d} {x['fp_mes']:>8.2f} "
          f"{x['h_fp_mes']:>8.1f} {x['lead_med']:>7.1f} {pm['nulo']:>7.2f} {pm['p']:>8.4f}"
          f"  {','.join(perd)}")
    return dict(serie=nome, det=f"{x['det']}/8", eps=x["episodios"],
                fp_mes=round(x["fp_mes"],3), h_fp_mes=round(x["h_fp_mes"],1),
                lead_h=round(x["lead_med"],1), nulo=round(pm["nulo"],2),
                p=pm["p"], perdidos=",".join(perd))

cab = (f"{'serie (repro v6 do Diego)':34s} {'det':>5s} {'eps':>5s} {'FP/mes':>8s} "
       f"{'h/mes':>8s} {'lead':>7s} {'nulo':>7s} {'p':>8s}  perdidos")
print(cab); print("-" * len(cab))
lin = [mede("alerta de producao (2 de 4)", r["alert_and_producao"].astype(bool), mask),
       mede("voto>=2 recomputado", voto2, mask)]
for c in ["temp_alert", "mancal_alert", "selagem_alert", "pressao_alert"]:
    lin.append(mede("  sinal " + c.replace("_alert",""), r[c].astype(bool), mask))
print()
lin.append(mede("producao, mascara is_running deles", r["alert_and_producao"].astype(bool), seu_op & sel))

x = AV.avalia(fin, alvo, mask); pm = AV.permuta(fin, mask, x["det"], x["n_ev"])
print("-" * len(cab))
print(f"{'NOSSO detector, mesma regua':34s} {x['det']:>3d}/8 {x['episodios']:>5d} "
      f"{x['fp_mes']:>8.2f} {x['h_fp_mes']:>8.1f} {x['lead_med']:>7.1f} "
      f"{pm['nulo']:>7.2f} {pm['p']:>8.4f}")
lin.append(dict(serie="NOSSO detector", det=f"{x['det']}/8", eps=x["episodios"],
                fp_mes=round(x["fp_mes"],3), h_fp_mes=round(x["h_fp_mes"],1),
                lead_h=round(x["lead_med"],1), nulo=round(pm["nulo"],2), p=pm["p"], perdidos=""))
pd.DataFrame(lin).to_csv("regua_neles.csv", index=False)
print("\n-> regua_neles.csv")
