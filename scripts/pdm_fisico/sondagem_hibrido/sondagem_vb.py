"""SONDAGEM: o nosso canal de vibracao, dentro da maquina de varredura do
Francisco, alcanca 7/8 ou 8/8 dentro do teto de 1 FP/mes?

CRITERIO DECLARADO ANTES DOS NUMEROS (a disciplina dele, e vale honrar):
  hoje o melhor dele dentro do teto e 6/8; o 7/8 so aparece a 1,13 FP/mes com
  filtro T5 + Mahalanobis. A pergunta e se o 5o sinal muda isso.

  8/8 no teto -> o hibrido supera os dois isolados, e e o detector final
  7/8 no teto -> o sinal transfere, mas parte do ganho esta na camada de decisao
  6/8 sem mudar -> o ganho e da combinacao sinal+decisao, nao do canal isolado

BRACO DE CONTROLE. A mesma grade SEM o sinal externo. Sem ele nao se pode
atribuir ganho ao `vb` em vez de a grade de limiar mais larga.

POR QUE O AVALIADOR OFICIAL, E NAO O DetectionReplay. Com limiar por sinal os
dois caminhos divergem na contagem de FP (24 vs 21) e a assercao interna dele
dispara. O avaliador do AutoML e o caminho que produziu a fronteira publicada.
"""
import sys, time, itertools; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import automl_clearml as automl
from automl_clearml import Trial, WalkForwardEvaluator, DataBundle, BaselinePolicy

CSV = Path("dados_locais/sensores_brutos_2025_2026_30s.csv")
TETO = 1.0
LIM_BASE = [99.0, 99.9, 99.97]     # os 4 sinais dele
LIM_VB   = [60.0, 70.0, 80.0, 90.0]  # o nosso -- a grade dele comeca em p99
POLITICAS = [BaselinePolicy(window_hours=3000), BaselinePolicy(window_hours=400)]
EIXOS = dict(ewma=["1h", "2h"], sustain=["30min", "1h"],
             confirm=[2, 3], min_alert=["0min", "30min"])

def trials(com_vb: bool):
    out = []
    for pol, ew, su, cf, ma, lb in itertools.product(
            POLITICAS, EIXOS["ewma"], EIXOS["sustain"], EIXOS["confirm"],
            EIXOS["min_alert"], LIM_BASE):
        limiares = [(lb, tvb) for tvb in LIM_VB] if com_vb else [(lb, None)]
        for lb_, tvb in limiares:
            thr = (lb_, lb_, lb_, lb_, tvb) if com_vb else lb_
            out.append((Trial(model="pca", grid="2min", baseline=pol, exclude_days=0,
                              exclude_alarm_h=1.0, ewma=ew, sustain=su, confirm=cf,
                              min_alert=ma, threshold=thr, threshold_kind="percentil"),
                        tvb))
    return out

linhas = []
for com_vb, rot in ((False, "controle 4 sinais"), (True, "5 sinais com vb")):
    b = DataBundle(automl.DATASET_ID, CSV, False, False).load()
    b.sinal_externo = com_vb
    ev = WalkForwardEvaluator(b, "2025-02", "2026-04", TETO)
    ts = trials(com_vb)
    print(f"\n[{rot}] {len(ts)} trials", flush=True)
    t0 = time.time()
    for i, (tr, tvb) in enumerate(ts, 1):
        try:
            r = ev.evaluate(tr)
        except Exception as e:
            print(f"  trial {i} falhou: {type(e).__name__}: {e}", flush=True); continue
        d = asdict(tr); d.pop("baseline", None)
        linhas.append(dict(braco=rot, com_vb=com_vb, lim_vb=tvb,
                           politica=tr.baseline.window_hours, ewma=tr.ewma,
                           sustain=tr.sustain, confirm=tr.confirm, min_alert=tr.min_alert,
                           lim_base=tr.threshold[0] if com_vb else tr.threshold,
                           det=r.eventos_detectados, fp_mes=r.fp_por_mes,
                           fp_h_mes=r.fp_horas_por_mes, lead=r.lead_medio_h,
                           dias=r.dias_avaliados, aprovado=r.aprovado,
                           vivo=getattr(r, "vivo", None)))
        if i % 40 == 0:
            print(f"  {i}/{len(ts)}  ({time.time()-t0:.0f} s)", flush=True)
    print(f"  concluido em {time.time()-t0:.0f} s", flush=True)

T = pd.DataFrame(linhas)
T.to_csv("sondagem_vb.csv", index=False)
print(f"\n{len(T)} resultados -> sondagem_vb.csv")
for rot, d in T.groupby("braco"):
    ok = d[d.fp_mes <= TETO]
    print(f"\n{rot}: {len(d)} trials, {len(ok)} dentro do teto")
    if len(ok):
        print(f"  melhor deteccao no teto: {ok.det.max()}/8")
        b = ok.sort_values(["det", "fp_h_mes"], ascending=[False, True]).iloc[0]
        print(f"  -> {b.det}/8 · {b.fp_mes:.3f} FP/mes · {b.fp_h_mes:.1f} h/mes · "
              f"lead {b.lead} h · lim_base=p{b.lim_base} lim_vb=p{b.lim_vb} "
              f"ewma={b.ewma} sust={b.sustain} conf={b.confirm} pol={b.politica}h")
