"""SEGUNDA SONDAGEM: o ganho e do SINAL ou do PORTAO?

A primeira sondagem foi negativa: o `vb` na maquina dele chega ao mesmo 6/8 do
controle e custa 4x mais falso positivo. O diagnostico apontou a estrutura de
voto -- ele tem `confirm = N de todos`, nos temos `voto >= 2 E (sp OU vb)`.

QUATRO BRACOS, para nao confundir as duas coisas:
  A  4 sinais, sem portao          <- o detector dele hoje
  B  5 sinais (com vb), sem portao <- a primeira sondagem
  C  4 sinais, portao = {spread}   <- o portao SOZINHO ajuda?
  D  5 sinais, portao = {spread, vb} <- a combinacao, que e o nosso desenho

CRITERIO, declarado antes: algum braco alcanca 7/8 ou 8/8 dentro do teto de
1 FP/mes? E, se C >> A, o portao transfere por si -- achado independente do
nosso sinal, e util para ele mesmo sem nos.
"""
import sys, time, itertools; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from pathlib import Path
import pandas as pd
import automl_clearml as automl
from automl_clearml import Trial, WalkForwardEvaluator, DataBundle, BaselinePolicy

CSV = Path("dados_locais/sensores_brutos_2025_2026_30s.csv")
TETO = 1.0
LIM_BASE = [99.0, 99.9, 99.97]
LIM_VB = [60.0, 70.0, 80.0, 90.0]
POLITICAS = [BaselinePolicy(window_hours=3000), BaselinePolicy(window_hours=400)]

BRACOS = [
    ("A · 4 sinais, sem portao",   False, set()),
    ("B · 5 sinais, sem portao",   True,  set()),
    ("C · 4 sinais, portao spread", False, {"mancal_spread"}),
    ("D · 5 sinais, portao spread+vb", True, {"mancal_spread", automl.SINAL_EXTERNO_NOME}),
]

linhas = []
for rot, com_vb, portao in BRACOS:
    automl.PORTAO_OBRIGATORIO = set(portao)
    b = DataBundle(automl.DATASET_ID, CSV, False, False).load()
    b.sinal_externo = com_vb
    ev = WalkForwardEvaluator(b, "2025-02", "2026-04", TETO)
    combos = list(itertools.product(POLITICAS, ["1h", "2h"], ["30min", "1h"],
                                    [2, 3], ["0min", "30min"], LIM_BASE,
                                    LIM_VB if com_vb else [None]))
    print(f"\n[{rot}] {len(combos)} trials  portao={sorted(portao) or 'nenhum'}", flush=True)
    t0 = time.time()
    for i, (pol, ew, su, cf, ma, lb, lvb) in enumerate(combos, 1):
        thr = (lb, lb, lb, lb, lvb) if com_vb else lb
        tr = Trial(model="pca", grid="2min", baseline=pol, exclude_days=0,
                   exclude_alarm_h=1.0, ewma=ew, sustain=su, confirm=cf,
                   min_alert=ma, threshold=thr, threshold_kind="percentil")
        try:
            r = ev.evaluate(tr)
        except Exception as e:
            print(f"  trial {i} falhou: {type(e).__name__}", flush=True); continue
        linhas.append(dict(braco=rot, com_vb=com_vb, portao=",".join(sorted(portao)) or "-",
                           lim_base=lb, lim_vb=lvb, politica=pol.window_hours, ewma=ew,
                           sustain=su, confirm=cf, min_alert=ma,
                           det=r.eventos_detectados, fp_mes=r.fp_por_mes,
                           fp_h_mes=r.fp_horas_por_mes, lead=r.lead_medio_h,
                           aprovado=r.aprovado, vivo=getattr(r, "vivo", None)))
        if i % 80 == 0:
            print(f"  {i}/{len(combos)} ({time.time()-t0:.0f} s)", flush=True)
    print(f"  concluido em {time.time()-t0:.0f} s", flush=True)

T = pd.DataFrame(linhas)
T.to_csv("sondagem_portao.csv", index=False)
print(f"\n{len(T)} resultados -> sondagem_portao.csv")
print("\n" + "=" * 96)
print("RESULTADO POR BRACO")
print("=" * 96)
print(f"{'braco':>34} {'trials':>7} {'no teto':>8} {'det max no teto':>16} {'melhor ponto':>34}")
for rot in [b[0] for b in BRACOS]:
    d = T[T.braco == rot]
    ok = d[d.fp_mes <= TETO]
    if not len(ok):
        print(f"{rot:>34} {len(d):7d} {0:8d} {'--':>16}"); continue
    bb = ok.sort_values(["det", "fp_h_mes"], ascending=[False, True]).iloc[0]
    pt = f"{bb.fp_mes:.2f} FP/mes · {bb.fp_h_mes:.1f} h/mes · conf={bb.confirm}"
    print(f"{rot:>34} {len(d):7d} {len(ok):8d} {ok.det.max():15d}/8 {pt:>34}")
