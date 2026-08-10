#!/usr/bin/env python3
"""
eval_confidence_grading_offline.py
Graduação de confiança do alerta do TC382_03_A — o alerta NÃO é suprimido nem
atrasado; ele nasce em `observação` no mesmo instante de hoje (lead preservado) e
em `t_onset + W` é revisto para `ação` ou rebaixado.

Por que W e não amplitude: o pico do score no falso positivo é tão alto quanto no
evento real (AUC 0,44), então nada que olhe a altura separa. O que separa é a
SUSTENTAÇÃO, e ela já é legível em 6h — antes do que a análise de duração sugeria:

    janela de 6h a partir do onset      FP        TP      AUC
      densidade acima do limiar        0,616     1,000    0,73
      inclinação do rank (por hora)   -0,036    -0,002    0,74

(as features de SENSOR ficaram todas em 0,52–0,66 — ver a memória
features-testadas-fp-vs-tp; estas são do próprio score e nunca tinham sido medidas.)

Este script mede a matriz de confusão da regra: quantos dos 71 FPs terminam em
`observação` e quantos dos 29 TPs sobem para `ação`. O custo relevante é TP
rebaixado — um evento real que o operador vê como ruído —, não FP mantido.

Uso:
    PYTHONPATH=. python scripts/eval_confidence_grading_offline.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
from clearml import Task

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")
on = _load("sweep_onset_rules_offline")

OUT_CSV = "eval_predictive_out/confidence_grading_TC382_03_A.csv"
W_GRID = [3.0, 6.0, 12.0]
DENS_GRID = [0.70, 0.80, 0.90, 0.95]
INCL_GRID = [-0.030, -0.010, 0.000]


def window_stats(h: pd.Series, q: float, s0: pd.Timestamp, w_h: float):
    """(densidade acima de q, inclinação do rank por hora) nas primeiras w_h horas."""
    w = h[(h.index >= s0) & (h.index <= s0 + pd.Timedelta(hours=w_h))]
    if len(w) < 3:
        return np.nan, np.nan
    x = (w.index - w.index[0]).total_seconds().values / 3600.0
    if x.ptp() <= 0:
        return float((w.values >= q).mean()), 0.0
    return float((w.values >= q).mean()), float(np.polyfit(x, w.values, 1)[0])


def main() -> None:
    running, tc03, _ = sw.load_raw()
    row = pd.read_csv(on.FLEET_CSV).set_index("sensor").loc[on.SENSOR]
    hl, q = float(row["hl"]), float(row["threshold_q"])
    mae = ev.load_mae_series(Task.get_task(task_id=on.TASK_B2024), [on.SENSOR])[on.SENSOR]
    h = sw.ewma_on(mae, hl, running).rank(pct=True)
    inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())

    alert = on.sticky_bool(h >= q, on.STICKY)
    eps = ev.detect_episodes_gap(alert)
    inc_s = np.array([t.timestamp() for t in inc], dtype=float)
    hs = on.HORIZON * 3600.0
    is_tp = [bool(np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
             for s0, s1 in eps]
    n_tp, n_fp = sum(is_tp), len(eps) - sum(is_tp)
    print(f"{n_fp} FP × {n_tp} TP  (ponto da auditoria: q={q:.4f}, hl={hl}, sticky {on.STICKY:.0f}h)\n")

    rows = []
    for w_h in W_GRID:
        stats = [window_stats(h, q, s0, w_h) for s0, _ in eps]
        dens = np.array([s[0] for s in stats])
        incl = np.array([s[1] for s in stats])
        tp = np.array(is_tp)
        # episódios curtos demais para medir a janela ficam em `ação` (não rebaixa
        # o que não deu para avaliar — o conservador aqui é NÃO silenciar)
        unmeasured = np.isnan(dens)
        print(f"=== janela W={w_h:.0f}h "
              f"({int(unmeasured.sum())} episódios curtos demais → mantidos em ação) ===")
        for d_min in DENS_GRID:
            for i_min in INCL_GRID:
                acao = unmeasured | ((dens >= d_min) & (incl >= i_min))
                tp_acao = int(np.sum(acao & tp))
                fp_obs = int(np.sum(~acao & ~tp))
                rows.append(dict(W=w_h, dens_min=d_min, incl_min=i_min,
                                 tp_acao=tp_acao, tp_total=n_tp,
                                 fp_observacao=fp_obs, fp_total=n_fp,
                                 tp_rebaixado=n_tp - tp_acao,
                                 fp_silenciado_pct=fp_obs / n_fp,
                                 tp_preservado_pct=tp_acao / n_tp))
                print(f"  dens≥{d_min:.2f} incl≥{i_min:+.3f}  "
                      f"ação: {tp_acao:>2}/{n_tp} TP ({tp_acao / n_tp:.0%})  |  "
                      f"observação: {fp_obs:>2}/{n_fp} FP ({fp_obs / n_fp:.0%})  |  "
                      f"TP rebaixado: {n_tp - tp_acao}")
        print()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Gravado: {OUT_CSV}")

    # ponto sugerido: máximo de FP silenciado com no máximo 1 TP rebaixado
    ok = df[df.tp_rebaixado <= 1].sort_values("fp_silenciado_pct", ascending=False)
    if ok.empty:
        print("\nNenhum ponto silencia FP sem rebaixar ≥2 TPs.")
        return
    b = ok.iloc[0]
    print(f"\n[sugestão] W={b.W:.0f}h dens≥{b.dens_min:.2f} incl≥{b.incl_min:+.3f}: "
          f"silencia {int(b.fp_observacao)}/{int(b.fp_total)} FP "
          f"({b.fp_silenciado_pct:.0%}) rebaixando {int(b.tp_rebaixado)} TP")


if __name__ == "__main__":
    main()
