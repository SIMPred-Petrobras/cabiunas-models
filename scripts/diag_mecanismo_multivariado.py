#!/usr/bin/env python3
"""
diag_mecanismo_multivariado.py
O multivariado eliminou falso positivo PELO MOTIVO CERTO, ou só deslocou o threshold?

A hipótese que justifica o experimento é mecânica: em [-6h,+3h] do onset, `|dT5/dt|`
mediano é 172 °C/h no FP contra 21 °C/h no TP — o falso positivo é a máquina MANOBRANDO.
O AE univariado não vê a manobra; o conjunto vê. Se a hipótese estiver certa, os FPs que
o multivariado elimina têm de ser preferencialmente os de RAMPA ALTA.

    elimina os de rampa alta  → mecanismo confirmado, o resultado deve generalizar
    elimina ao acaso          → foi deslocamento de threshold, e não generaliza

Este diagnóstico vale mesmo que o multivariado reprove no critério pré-registrado: um
ganho pelo motivo errado é mais perigoso que uma derrota limpa.

Uso:
    PYTHONPATH=. python scripts/diag_mecanismo_multivariado.py \
        --multi_task <id> [--multi_task <id> ...]
"""
from __future__ import annotations

import argparse
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
bl = _load("baseline_trivial_vs_ae")
on = _load("sweep_onset_rules_offline")
lg = _load("sweep_load_gate_offline")
ma = _load("eval_multivar_antigo")

SENSOR = "TC382_03_A"
OUT = "eval_predictive_out/mecanismo_multivariado.csv"


def episodios(health: pd.Series, q: float, inc: list) -> tuple[list, np.ndarray]:
    """Episódios do ponto de operação + rótulo TP/FP (mesma regra do resto do projeto)."""
    alert = on.sticky_bool(health >= q, on.STICKY)
    eps = ev.detect_episodes_gap(alert)
    inc_s = np.array([t.timestamp() for t in inc], dtype=float)
    hs = on.HORIZON * 3600.0
    is_tp = np.array([bool(np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
                      for s0, s1 in eps])
    return eps, is_tp


def cobre(ep, outros) -> bool:
    """O episódio `ep` sobrevive no outro braço? (sobreposição temporal com algum episódio)"""
    a0, a1 = ep
    return any((b0 <= a1) and (b1 >= a0) for b0, b1 in outros)


def rampa_no_onset(ramp: pd.Series, t0: pd.Timestamp, horas_depois: float) -> float:
    """Máximo de |dT5/dt| em [-6h, onset + horas_depois].

    Duas leituras, e a diferença importa:
      • `horas_depois=0` é CAUSAL — é o que o modelo poderia ter visto ao disparar.
      • `horas_depois=3` é a janela de `fp-reducao-portao-contexto-graduacao` (172 °C/h
        no FP × 21 no TP), que enxerga 3h ADIANTE do onset. Se a manobra do falso
        positivo vem depois do alerta, só a segunda a vê — e aí o sinal existe mas é
        inútil para um portão em tempo real.
    """
    w = ramp[(ramp.index >= t0 - pd.Timedelta(hours=6))
             & (ramp.index <= t0 + pd.Timedelta(hours=horas_depois))]
    return float(np.nanmax(w.values)) if len(w) else float("nan")


def auc_mw(x: np.ndarray, y: np.ndarray) -> float:
    """AUC de Mann-Whitney: P(x > y). 0,5 = indistinguível."""
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if not len(x) or not len(y):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--multi_task", action="append", required=True,
                   help="task id do braço multivariado (repetível: uma por semente)")
    p.add_argument("--ctrl_task", default=on.TASK_B2024,
                   help="task do controle univariado (default: b2024 da auditoria)")
    args = p.parse_args()

    running, tc03, t5 = sw.load_raw()
    ramp, _ = lg.ramp_signal(t5)

    ctrl = ev.load_mae_series(Task.get_task(task_id=args.ctrl_task), [SENSOR])[SENSOR].dropna()
    multis = {tid[:8]: ma.load_multi_mae(Task.get_task(task_id=tid), SENSOR).dropna()
              for tid in args.multi_task}

    # grade comum: os episódios têm de ser comparáveis instante a instante
    idx = ctrl.index
    for s in multis.values():
        idx = idx.intersection(s.index)
    ctrl = ctrl.reindex(idx).dropna()
    multis = {k: v.reindex(idx).dropna() for k, v in multis.items()}
    print(f"grade comum: {len(idx):,} pts  {idx.min().date()} → {idx.max().date()}")

    inc = sw.incidents_on(running, tc03, idx.min(), idx.max())
    print(f"{len(inc)} incidentes ON na grade\n")

    r_ctrl = bl.best_over_hl(ctrl, inc, running)
    h_ctrl = sw.ewma_on(ctrl, r_ctrl["hl"], running).rank(pct=True)
    eps_c, tp_c = episodios(h_ctrl, r_ctrl["threshold_q"], inc)
    fp_c = [e for e, t in zip(eps_c, tp_c) if not t]
    print(f"controle: {int(tp_c.sum())} TP / {len(fp_c)} FP "
          f"(recall_raw {r_ctrl['recall_raw']*100:.1f}%, FA {r_ctrl['fa_per_day']:.3f}, "
          f"hl={r_ctrl['hl']})")

    # rampa no onset de cada FP do controle: é a variável que a hipótese aponta.
    # Medida nas duas janelas — a causal e a da memória — porque a conclusão pode
    # depender de qual delas separa FP de TP.
    ramp_fp, ramp_tp = {}, {}
    print("  rampa |dT5/dt| — o que de fato separa FP de TP no controle:")
    for hd in (0.0, 3.0):
        ramp_fp[hd] = np.array([rampa_no_onset(ramp, s0, hd) for s0, _ in fp_c])
        ramp_tp[hd] = np.array([rampa_no_onset(ramp, s0, hd)
                                for (s0, _), t in zip(eps_c, tp_c) if t])
        jan = "[-6h, onset]  (causal)" if hd == 0 else "[-6h, onset+3h] (memória)"
        print(f"    {jan:<28} FP p50={np.nanmedian(ramp_fp[hd]):>5.0f}  "
              f"TP p50={np.nanmedian(ramp_tp[hd]):>5.0f} °C/h   "
              f"AUC={auc_mw(ramp_fp[hd], ramp_tp[hd]):.2f}")
    print()

    rows = []
    for lab, s in multis.items():
        r_m = bl.best_over_hl(s, inc, running)
        h_m = sw.ewma_on(s, r_m["hl"], running).rank(pct=True)
        eps_m, tp_m = episodios(h_m, r_m["threshold_q"], inc)
        print(f"=== multivariado {lab}: recall_raw {r_m['recall_raw']*100:.1f}%, "
              f"FA {r_m['fa_per_day']:.3f}, {int(tp_m.sum())} TP / "
              f"{int((~tp_m).sum())} FP ===")

        morto = np.array([not cobre(e, eps_m) for e in fp_c])
        n_morto = int(morto.sum())
        if n_morto == 0 or n_morto == len(fp_c):
            print(f"  {n_morto}/{len(fp_c)} FP do controle eliminados — "
                  "sem contraste para medir o mecanismo\n")
            continue

        print(f"  {n_morto}/{len(fp_c)} FP do controle eliminados")
        linha = dict(multi=lab, n_fp_ctrl=len(fp_c), n_eliminados=n_morto,
                     recall_multi=r_m["recall_raw"], fa_multi=r_m["fa_per_day"])
        for hd in (0.0, 3.0):
            r_morto, r_vivo = ramp_fp[hd][morto], ramp_fp[hd][~morto]
            auc = auc_mw(r_morto, r_vivo)
            # 0,5 = acaso. O sinal só é forte o bastante para agir a partir de ~0,70,
            # régua já usada em features-testadas-fp-vs-tp com esta mesma amostra.
            if auc >= 0.70:
                v = "CONFIRMADO — mata preferencialmente FP de manobra"
            elif auc <= 0.35:
                v = "INVERTIDO — mata os FP de máquina ESTÁVEL, o oposto da hipótese"
            else:
                v = "SEM MECANISMO — indistinguível do acaso (deslocou threshold)"
            jan = "causal" if hd == 0 else "c/ +3h"
            print(f"    [{jan}] eliminados p50={np.nanmedian(r_morto):>6.0f} × "
                  f"sobreviventes p50={np.nanmedian(r_vivo):>6.0f} °C/h  "
                  f"AUC={auc:.2f} → {v}")
            suf = "causal" if hd == 0 else "mais3h"
            linha.update({f"ramp_p50_eliminados_{suf}": float(np.nanmedian(r_morto)),
                          f"ramp_p50_sobreviventes_{suf}": float(np.nanmedian(r_vivo)),
                          f"auc_rampa_{suf}": auc, f"veredito_{suf}": v})
        print()
        rows.append(linha)

    if rows:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"Gravado: {OUT}")


if __name__ == "__main__":
    main()
