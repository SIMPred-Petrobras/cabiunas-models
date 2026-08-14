#!/usr/bin/env python3
"""EDA: o drift do T5_AVG_A a partir de meados/2024 é artefato de regime de
operação (common-mode, removível por resíduo) ou degradação específica do sensor?

Estratégia (sem vazamento — só diagnóstico):
  1. Verifica se T5_AVG_A == média dos TC382_xx (se sim, condicionar é trivial).
  2. Confirma o drift: média mensal de T5 (só RUNNING on).
  3. Regride T5 ~ proxies de regime (vibrações TV_* e termopares vizinhos),
     ajustando os coeficientes APENAS no período pré-drift (≤ 2024-09),
     e checa se o resíduo achata a média mensal no período pós-drift.

Saída: eda_load_residual_out/{verdict.txt, fig_t5_drift_vs_residual.png}
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

CSV = "../dados/sensores_2024h2_2025_2026_30s.csv"
OUT = "eda_load_residual_out"
TC = [f"TC382_0{i}_A" for i in range(1, 7)]
TV = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
      "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A"]
PRE_DRIFT_END = "2024-09-01"   # coeficientes ajustados antes disso


def fit_residual(df: pd.DataFrame, target: str, regressors: list[str]) -> pd.Series:
    """Ajusta OLS target ~ regressors no período pré-drift, retorna resíduo na série toda."""
    train = df[df.index < PRE_DRIFT_END]
    X_tr = train[regressors].values
    X_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    y_tr = train[target].values
    coefs, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    X_all = np.column_stack([np.ones(len(df)), df[regressors].values])
    pred = X_all @ coefs
    r2_tr = 1 - np.sum((y_tr - (X_tr @ coefs))**2) / np.sum((y_tr - y_tr.mean())**2)
    return df[target] - pred, r2_tr


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cols = ["data_datetime", "T5_AVG_A", "RUNNING_A"] + TC + TV
    print(f"[1/4] Carregando {CSV} (resample 1h, só RUNNING on)...")
    df = pd.read_csv(CSV, usecols=cols)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()

    # Filtra operação (RUNNING_A on) — aceita 1/True/"on"
    run = df["RUNNING_A"]
    on = run.astype(str).str.lower().isin(["1", "1.0", "true", "on"]) | (
        pd.to_numeric(run, errors="coerce") > 0.5)
    df = df[on].drop(columns=["RUNNING_A"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df_h = df.resample("1h").mean().dropna()
    print(f"      {len(df_h):,} horas de operação | {df_h.index.min()} → {df_h.index.max()}")

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    # --- 1. T5 é só a média dos TC382? ---
    log("\n[2/4] T5_AVG_A vs média dos TC382_xx")
    tc_mean = df_h[TC].mean(axis=1)
    corr = df_h["T5_AVG_A"].corr(tc_mean)
    diff_std = (df_h["T5_AVG_A"] - tc_mean).std()
    log(f"      corr(T5, mean(TC382)) = {corr:.4f}")
    log(f"      std(T5 - mean(TC382)) = {diff_std:.3f} °C")
    trivial = corr > 0.999 and diff_std < 1.0
    log(f"      → T5 {'É essencialmente a média dos TC' if trivial else 'tem informação independente dos TC'}")

    # --- 2. Confirma o drift ---
    log("\n[3/4] Drift mensal do T5_AVG_A (média por mês)")
    monthly = df_h["T5_AVG_A"].resample("MS").mean()
    base = monthly[monthly.index < PRE_DRIFT_END].mean()
    post = monthly[monthly.index >= "2025-01-01"].mean()
    log(f"      média pré-drift (≤2024-09): {base:.2f} °C")
    log(f"      média 2025+:               {post:.2f} °C")
    log(f"      deslocamento:              {post - base:+.2f} °C")

    # --- 3. Resíduo achata o drift? ---
    log("\n[4/4] Resíduo de regime — coefs ajustados ≤2024-09")
    results = {}
    for name, regs in [("vibrações (TV_*)", TV),
                       ("termopares vizinhos", TC)]:
        if name.startswith("termopares") and trivial:
            log(f"      [pulado] {name}: T5≈média(TC), regressão trivial")
            continue
        resid, r2 = fit_residual(df_h, "T5_AVG_A", regs)
        rm = resid.resample("MS").mean()
        drift_abs = abs(rm[rm.index >= "2025-01-01"].mean() - rm[rm.index < PRE_DRIFT_END].mean())
        results[name] = (resid, r2, drift_abs)
        log(f"      {name:<22} R²(treino)={r2:.3f}  drift residual={drift_abs:+.2f} °C")

    # Veredito
    log("\n" + "=" * 60)
    raw_drift = abs(post - base)
    log(f"DRIFT BRUTO do T5: {raw_drift:.2f} °C")
    best = None
    for name, (_, r2, drift_abs) in results.items():
        reducao = 100 * (1 - drift_abs / max(raw_drift, 1e-6))
        log(f"  {name:<22} reduz drift em {reducao:5.1f}%  (R²={r2:.2f})")
        if best is None or drift_abs < best[1]:
            best = (name, drift_abs, reducao)
    if best and best[2] > 60:
        log(f"\n✅ VEREDITO: drift é em grande parte REGIME DE OPERAÇÃO.")
        log(f"   '{best[0]}' remove {best[2]:.0f}% do drift → modelar resíduo")
        log(f"   é mais elegante que re-treinar com 2024.")
    else:
        log(f"\n⚠️  VEREDITO: resíduo NÃO achata o drift suficientemente.")
        log(f"   Drift parece específico do T5 (degradação real ou recalibração).")
        log(f"   Re-treino com dados 2024 continua sendo a abordagem certa.")
    log("=" * 60)

    with open(os.path.join(OUT, "verdict.txt"), "w") as f:
        f.write("\n".join(lines))

    # Figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(monthly.index, monthly.values, "o-", label="T5_AVG_A (bruto)")
        ax[0].axvline(pd.Timestamp(PRE_DRIFT_END, tz="UTC"), color="r", ls="--", alpha=0.5,
                      label="fim pré-drift")
        ax[0].set_ylabel("°C"); ax[0].set_title("T5_AVG_A — média mensal (bruto)"); ax[0].legend()
        for name, (resid, r2, _) in results.items():
            rm = resid.resample("MS").mean()
            ax[1].plot(rm.index, rm.values, "o-", label=f"{name} (R²={r2:.2f})")
        ax[1].axvline(pd.Timestamp(PRE_DRIFT_END, tz="UTC"), color="r", ls="--", alpha=0.5)
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_ylabel("resíduo °C"); ax[1].set_title("Resíduo de regime — drift achatado?"); ax[1].legend()
        plt.tight_layout()
        figpath = os.path.join(OUT, "fig_t5_drift_vs_residual.png")
        plt.savefig(figpath, dpi=120)
        log(f"\nFigura salva: {figpath}")
    except Exception as e:
        log(f"\n[fig] erro: {e}")


if __name__ == "__main__":
    main()
