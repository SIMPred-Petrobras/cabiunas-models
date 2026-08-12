#!/usr/bin/env python3
"""
forecast_crossing.py
Reformula o TC382_03_A de DETECÇÃO DE ANOMALIA para PREVISÃO DE CRUZAMENTO.

Por que mudar. O alarme do DCS é `TC382_03_A > 760 °C` (cruzamentos/dia batem 1:1 com
alarmes/dia). Um limiar trivial na própria temperatura já entrega 81,0% @ FA 0,047 na
janela FULL, contra 62,0% @ 0,114 do autoencoder — e a fusão dos dois foi refutada. Logo
não adianta ajustar a rede: o alvo não é reconstrução, é **prever o futuro do sinal**.

    alvo   y_t = 1  se  max(TC382_03_A) em (t, t+H]  >  760 °C
    saída  p_t = P(y_t = 1)  →  entra como health-index no MESMO maquinário de avaliação

⚠️ O RÓTULO DE TREINO é o cruzamento calculado (denso: ~10^5 rótulos, contra os 79
incidentes que o AE tinha). A AVALIAÇÃO continua sendo o alarme REGISTRADO no DCS, pelo
mesmo `ev.best_point_for_sensor` de sempre — é isso que mantém os números comparáveis com
os 81,0%/62,0% já na mesa. Não misturar os dois papéis.

⚠️ NUNCA comparar recall entre horizontes: em H=72 h qualquer detector acerta mais só
porque a janela de crédito é maior. A leitura é braço-contra-braço DENTRO de um H, e a
grandeza de interesse é como o gap (supervisionado − trivial) evolui com H.

Tudo roda na grade do MAE do controle `3b34a312` — mesma grade, mesmo duty, mesmo
denominador de incidente para todos os braços, inclusive o AE de referência.

Etapas (`--stage`):
    horizon      escada de braços × H ∈ {8,24,72}, ponto buscado na janela (comparável
                 ao legado)
    frozen       split temporal honesto: modelo E threshold fixados no treino, congelados,
                 aplicados no teste
    walkforward  refit trimestral em janela expansiva → série de health genuinamente fora
                 de amostra sobre ~22 meses
    leakprobe    sonda de vazamento: com o rótulo de treino embaralhado, tem de colapsar

Uso:
    PYTHONPATH=. python scripts/forecast_crossing.py --stage horizon
    PYTHONPATH=. python scripts/forecast_crossing.py --stage all
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from clearml import Task

warnings.filterwarnings("ignore", category=FutureWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

SENSOR = "TC382_03_A"
RAW = "../dados/sensores_full_2024_2026_30s.csv"
TASK = "3b34a312aa234aae9ac1f5c1f922791f"     # controle = melhor AE atual, congelado
CACHE = "eval_predictive_out/.cache_forecast_features.parquet"
OUTDIR = "eval_predictive_out"

SETPOINT = 760.0
HOT = 500.0                 # árbitro FÍSICO de máquina ligada (RUNNING_A é não-confiável)
GRID = ev.SAMPLING_INTERVAL  # "5min"
PPH = int(pd.Timedelta("1h") / pd.Timedelta(GRID))   # pontos por hora = 12

HORIZONS = [8.0, 24.0, 72.0]
STICKY, FA_BUDGET = 12.0, 1.0
MAX_DUTY, MAX_STICKY = 0.35, 0.25
HL_GRID = [0.5, 1.0, 2.0, 4.0]
SEED = 17

SIBS = ["TC382_01_A", "TC382_02_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]
PDI = "954005_624_PDI_0317"
NEEDED = ["data_datetime", SENSOR, "T5_AVG_A", "RUNNING_A", PDI] + SIBS

T_FULL = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC"))
T_TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
JANELAS = [("2024 jun–dez", "2024-06-01", "2025-01-01"),
           ("2026 jan–abr", "2026-01-01", "2026-05-01"),
           ("OOS jul/25→abr/26", "2025-07-01", "2026-05-01"),
           ("FULL jan/24→abr/26", "2024-01-01", "2026-05-01")]

BASELINE_CSV = f"{OUTDIR}/baseline_trivial_vs_ae.csv"
ANCHOR_JANELA = "FULL jan/24→abr/26"


def check_anchor(df: pd.DataFrame) -> None:
    """O braço A0 em H=8 h TEM de reproduzir os 81,0% @ FA 0,047 do baseline. É o mesmo
    sinal, o mesmo maquinário e a mesma grade — se não bater, alguma coisa mudou de baixo
    para cima e nenhum número deste script vale. Aborta em vez de reportar."""
    if not os.path.exists(BASELINE_CSV):
        print(f"[âncora] {BASELINE_CSV} ausente — pulando (rode baseline_trivial_vs_ae.py)")
        return
    base = pd.read_csv(BASELINE_CSV).query("janela == @ANCHOR_JANELA").set_index("braco")
    exp = base.loc[[i for i in base.index if i.startswith("temp")][0]]
    got = df.query("H == 8.0 and janela == @ANCHOR_JANELA and braco.str.startswith('A0')")
    if got.empty:
        return
    g = got.iloc[0]
    ok = abs(g.recall_raw - exp.recall_raw) < 0.02 and abs(g.fa_per_day - exp.fa_per_day) < 0.01
    print(f"\n[âncora] A0 H=8h FULL: {g.recall_raw:.1%} @ {g.fa_per_day:.3f}  "
          f"(baseline {exp.recall_raw:.1%} @ {exp.fa_per_day:.3f})  → {'OK' if ok else 'DIVERGE'}")
    if not ok:
        raise SystemExit("[âncora] A0 não reproduz o baseline — resultados invalidados.")


# ---------------------------------------------------------------------------
# Features causais — nenhuma janela pode atravessar t
# ---------------------------------------------------------------------------

def _slope(v: pd.Series, hours: float) -> pd.Series:
    k = max(1, int(round(hours * PPH)))
    return (v - v.shift(k)) / hours


def build_features(force: bool = False) -> pd.DataFrame:
    """Matriz de features na grade do MAE. Cacheada — o CSV bruto tem 1 GB."""
    if os.path.exists(CACHE) and not force:
        print(f"[features] cache: {CACHE}")
        return pd.read_parquet(CACHE)

    print("[features] lendo o CSV bruto (1 GB, ~1 min)...", flush=True)
    raw = pd.read_csv(RAW, usecols=NEEDED, low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    raw = raw[(raw.index >= T_FULL[0]) & (raw.index < T_FULL[1])]
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # duas reamostragens: média para features, MÁXIMO do TC03 para o rótulo (o pico de
    # 30 s é o que dispara o alarme; a média de 5 min o achataria)
    g = raw.resample(GRID)
    f = g.mean()
    tc_max = g[SENSOR].max()

    tc = f[SENSOR]
    hot = tc > HOT
    X = pd.DataFrame(index=f.index)

    X["tc"] = tc
    for hl in HL_GRID:
        X[f"tc_ewma{hl}"] = tc.ewm(halflife=max(1, int(round(hl * PPH)))).mean()
    for h in (0.25, 1.0, 4.0):
        X[f"tc_slope{h}"] = _slope(tc, h)
    for h in (1, 6, 24):
        n = h * PPH
        X[f"tc_max{h}h"] = tc.rolling(n, min_periods=1).max()
        X[f"tc_std{h}h"] = tc.rolling(n, min_periods=2).std()
    for lim in (700, 730, 760):
        for h in (6, 24):
            X[f"tc_frac{lim}_{h}h"] = (tc > lim).rolling(h * PPH, min_periods=1).mean()

    sib = f[SIBS].mean(axis=1)
    X["sib_mean"] = sib
    X["sib_spread"] = tc - sib
    X["sib_spread_slope1h"] = _slope(tc - sib, 1.0)

    t5 = f["T5_AVG_A"]
    X["t5"] = t5
    X["t5_slope1h"] = _slope(t5, 1.0)
    X["t5_ewma2h"] = t5.ewm(halflife=2 * PPH).mean()

    pdi = f[PDI]
    X["pdi"] = pdi
    X["pdi_slope1h"] = _slope(pdi, 1.0)

    # horas desde a partida, pelo árbitro físico
    run_id = (~hot).cumsum()
    X["h_desde_partida"] = (hot.groupby(run_id).cumsum() / PPH).where(hot)

    # rótulo por horizonte: max do TC03 em (t, t+H] — grade regular, então rolling
    # invertido é exato. O `shift(-1)` exclui o próprio t.
    fut = tc_max.shift(-1)
    for H in HORIZONS:
        n = int(round(H * PPH))
        fmax = fut[::-1].rolling(n, min_periods=1).max()[::-1]
        X[f"y{int(H)}"] = (fmax > SETPOINT).astype(float)
        # as últimas H horas têm janela futura truncada → rótulo não confiável
        X.loc[X.index > X.index[-1] - pd.Timedelta(hours=H), f"y{int(H)}"] = np.nan

    X["hot"] = hot.astype(float)
    X["running"] = f["RUNNING_A"]

    # feature do AE (só o braço A3 usa) + a grade oficial de avaliação
    mae = ev.load_mae_series(Task.get_task(task_id=TASK), [SENSOR])[SENSOR]
    mae = mae[(mae.index >= T_FULL[0]) & (mae.index < T_FULL[1])]
    X = X.reindex(mae.index, method="nearest")
    X["ae_mae"] = mae
    X["ae_ewma2h"] = mae.ewm(halflife=2 * PPH).mean()

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    X.to_parquet(CACHE)
    print(f"[features] {X.shape[0]:,} linhas × {X.shape[1]} colunas → {CACHE}")
    return X


def feature_cols(X: pd.DataFrame, com_ae: bool) -> list[str]:
    drop = {"hot", "running", "ae_mae", "ae_ewma2h"} | {f"y{int(h)}" for h in HORIZONS}
    cols = [c for c in X.columns if c not in drop]
    return cols + ["ae_ewma2h"] if com_ae else list(cols)


# ---------------------------------------------------------------------------
# Braços
# ---------------------------------------------------------------------------

def fit_predict(X: pd.DataFrame, cols: list[str], H: float,
                fit_mask: pd.Series, kind: str, seed: int = SEED,
                shuffle_y: bool = False) -> pd.Series:
    """Ajusta no `fit_mask` e devolve p_t sobre TODA a grade quente."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    y = X[f"y{int(H)}"]
    # treina só com máquina quente E ligada; PREVÊ em toda a grade, porque a máscara ON
    # da avaliação é aplicada depois por `ewma_on` — exatamente como nos braços A0 e REF.
    # Sem isso os braços teriam denominadores de duty diferentes e não seriam comparáveis.
    tr = fit_mask & (X["hot"] > 0.5) & (X["running"] > 0.5) & y.notna()
    yt = y[tr].values
    if shuffle_y:
        yt = np.random.default_rng(seed).permutation(yt)
    if len(np.unique(yt)) < 2:
        return pd.Series(np.nan, index=X.index)

    if kind == "lr":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=seed))
    else:
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=seed)

    model.fit(X.loc[tr, cols], yt)
    return pd.Series(model.predict_proba(X[cols])[:, 1], index=X.index)


def best_over_hl(score: pd.Series, inc: list, running: pd.Series, H: float) -> dict:
    """Varre half-life e devolve o melhor ponto — mesma regra do protocolo honesto."""
    best = None
    for hl in HL_GRID:
        h = sw.ewma_on(score, hl, running).rank(pct=True)
        if h.empty:
            continue
        r = ev.best_point_for_sensor(h, inc, horizon_hours=H, sticky_hours=STICKY,
                                     fa_budget=FA_BUDGET, n_thresholds=120,
                                     max_duty_cycle=MAX_DUTY, max_sticky_duty=MAX_STICKY)
        r["hl"] = hl
        key = (r.get("recall_raw") or 0.0, -(r.get("fa_per_day") or 9e9))
        if best is None or key > (best.get("recall_raw") or 0.0,
                                  -(best.get("fa_per_day") or 9e9)):
            best = r
    return best or {}


def score_at(health: pd.Series, q: float, inc: list, H: float) -> dict:
    """Métricas num threshold FIXO (ponto congelado do treino)."""
    if health.empty or not inc:
        return {}
    total_days = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s = np.array([t.timestamp() for t in inc])
    alert = ev.apply_sticky(health, q, STICKY)
    eps = ev.detect_episodes_gap(alert)
    raw_s = np.array([t.timestamp() for t in health.index[health >= q]])
    hs = H * 3600.0
    n_raw = sum(1 for ti in inc_s if raw_s.size and np.any((raw_s >= ti - hs) & (raw_s <= ti)))
    n_fp = sum(1 for (s0, s1) in eps
               if not np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())))
    return dict(recall_raw=n_raw / len(inc_s), fa_per_day=n_fp / max(total_days, 1.0),
                duty_sticky=float(alert.mean()), n_fp=n_fp)


def arms_for(X: pd.DataFrame, H: float, fit_mask: pd.Series,
             shuffle_y: bool = False) -> dict[str, pd.Series]:
    """Os cinco braços, como séries de score cru (o EWMA/rank vem depois)."""
    cols_sem = feature_cols(X, com_ae=False)
    cols_com = feature_cols(X, com_ae=True)
    return {
        "A0 trivial (limiar de T)": X["tc"],
        "A1 logística": fit_predict(X, cols_sem, H, fit_mask, "lr", shuffle_y=shuffle_y),
        "A2 GBM": fit_predict(X, cols_sem, H, fit_mask, "gbm", shuffle_y=shuffle_y),
        "A3 GBM + AE": fit_predict(X, cols_com, H, fit_mask, "gbm", shuffle_y=shuffle_y),
        "REF autoencoder": X["ae_mae"],
    }


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def _running(X: pd.DataFrame) -> pd.Series:
    """Máscara ON usada na avaliação. É o RUNNING_A, e NÃO o árbitro físico `hot`, por um
    motivo só: é o que `baseline_trivial_vs_ae.py` usou para chegar aos 79 incidentes e
    aos 81,0%. Trocar aqui mudaria o denominador e quebraria a comparabilidade — o filtro
    físico de 500 °C continua aplicado dentro de `sw.incidents_on`."""
    return X["running"]


def stage_horizon(X: pd.DataFrame) -> pd.DataFrame:
    """Escada × horizonte, ponto buscado na janela — comparável ao legado."""
    running = _running(X)
    tc = X["tc"]
    rows = []
    for H in HORIZONS:
        # ajuste no treino (jan/24→jun/25) e leitura em todas as janelas: as janelas
        # 2024 e OOS ficam, respectivamente, dentro e fora do treino — reportado como tal
        fit_mask = pd.Series(X.index < T_TRAIN_END, index=X.index)
        arms = arms_for(X, H, fit_mask)
        for wlab, a, b in JANELAS:
            t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
            inc = sw.incidents_on(running, tc, t0, t1)
            print(f"\n=== H={int(H)}h · {wlab} — {len(inc)} incidentes ON "
                  f"{'(treino)' if t1 <= T_TRAIN_END else '(fora do treino)' if t0 >= T_TRAIN_END else '(misto)'} ===")
            print(f"  {'braço':<26}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'lead h':>9}{'hl':>6}")
            for name, sc in arms.items():
                s = sc[(sc.index >= t0) & (sc.index < t1)].dropna()
                r = best_over_hl(s, inc, running, H) if not s.empty else {}
                rr = r.get("recall_raw")
                rows.append(dict(H=H, janela=wlab, braco=name, n_inc=len(inc),
                                 recall_raw=rr, fa_per_day=r.get("fa_per_day"),
                                 duty=r.get("duty_sticky"),
                                 lead_h=r.get("median_lead_hours"), hl=r.get("hl")))
                print(f"  {name:<26}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                      f"{r.get('fa_per_day', float('nan')):>10.3f}"
                      f"{r.get('duty_sticky', float('nan')):>8.2f}"
                      f"{r.get('median_lead_hours', float('nan')):>9.1f}"
                      f"{str(r.get('hl')):>6}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/forecast_crossing_horizon.csv", index=False)
    check_anchor(df)
    _verdict_horizon(df)
    return df


def _verdict_horizon(df: pd.DataFrame) -> None:
    print("\n=== O GAP (supervisionado − trivial) CRESCE COM O HORIZONTE? ===")
    print("  Regra: comparar SÓ dentro de um mesmo H. Recall entre H's não é comparável.")
    print(f"\n  {'janela':<22}{'H':>5}" + "".join(f"{n[:14]:>16}" for n in
                                                 ["A1 logística", "A2 GBM", "A3 GBM+AE", "REF AE"]))
    for wlab, _, _ in JANELAS:
        for H in HORIZONS:
            d = df[(df.janela == wlab) & (df.H == H)].set_index("braco")
            if d.empty or pd.isna(d.recall_raw.get("A0 trivial (limiar de T)")):
                continue
            a0 = d.recall_raw["A0 trivial (limiar de T)"]
            cells = []
            for n in ["A1 logística", "A2 GBM", "A3 GBM + AE", "REF autoencoder"]:
                v = d.recall_raw.get(n)
                cells.append("—" if pd.isna(v) else f"{v*100:5.1f}% ({(v-a0)*100:+5.1f})")
            print(f"  {wlab:<22}{int(H):>5}" + "".join(f"{c:>16}" for c in cells))
    print("\n  (Δpp contra o braço trivial A0 da MESMA linha; > +10pp = acima do ruído de semente)")


def stage_frozen(X: pd.DataFrame) -> pd.DataFrame:
    """Modelo E threshold fixados no treino, congelados, aplicados no teste.

    Corrige o vício que atravessa o projeto: o `q` sempre foi escolhido na mesma janela
    em que é reportado. Aqui o rank vem da CDF empírica do TREINO — nada do teste entra
    na calibração.
    """
    running = _running(X)
    tc = X["tc"]
    tr_mask = pd.Series(X.index < T_TRAIN_END, index=X.index)
    t0, t1 = T_TRAIN_END, T_FULL[1]
    inc_tr = sw.incidents_on(running, tc, T_FULL[0], T_TRAIN_END)
    inc_te = sw.incidents_on(running, tc, t0, t1)

    rows = []
    for H in HORIZONS:
        arms = arms_for(X, H, tr_mask)
        print(f"\n=== H={int(H)}h · ponto CONGELADO no treino "
              f"({len(inc_tr)} inc treino → {len(inc_te)} inc teste) ===")
        print(f"  {'braço':<26}{'recall teste':>14}{'FA/dia':>10}{'duty':>8}{'hl':>6}{'q':>9}")
        for name, sc in arms.items():
            s = sc.dropna()
            if s.empty:
                continue
            s_tr = s[s.index < T_TRAIN_END]
            best, bq, bhl = None, None, None
            for hl in HL_GRID:
                h_tr = sw.ewma_on(s_tr, hl, running).rank(pct=True)
                if h_tr.empty:
                    continue
                r = ev.best_point_for_sensor(h_tr, inc_tr, horizon_hours=H,
                                             sticky_hours=STICKY, fa_budget=FA_BUDGET,
                                             n_thresholds=120, max_duty_cycle=MAX_DUTY,
                                             max_sticky_duty=MAX_STICKY)
                key = (r.get("recall_raw") or 0.0, -(r.get("fa_per_day") or 9e9))
                if best is None or key > (best.get("recall_raw") or 0.0,
                                          -(best.get("fa_per_day") or 9e9)):
                    best, bq, bhl = r, r["threshold_q"], hl
            if best is None:
                continue
            # CDF do TREINO aplicada ao teste — calibração congelada de verdade
            ew_tr = sw.ewma_on(s_tr, bhl, running)
            ref = np.sort(ew_tr.values)
            ew_te = sw.ewma_on(s[(s.index >= t0) & (s.index < t1)], bhl, running)
            h_te = pd.Series(np.searchsorted(ref, ew_te.values, side="right") / len(ref),
                             index=ew_te.index)
            m = score_at(h_te, bq, inc_te, H)
            rows.append(dict(H=H, braco=name, n_inc_treino=len(inc_tr), n_inc_teste=len(inc_te),
                             recall_treino=best.get("recall_raw"), **m, hl=bhl, q=bq))
            print(f"  {name:<26}{m.get('recall_raw', float('nan'))*100:13.1f}%"
                  f"{m.get('fa_per_day', float('nan')):>10.3f}"
                  f"{m.get('duty_sticky', float('nan')):>8.2f}{bhl:>6}{bq:>9.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/forecast_crossing_frozen.csv", index=False)
    return df


def stage_walkforward(X: pd.DataFrame) -> pd.DataFrame:
    """Refit trimestral em janela expansiva → health genuinamente fora de amostra.

    Primeiro treino = jan–jun/2024; a partir daí cada trimestre é previsto por um modelo
    que só viu o passado. A série concatenada é o primeiro número "FULL honesto" do
    projeto — e sua janela (jul/24→abr/26) tem denominador PRÓPRIO, que não é o dos 79.
    """
    running = _running(X)
    tc = X["tc"]
    cortes = pd.date_range("2024-07-01", "2026-04-01", freq="QS", tz="UTC")
    rows = []
    for H in HORIZONS:
        preds: dict[str, list[pd.Series]] = {}
        for i, c0 in enumerate(cortes):
            c1 = cortes[i + 1] if i + 1 < len(cortes) else T_FULL[1]
            fit_mask = pd.Series(X.index < c0, index=X.index)
            seg = (X.index >= c0) & (X.index < c1)
            print(f"  [wf H={int(H)}h] treino < {c0.date()} → prevê {c0.date()}..{c1.date()}",
                  flush=True)
            for name, sc in arms_for(X, H, fit_mask).items():
                preds.setdefault(name, []).append(sc[seg])
        wf0, wf1 = cortes[0], T_FULL[1]
        inc = sw.incidents_on(running, tc, wf0, wf1)
        print(f"\n=== H={int(H)}h · WALK-FORWARD {wf0.date()}→{wf1.date()} — "
              f"{len(inc)} incidentes ON (fora de amostra) ===")
        print(f"  {'braço':<26}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")
        for name, parts in preds.items():
            s = pd.concat(parts).sort_index().dropna()
            r = best_over_hl(s, inc, running, H) if not s.empty else {}
            rr = r.get("recall_raw")
            rows.append(dict(H=H, braco=name, n_inc=len(inc), recall_raw=rr,
                             fa_per_day=r.get("fa_per_day"), duty=r.get("duty_sticky"),
                             lead_h=r.get("median_lead_hours"), hl=r.get("hl")))
            print(f"  {name:<26}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}{str(r.get('hl')):>6}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/forecast_crossing_walkforward.csv", index=False)
    return df


def stage_leakprobe(X: pd.DataFrame) -> pd.DataFrame:
    """O PISO DO ACASO por horizonte — sem ele nenhum recall é interpretável.

    Duas sondas, nas duas janelas de leitura:

      RUÍDO PURO   score aleatório. Com duty travado em 0,25, sticky de 12 h e horizonte
                   de 72 h, um alerta que fica ligado um quarto do tempo cai perto de
                   qualquer incidente por construção. Este é o piso a descontar: um recall
                   de 91% a 72 h só é resultado se o ruído ficar MUITO abaixo dele.
      RÓTULO EMBARALHADO  o GBM com y permutado no treino. Se não colapsar até o piso,
                   alguma janela de feature atravessa `t` ou a avaliação vaza.

    Média de 5 sementes no ruído — uma amostra só de ruído é, ela mesma, ruído.
    """
    running, tc = _running(X), X["tc"]
    rows = []
    for wlab, a, b in [("OOS jul/25→abr/26", "2025-07-01", "2026-05-01"),
                       ("FULL jan/24→abr/26", "2024-01-01", "2026-05-01")]:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc, t0, t1)
        fit_mask = pd.Series(X.index < T_TRAIN_END, index=X.index)
        idx = X.index[(X.index >= t0) & (X.index < t1)]
        for H in HORIZONS:
            print(f"\n=== PISO DO ACASO · H={int(H)}h · {wlab} — {len(inc)} incidentes ===")
            recs, fas = [], []
            for seed in range(5):
                rnd = pd.Series(np.random.default_rng(1000 + seed).random(len(idx)), index=idx)
                r = best_over_hl(rnd, inc, running, H)
                recs.append(r.get("recall_raw") or 0.0)
                fas.append(r.get("fa_per_day") or float("nan"))
            piso, piso_sd = float(np.mean(recs)), float(np.std(recs))
            sc = fit_predict(X, feature_cols(X, com_ae=False), H, fit_mask, "gbm", shuffle_y=True)
            s = sc[(sc.index >= t0) & (sc.index < t1)].dropna()
            rs = best_over_hl(s, inc, running, H)
            shuf = rs.get("recall_raw") or 0.0
            rows.append(dict(janela=wlab, H=H, n_inc=len(inc), piso_ruido=piso,
                             piso_sd=piso_sd, piso_fa=float(np.nanmean(fas)),
                             rotulo_embaralhado=shuf))
            print(f"  ruído puro (5 sementes)   recall_raw={piso:.1%} ± {piso_sd:.1%}"
                  f"   FA={np.nanmean(fas):.3f}")
            print(f"  rótulo EMBARALHADO        recall_raw={shuf:.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTDIR}/forecast_crossing_chancefloor.csv", index=False)
    print("\n  Leitura: recall só conta o quanto EXCEDE o piso da mesma linha. Se o piso a")
    print("  72 h já for alto, o horizonte longo não é mérito de modelo — é aritmética da")
    print("  janela de crédito, e comparar 72 h com 8 h em recall bruto é enganoso.")
    return df


def register_clearml(stage: str, tables: dict[str, pd.DataFrame]) -> None:
    """Registra a rodada como task do ClearML.

    Convenção do repo: `src/main.py` é o ÚNICO que faz `Task.init` — os ~30 scripts de
    análise só LEEM artefatos com `Task.get_task`. Este script começou nessa segunda
    família (varredura exploratória), mas o braço supervisionado é um CANDIDATO A MODELO,
    e candidato precisa de rastro. Daí a flag: exploração roda solta, resultado que vai
    para a mesa fica registrado.

    Não usa `execute_remotely`: o GBM leva segundos de CPU: a fila remota existe para o
    treino do AE em GPU. Aqui o ClearML serve para RASTREIO, não para computação.
    """
    task = Task.init(project_name="TesteMLCab", task_name=f"forecast-crossing-{stage}",
                     output_uri=True, reuse_last_task_id=False)
    task.connect({"sensor": SENSOR, "setpoint": SETPOINT, "horizontes": HORIZONS,
                  "sticky_h": STICKY, "fa_budget": FA_BUDGET, "max_duty": MAX_DUTY,
                  "max_sticky_duty": MAX_STICKY, "hl_grid": HL_GRID,
                  "treino_ate": str(T_TRAIN_END.date()), "ae_ref_task": TASK,
                  "seed": SEED}, name="protocolo")
    for name, df in tables.items():
        task.upload_artifact(name, df)
        task.get_logger().report_table(title=name, series=name, table_plot=df)
    print(f"[clearml] task {task.id} — {task.get_output_log_web_page()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="horizon",
                    choices=["horizon", "frozen", "walkforward", "leakprobe", "all"])
    ap.add_argument("--rebuild", action="store_true", help="ignora o cache de features")
    ap.add_argument("--clearml", action="store_true",
                    help="registra a rodada como task (rastreio, não execução remota)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    X = build_features(force=args.rebuild)
    hot = X["hot"] > 0.5
    print(f"[dados] {len(X):,} pontos na grade do MAE · {int(hot.sum()):,} quentes "
          f"({hot.mean():.0%}) · {X.index[0].date()} → {X.index[-1].date()}")
    for H in HORIZONS:
        y = X.loc[hot, f"y{int(H)}"]
        print(f"        rótulo H={int(H):>2}h: {int(y.sum()):,} positivos "
              f"({y.mean():.1%} da amostra quente)")

    tables: dict[str, pd.DataFrame] = {}
    if args.stage in ("horizon", "all"):
        tables["horizon"] = stage_horizon(X)
    if args.stage in ("leakprobe", "all"):
        stage_leakprobe(X)
    if args.stage in ("frozen", "all"):
        tables["frozen"] = stage_frozen(X)
    if args.stage in ("walkforward", "all"):
        tables["walkforward"] = stage_walkforward(X)

    if args.clearml and tables:
        register_clearml(args.stage, tables)


if __name__ == "__main__":
    main()
