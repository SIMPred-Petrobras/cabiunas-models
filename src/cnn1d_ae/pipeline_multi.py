"""
Task 2: Pipeline multivariado — treina UM único CNN-1D AE sobre TODOS os sensores
empilhados como canais independentes.

Diferença do pipeline univariado:
  - Input shape: (n_seq, TIME_STEPS, n_sensores)   vs  (n_seq, TIME_STEPS, n_features_1_sensor)
  - A máscara de exclusão do treino usa alarmes de TODOS os sensores combinados
  - O threshold é calibrado sobre o MAE geral (média de todos os canais)
  - O report de avaliação compara as predições contra TODAS as falhas (todos os sensores)
  - Por-sensor MAE é calculado para identificar qual canal contribuiu mais para cada anomalia

Limitações desta implementação:
  - Todos os sensores são normalizados independentemente (z-score por coluna)
  - Apenas sinal bruto entra no modelo (sem feature engineering por sensor)
  - Timestamps são alinhados por inner-join: só instantes com TODOS os sensores disponíveis
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config, resolve_output_dir
from .model import setup_gpu
from .preprocess import (
    build_sensor_dataframe,
    apply_hampel_filter,
    build_exclusion_mask,
    build_startup_exclusion_mask,
)
from .sequences import make_sequences, train_val_split
from .tuning import run_tuner, refit_best_model
from .scoring import (
    reconstruction_mae_per_seq,
    reconstruction_mae_and_per_sensor,
    compute_threshold,
    compute_threshold_alarm_optimized,
    apply_adaptive_monthly_threshold,
    map_seq_to_point_anomalies,
    build_operational_state_from_running,
    mask_anomaly_seq_by_operational_state,
    build_sequence_scores_df,
    compute_anomaly_rate_per_day,
    evaluate_alarm_detection,
)
from .plots import (
    plot_loss,
    plot_hist_mae,
    plot_hist_normal_vs_anomaly,
    plot_series_with_anomalies,
    plot_series_alarm_anomaly_subplots,
    plot_predictive_curve,
    plot_health_index_over_time,
)
from .predictive import (
    extract_incidents,
    compute_health_index_ewma,
    compute_predictive_curve,
    compute_predictive_curve_per_sensor,
    pick_operating_point,
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_running_col(cfg: PipelineConfig, df_raw: pd.DataFrame, index: pd.DatetimeIndex) -> Optional[pd.Series]:
    if not cfg.RUNNING_COL:
        return None
    src = df_raw if cfg.TRAIN_SOURCE.lower() == "raw" else None
    if src is None or cfg.RUNNING_COL not in src.columns:
        return None
    running = (
        src.drop_duplicates(subset=[cfg.TIME_COL])
        .set_index(cfg.TIME_COL)[cfg.RUNNING_COL]
    )
    return pd.to_numeric(running, errors="coerce").reindex(index).fillna(0.0)


def _per_sensor_mae(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """MAE por sequência por sensor. Shape: (n_seq, n_sensors)."""
    return np.mean(np.abs(x_pred - x_true), axis=1)   # axis=1 = eixo tempo


def _mse_per_seq(model, x: np.ndarray, batch_size: int) -> np.ndarray:
    """MSE puro por sequência (média sobre tempo×features), em batches.
    Não inclui o termo L2 (ao contrário do model.evaluate 'loss')."""
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float32)
    bs = max(1, int(batch_size))
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, bs):
        xb = x[s:s + bs]
        pb = np.asarray(model.predict_on_batch(xb))
        out[s:s + len(xb)] = np.mean((pb - xb) ** 2, axis=(1, 2))
    return out


def _normalize_multi(
    df_normal: pd.DataFrame,
    df_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    center = df_normal.mean(axis=0)
    scale  = df_normal.std(axis=0).replace(0, 1.0)
    return (df_normal - center) / scale, (df_all - center) / scale, center, scale


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline_multivariado(
    cfg: PipelineConfig,
    df_alarm: pd.DataFrame,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    sensors: List[str],
) -> Dict:
    """
    Treina um único autoencoder com todos os `sensors` como canais simultâneos.

    Retorna dicionário com métricas de avaliação e caminhos dos arquivos gerados.
    """
    cfg = PipelineConfig(**asdict(cfg))
    setup_gpu()

    joint_label = "MULTI_" + "_".join(s[:6] for s in sensors[:5])
    if len(sensors) > 5:
        # Evita '+' no nome (quebra URL de artefato no ClearML / 404 no download).
        joint_label += f"_{len(sensors)-5}more"

    out_dirs = {
        "root":       resolve_output_dir(cfg, joint_label),
        "tuner":      os.path.join(resolve_output_dir(cfg, joint_label), "tuner"),
        "best_model": os.path.join(resolve_output_dir(cfg, joint_label), "best_model"),
        "figs":       os.path.join(resolve_output_dir(cfg, joint_label), "figs"),
        "csv":        os.path.join(resolve_output_dir(cfg, joint_label), "csv"),
    }
    for p in out_dirs.values():
        os.makedirs(p, exist_ok=True)
    save_run_config(cfg, out_dirs)

    print("\n" + "=" * 60)
    print(f"[MULTI] Treinando modelo multivariado com {len(sensors)} sensores")
    print(f"[MULTI] Sensores: {sensors}")
    print(f"[MULTI] Output: {out_dirs['root']}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Preprocessar cada sensor individualmente (sentinel + interpolação)
    # ------------------------------------------------------------------
    sensor_dfs: Dict[str, pd.DataFrame] = {}
    for sensor in sensors:
        print(f"  [PREPROC] {sensor} ...", end=" ", flush=True)
        try:
            df_s, _ = build_sensor_dataframe(cfg, df_feat, df_raw, sensor)
            df_s = apply_hampel_filter(df_s, sensor, cfg)
            sensor_dfs[sensor] = df_s[[sensor]]
            print("OK")
        except Exception as exc:
            print(f"ERRO ({exc}) — sensor ignorado")

    if len(sensor_dfs) < 2:
        return {"skipped": True, "reason": "menos de 2 sensores preprocessados com sucesso"}

    sensors_ok = list(sensor_dfs.keys())

    # ------------------------------------------------------------------
    # 2. Inner-join: só timestamps presentes em TODOS os sensores
    # ------------------------------------------------------------------
    print(f"[MULTI] Alinhando {len(sensors_ok)} séries...")
    common_index = None
    for df_s in sensor_dfs.values():
        common_index = df_s.index if common_index is None else common_index.intersection(df_s.index)

    common_index = common_index.sort_values()
    print(f"[MULTI] Índice comum: {len(common_index)} pontos "
          f"({common_index.min()} → {common_index.max()})")

    df_all = pd.DataFrame(index=common_index)
    for sensor, df_s in sensor_dfs.items():
        df_all[sensor] = df_s.loc[common_index, sensor]

    if df_all.isna().any().any():
        df_all = df_all.ffill().bfill().fillna(0.0)

    # ------------------------------------------------------------------
    # 3. Máscara de exclusão combinada (alarmes de TODOS os sensores)
    # ------------------------------------------------------------------
    all_alarm_times = pd.to_datetime(
        df_alarm["Data da Ocorrencia"], errors="coerce"
    ).dropna().drop_duplicates()

    alarm_exclude = build_exclusion_mask(common_index, all_alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    exclude = alarm_exclude.copy() if hasattr(alarm_exclude, "copy") else np.array(alarm_exclude)

    # RUNNING_COL (usa primeiro sensor como referência)
    running = _load_running_col(cfg, df_raw, common_index)
    if running is not None:
        n_off = int((running <= 0.5).sum())
        print(f"[RUNNING_COL] {n_off} pontos OFF excluídos do treino.")
        exclude = exclude | (running <= 0.5)

    # Startup exclusion (referencia o primeiro sensor que é de temperatura)
    temp_sensors = [s for s in sensors_ok if s.startswith("TC") or s.startswith("T5")]
    ref_sensor = temp_sensors[0] if temp_sensors else sensors_ok[0]
    if cfg.EXCLUDE_STARTUP_MINUTES > 0:
        off_thr = float(df_all[ref_sensor].quantile(cfg.OFF_VALUE_QUANTILE))
        startup_excl = build_startup_exclusion_mask(
            common_index, df_all[ref_sensor], off_thr, cfg.EXCLUDE_STARTUP_MINUTES
        )
        exclude = exclude | startup_excl

    df_normal = df_all.loc[~exclude].copy()
    print(f"[MULTI] Dados normais (treino): {len(df_normal):,} pts "
          f"({100*len(df_normal)/max(len(df_all),1):.1f}% do total)")

    if len(df_normal) < cfg.TIME_STEPS + 10:
        return {"skipped": True, "reason": "poucos dados normais após exclusão"}

    # ------------------------------------------------------------------
    # 4. Normalização por sensor (z-score) — fit APENAS na porção de treino
    #    (cronológica) para não vazar a validação na média/desvio.
    # ------------------------------------------------------------------
    n_norm = len(df_normal)
    n_val_rows = int(np.floor(cfg.VAL_FRAC * n_norm))
    df_fit = df_normal.iloc[: n_norm - n_val_rows] if n_val_rows > 0 else df_normal
    center = df_fit.mean(axis=0)
    scale = df_fit.std(axis=0).replace(0, 1.0)
    df_normal_z = (df_normal - center) / scale
    df_all_z = (df_all - center) / scale

    values_normal = df_normal_z.values.astype(np.float32)
    values_all    = df_all_z.values.astype(np.float32)
    n_features = len(sensors_ok)   # 1 canal por sensor

    # ------------------------------------------------------------------
    # 5. Sequências
    # ------------------------------------------------------------------
    x_train_full = make_sequences(values_normal, cfg.TIME_STEPS, cfg.STRIDE)
    x_all        = make_sequences(values_all,    cfg.TIME_STEPS, cfg.STRIDE)

    x_train, x_val = train_val_split(
        x_train_full, cfg.VAL_FRAC, cfg.SHUFFLE_TRAIN, cfg.RANDOM_SEED,
        split_mode=cfg.SPLIT_MODE,
    )

    # Conjunto ANÔMALO p/ diagnóstico do AE: sequências cujo centro cai numa
    # janela de alarme (o AE deve reconstruí-las PIOR que as normais).
    alarm_pts = np.asarray(alarm_exclude).astype(bool)
    seq_centers = np.arange(x_all.shape[0]) * max(1, int(cfg.STRIDE)) + cfg.TIME_STEPS // 2
    seq_centers = np.clip(seq_centers, 0, len(alarm_pts) - 1)
    x_anom = x_all[alarm_pts[seq_centers]]
    print(f"[MULTI] x_train={x_train.shape} | x_val={x_val.shape} | "
          f"x_all={x_all.shape} | x_anom={x_anom.shape} | n_features={n_features}")

    # ------------------------------------------------------------------
    # 6. Treino — backend depende de cfg.MODEL_MODE
    # ------------------------------------------------------------------
    val_mse_normal = None
    val_mse_anomaly = None
    separation_ratio = None
    if cfg.MODEL_MODE == "per_sensor":
        from .per_sensor import train_per_sensor
        print(f"[MULTI] MODEL_MODE=per_sensor → treinando {n_features} AEs univariados "
              f"(combinacao MAX entre sensores)")
        psr = train_per_sensor(cfg, x_train, x_val, x_train_full, x_all, sensors_ok)
        # signals que substituem os do backend multivariado
        train_mae_seq = psr["train_combined_mae_seq"]
        mae_seq_all = psr["combined_mae_seq"]
        mae_per_sensor_seq = psr["mae_per_sensor_seq"]
        # placeholders para artefatos que so existem no caminho multivariado
        best_model = None
        history = None
        # salva cada modelo univariado em best_model/
        for sensor, model in psr["models"].items():
            mp = os.path.join(out_dirs["best_model"], f"model_{sensor}.keras")
            try:
                model.save(mp)
            except Exception as exc:
                print(f"[WARN] save per_sensor model {sensor}: {exc}")
        with open(os.path.join(out_dirs["best_model"], "per_sensor_config.json"), "w") as f:
            json.dump({
                "f1": cfg.PER_SENSOR_F1, "f2": cfg.PER_SENSOR_F2,
                "s1": cfg.PER_SENSOR_S1, "s2": cfg.PER_SENSOR_S2,
                "epochs": cfg.PER_SENSOR_EPOCHS, "sensors": sensors_ok,
                "aggregation": "MAX",
            }, f, indent=2, ensure_ascii=False)
    else:
        # backend multivariado (default): 1 AE com N canais
        best_hp, best_model, df_trials = run_tuner(cfg, out_dirs, x_train, x_val, n_features)
        df_trials.to_csv(os.path.join(out_dirs["csv"], "trials_ranking.csv"), index=False)

        with open(os.path.join(out_dirs["best_model"], "best_hyperparameters.json"), "w") as f:
            json.dump(best_hp.values, f, indent=2, ensure_ascii=False)

        history = refit_best_model(cfg, best_model, x_train, x_val, x_anom=x_anom)
        model_path = os.path.join(out_dirs["best_model"], "model.keras")
        best_model.save(model_path)

        plot_loss(history, os.path.join(out_dirs["figs"], "loss_curve.png"))

        # ------------------------------------------------------------------
        # 6b. Diagnóstico do AE: MSE de reconstrução normal (val) vs anomalia
        # ------------------------------------------------------------------
        mse_val_normal = _mse_per_seq(best_model, x_val, cfg.BATCH_SIZE)
        mse_anom = _mse_per_seq(best_model, x_anom, cfg.BATCH_SIZE)
        val_mse_normal = float(np.mean(mse_val_normal)) if len(mse_val_normal) else None
        val_mse_anomaly = float(np.mean(mse_anom)) if len(mse_anom) else None
        separation_ratio = (
            float(val_mse_anomaly / val_mse_normal)
            if (val_mse_normal and val_mse_anomaly) else None
        )
        print(f"[AE-DIAG] val_mse_normal={val_mse_normal} | val_mse_anomalia={val_mse_anomaly} "
              f"| separacao(anom/normal)={separation_ratio}")

        # ------------------------------------------------------------------
        # 7. Scoring: MAE geral + por sensor (backend multivariado)
        # ------------------------------------------------------------------
        train_mae_seq = reconstruction_mae_per_seq(best_model, x_train_full, cfg.BATCH_SIZE)
        mae_seq_all, mae_per_sensor_seq = reconstruction_mae_and_per_sensor(
            best_model, x_all, cfg.BATCH_SIZE
        )  # (n_seq,), (n_seq, n_sensors)

    # ------------------------------------------------------------------
    # 8. Threshold — usando TODOS os alarmes
    # ------------------------------------------------------------------
    _eval_win = int(cfg.EVAL_WINDOW_MINUTES) if cfg.EVAL_WINDOW_MINUTES else cfg.EXCLUDE_MINUTES_AROUND_ALARM

    if cfg.THRESH_MODE.lower() == "alarm_f2":
        threshold = compute_threshold_alarm_optimized(
            mae_seq=mae_seq_all,
            index=df_all_z.index,
            alarm_times=all_alarm_times,
            time_steps=cfg.TIME_STEPS,
            eval_window_minutes=_eval_win,
            target_recall=cfg.ALARM_F2_TARGET_RECALL,
            max_fp_per_day=cfg.ALARM_F2_MAX_FP_PER_DAY,
            incident_gap_hours=cfg.ALARM_F2_INCIDENT_GAP_HOURS,
            stride=cfg.STRIDE,
        )
    else:
        threshold = compute_threshold(train_mae_seq, cfg.THRESH_MODE, cfg.TARGET_ANOMALY_RATE)

    plot_hist_mae(train_mae_seq, threshold, os.path.join(out_dirs["figs"], "train_mae_hist.png"))

    # Histograma normal (val) vs anomalia (janelas de alarme), em MAE (mesma
    # unidade do threshold) — mostra a separação que de fato importa num AE.
    # Em per_sensor mode best_model é None; pula esse diagnostico (legado do multi).
    if best_model is not None:
        mae_val_normal = reconstruction_mae_per_seq(best_model, x_val, cfg.BATCH_SIZE)
        mae_anom_seq = reconstruction_mae_per_seq(best_model, x_anom, cfg.BATCH_SIZE)
        plot_hist_normal_vs_anomaly(
            mae_val_normal, mae_anom_seq, threshold,
            os.path.join(out_dirs["figs"], "hist_normal_vs_anomaly.png"),
        )

    # Threshold adaptativo mensal
    monthly_thresholds: dict = {}
    if cfg.ADAPTIVE_THRESHOLD_MODE.lower() == "monthly":
        anomaly_seq, monthly_thresholds = apply_adaptive_monthly_threshold(
            mae_seq=mae_seq_all,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            alarm_times=all_alarm_times,
            eval_window_minutes=_eval_win,
            percentile=cfg.ADAPTIVE_THRESHOLD_PERCENTILE,
            stride=cfg.STRIDE,
        )
        print(f"[ADAPTIVE] Thresholds mensais: { {k: f'{v:.5f}' for k,v in monthly_thresholds.items()} }")
    else:
        anomaly_seq = mae_seq_all > threshold

    # ------------------------------------------------------------------
    # 8b. Máscara de operação (running_a, proxy de NGP_A)
    #     Zera anomalias fora de operação e no buffer de transição liga/desliga,
    #     eliminando os falsos positivos de desligamento normal da turbina.
    # ------------------------------------------------------------------
    if cfg.ENABLE_OPERATIONAL_MASK and running is not None:
        op_state = build_operational_state_from_running(
            common_index, running,
            buffer_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            asymmetric=cfg.OPERATIONAL_MASK_ASYMMETRIC,
        )
        n_before = int(np.asarray(anomaly_seq).sum())
        anomaly_seq = mask_anomaly_seq_by_operational_state(
            anomaly_seq=anomaly_seq, index=common_index, time_steps=cfg.TIME_STEPS,
            state=op_state, stride=cfg.STRIDE,
        )
        n_after = int(np.asarray(anomaly_seq).sum())
        print(f"[OP-MASK] {cfg.RUNNING_COL} (proxy NGP_A) buffer={cfg.TRANSIENT_PADDING_MINUTES}min: "
              f"anomalias {n_before} -> {n_after} (removidas {n_before - n_after})")

    # ------------------------------------------------------------------
    # 8c. Camada preditiva: health-index EWMA + curva lead-time × FA/dia
    #     Saída oficial robusta a variância de tuning (substitui hit_rate@thr_único).
    # ------------------------------------------------------------------
    predictive_summary: dict = {}
    op_24 = None
    # tier_anomaly_seqs precisa estar acessivel apos o bloco predictive
    # (para mapeamento sequencia->ponto e inclusao no df_point)
    tier_anomaly_seqs: dict = {}
    if cfg.ENABLE_PREDICTIVE_LAYER:
        incidents = extract_incidents(
            df_alarm,
            priorities=cfg.PREDICTIVE_INCIDENT_PRIORITY,
            incident_gap_hours=cfg.ALARM_F2_INCIDENT_GAP_HOURS,
        )
        # Filtra ao range dos dados dos sensores: incidentes fora dessa janela
        # nao podem ser detectados (sem dados), entao nao devem contar como "perdidos".
        if len(incidents):
            t_min, t_max = common_index.min(), common_index.max()
            n_total = len(incidents)
            incidents = incidents[(incidents >= t_min) & (incidents <= t_max)]
            print(f"[PRED] incidentes no range dos dados: {len(incidents)} (de {n_total} totais)")
        # timestamps do fim de cada sequência + fração de operação na janela
        seq_starts = np.arange(len(mae_seq_all)) * max(1, int(cfg.STRIDE))
        seq_ends_pos = np.clip(seq_starts + cfg.TIME_STEPS - 1, 0, len(common_index) - 1)
        seq_end_times = common_index[seq_ends_pos]
        seq_end_seconds = pd.DatetimeIndex(seq_end_times).values.astype("datetime64[s]").astype("int64")
        if running is not None:
            rv = np.asarray(running, dtype=float)
            seq_run_frac = np.array([rv[s:s + cfg.TIME_STEPS].mean() for s in seq_starts])
        else:
            seq_run_frac = np.ones(len(mae_seq_all))
        seq_run_full = seq_run_frac >= 0.999
        dt_seconds = max(1, int(cfg.STRIDE)) * 30.0  # amostragem 30s/pt
        health_ewma = compute_health_index_ewma(
            mae_seq_all, seq_run_frac,
            half_life_hours=cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS,
            dt_seconds=dt_seconds,
        )
        # No MODE per_sensor, computa EWMA por sensor pra ter o sinal multi-canal
        # necessario pro aggregator OR-de-quantile (vs MAX que so usa 1-2 sensores dominantes).
        per_sensor_health_matrix = None
        if cfg.MODEL_MODE == "per_sensor":
            n_seq_ps, n_sens_ps = mae_per_sensor_seq.shape
            per_sensor_health_matrix = np.empty_like(mae_per_sensor_seq, dtype=np.float32)
            _hl_over = cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS_PER_SENSOR or {}
            for j in range(n_sens_ps):
                _sensor_j = sensors_ok[j] if j < len(sensors_ok) else None
                _hl = _hl_over.get(_sensor_j, cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS)
                per_sensor_health_matrix[:, j] = compute_health_index_ewma(
                    mae_per_sensor_seq[:, j], seq_run_frac,
                    half_life_hours=_hl,
                    dt_seconds=dt_seconds,
                )
            print(f"[PRED] per_sensor EWMA matrix={per_sensor_health_matrix.shape} "
                  f"(curva via OR-de-quantile)")

        # === Sistema tier de alertas (warn / alarm / critical) ===
        # Reutiliza per_sensor_health_matrix. Para cada tier: threshold por sensor
        # no quantile q e alerta se >= k sensores acima do seu proprio threshold.
        # Aplica a MESMA mascara operacional do anomaly_seq principal.
        if cfg.ENABLE_TIER_ALERTS and per_sensor_health_matrix is not None:
            n_sens_t = per_sensor_health_matrix.shape[1]
            tiers_cfg = [
                ("warn", float(cfg.WARN_Q), int(cfg.WARN_K)),
                ("alarm", float(cfg.ALARM_Q), int(cfg.ALARM_K)),
                ("critical", float(cfg.CRITICAL_Q), int(cfg.CRITICAL_K)),
            ]
            for tier_name, q, k in tiers_cfg:
                thr_q = np.empty(n_sens_t, dtype=float)
                for j in range(n_sens_t):
                    valid_health = per_sensor_health_matrix[seq_run_full, j]
                    thr_q[j] = (float(np.quantile(valid_health, q))
                                if valid_health.size else float("inf"))
                above_tier = (per_sensor_health_matrix >= thr_q[None, :])
                n_above_tier = above_tier.sum(axis=1)
                seq_tier = (n_above_tier >= k)
                # mascara operacional (se ativa): reusa op_state ja construido na etapa 8b
                if cfg.ENABLE_OPERATIONAL_MASK and running is not None:
                    seq_tier = mask_anomaly_seq_by_operational_state(
                        anomaly_seq=seq_tier, index=common_index,
                        time_steps=cfg.TIME_STEPS, state=op_state, stride=cfg.STRIDE,
                    )
                tier_anomaly_seqs[tier_name] = seq_tier
                n_alerts = int(np.asarray(seq_tier).sum())
                print(f"[TIER] {tier_name.upper():<8} q={q:.2f} k={k} -> {n_alerts} alertas seq")

        inc_seconds = (
            pd.DatetimeIndex(incidents).values.astype("datetime64[s]").astype("int64")
            if len(incidents) else np.array([], dtype="int64")
        )
        curves: dict = {}
        op_points: dict = {}
        for h in cfg.PREDICTIVE_HORIZONS_HOURS:
            if cfg.MODEL_MODE == "per_sensor" and per_sensor_health_matrix is not None:
                curve = compute_predictive_curve_per_sensor(
                    per_sensor_health=per_sensor_health_matrix,
                    seq_running_full=seq_run_full,
                    t_end_seconds=seq_end_seconds.astype(float),
                    incident_seconds=inc_seconds.astype(float),
                    horizon_hours=float(h),
                    debounce_hours=cfg.PREDICTIVE_ALERT_DEBOUNCE_HOURS,
                )
            else:
                curve = compute_predictive_curve(
                    health_ewma=health_ewma,
                    seq_running_full=seq_run_full,
                    t_end_seconds=seq_end_seconds.astype(float),
                    incident_seconds=inc_seconds.astype(float),
                    horizon_hours=float(h),
                    debounce_hours=cfg.PREDICTIVE_ALERT_DEBOUNCE_HOURS,
                )
            curves[float(h)] = curve
            op = pick_operating_point(curve, cfg.PREDICTIVE_FA_BUDGET_PER_DAY)
            op_points[float(h)] = op
            if curve is not None and len(curve):
                curve.to_csv(
                    os.path.join(out_dirs["csv"], f"predictive_curve_H{int(h)}h.csv"),
                    index=False,
                )
            if op:
                print(f"[PRED] H={int(h)}h | recall={op['recall']:.2f} "
                      f"fa/dia={op['fa_per_day']:.2f} lead={op['median_lead_hours']:.1f}h "
                      f"thr={op['threshold']:.4f} eps={int(op['n_episodes'])}")
                predictive_summary[f"H{int(h)}h"] = op
        op_24 = op_points.get(24.0)
        try:
            plot_predictive_curve(curves, op_points,
                                  os.path.join(out_dirs["figs"], "predictive_curve.png"))
            plot_health_index_over_time(
                seq_end_times, health_ewma,
                op_threshold=(op_24["threshold"] if op_24 else None),
                incident_times=incidents,
                out_path=os.path.join(out_dirs["figs"], "health_index_ewma.png"),
            )
        except Exception as exc:
            print(f"[WARN] plot preditivo falhou: {exc}")
        print(f"[PRED] incidentes genuinos usados: {len(incidents)} "
              f"(prioridade={cfg.PREDICTIVE_INCIDENT_PRIORITY})")

    # ------------------------------------------------------------------
    # 9. Mapeamento sequência → ponto
    # ------------------------------------------------------------------
    df_point = map_seq_to_point_anomalies(
        anomaly_seq, common_index, cfg.TIME_STEPS,
        cfg.POINT_RULE, cfg.POINT_WINDOW, cfg.POINT_MIN_COUNT,
        stride=cfg.STRIDE,
    )

    # 9b. Tiers de alerta como colunas adicionais no df_point
    # is_anom_point_warn / is_anom_point_alarm / is_anom_point_critical
    tier_summary: dict = {}
    if tier_anomaly_seqs:
        for tier_name, seq_tier in tier_anomaly_seqs.items():
            df_tier_point = map_seq_to_point_anomalies(
                seq_tier, common_index, cfg.TIME_STEPS,
                cfg.POINT_RULE, cfg.POINT_WINDOW, cfg.POINT_MIN_COUNT,
                stride=cfg.STRIDE,
            )
            col = f"is_anom_point_{tier_name}"
            df_point[col] = df_tier_point["is_anom_point"].reindex(df_point.index).fillna(0).astype(int)
            n_pts_tier = int(df_point[col].sum())
            tier_summary[tier_name] = {
                "n_seq_alerts": int(np.asarray(seq_tier).sum()),
                "n_point_alerts": n_pts_tier,
            }
            print(f"[TIER] {tier_name.upper():<8} → {n_pts_tier} pontos marcados em df_point")

    # ------------------------------------------------------------------
    # 10. Salva CSVs
    # ------------------------------------------------------------------
    df_seq_scores = build_sequence_scores_df(common_index, mae_seq_all, anomaly_seq, stride=cfg.STRIDE)

    # Adiciona coluna por sensor no CSV de sequências
    for i, sensor in enumerate(sensors_ok):
        df_seq_scores[f"mae_{sensor}"] = np.nan
        n = min(len(mae_per_sensor_seq), len(df_seq_scores))
        df_seq_scores.iloc[:n, df_seq_scores.columns.get_loc(f"mae_{sensor}")] = mae_per_sensor_seq[:n, i]

    df_seq_scores.to_csv(os.path.join(out_dirs["csv"], "sequence_scores_all.csv"), index=False)
    df_point.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    # ------------------------------------------------------------------
    # 11. Avaliação contra TODOS os alarmes (todos os sensores)
    # ------------------------------------------------------------------
    eval_stats = evaluate_alarm_detection(df_alarm, df_point, _eval_win)
    fp_per_day = compute_anomaly_rate_per_day(df_point[df_point["is_anom_point"] == 1])

    # Quais sensores mais contribuíram para as anomalias detectadas?
    anom_seq_idx = np.where(anomaly_seq)[0]
    if len(anom_seq_idx) > 0:
        mean_contrib = mae_per_sensor_seq[anom_seq_idx].mean(axis=0)
        sensor_contrib = dict(zip(sensors_ok, [float(v) for v in mean_contrib]))
        sensor_contrib_sorted = dict(sorted(sensor_contrib.items(), key=lambda x: -x[1]))
    else:
        sensor_contrib_sorted = {}

    # ------------------------------------------------------------------
    # 12. Plot série (usa o sensor de temperatura de referência)
    # ------------------------------------------------------------------
    if ref_sensor in df_raw.columns:
        ref_series = df_raw.set_index(cfg.TIME_COL)[ref_sensor]
    else:
        ref_series = df_all[ref_sensor]

    anomalous_times = df_point.index[df_point["is_anom_point"] == 1]
    # Alarmes do plot: por padrão só os da própria variável (Tag == ref_sensor),
    # evitando poluir a curva com alarmes de outros sensores/instrumentos.
    if "Data da Ocorrencia" in df_alarm.columns:
        if cfg.PLOT_ALARMS_PER_VARIABLE and "Tag" in df_alarm.columns:
            sel = df_alarm[df_alarm["Tag"].astype(str) == ref_sensor]
            alarm_times = sel["Data da Ocorrencia"]
            print(f"[PLOT] alarmes da variável {ref_sensor}: {len(alarm_times)} "
                  f"(de {len(df_alarm)} totais)")
        else:
            alarm_times = df_alarm["Data da Ocorrencia"]
    else:
        alarm_times = pd.Series(dtype="datetime64[ns]")
    try:
        plot_series_with_anomalies(
            ref_series,
            anomalous_times,
            os.path.join(out_dirs["figs"], f"series_{ref_sensor}_anomalies.png"),
            title=f"Serie + anomalias (multivariado) | sensor={ref_sensor}",
        )
        plot_series_alarm_anomaly_subplots(
            ref_series,
            anomalous_times,
            alarm_times,
            os.path.join(out_dirs["figs"], f"series_{ref_sensor}_alarm_anomaly_subplots.png"),
            title=f"{ref_sensor}",
        )
    except Exception as exc:
        print(f"[WARN] plot_series falhou: {exc}")

    # ------------------------------------------------------------------
    # 13. Relatório final
    # ------------------------------------------------------------------
    report = {
        "sensors": sensors_ok,
        "n_sensors": len(sensors_ok),
        "n_train_sequences": int(x_train.shape[0]),
        "n_val_sequences": int(x_val.shape[0]),
        "n_anomaly_sequences": int(x_anom.shape[0]),
        "n_all_sequences": int(x_all.shape[0]),
        "val_mse_normal": val_mse_normal,
        "val_mse_anomaly": val_mse_anomaly,
        "anomaly_normal_separation_ratio": separation_ratio,
        "threshold": float(threshold),
        "monthly_thresholds": {str(k): float(v) for k, v in monthly_thresholds.items()},
        "anomaly_rate_per_day": float(fp_per_day),
        "sensor_contribution_to_anomalies": sensor_contrib_sorted,
        "predictive_operating_points": predictive_summary,
        "tier_alerts": tier_summary,
        **eval_stats,
    }

    report_path = os.path.join(out_dirs["csv"], "multivariado_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[MULTI] Resultado:")
    print(f"  hit_rate          : {eval_stats.get('hit_rate')}")
    print(f"  n_alarms          : {eval_stats.get('n_alarms')}")
    print(f"  fp_per_day        : {fp_per_day:.2f}")
    print(f"  threshold         : {threshold:.6f}")
    print(f"  sensor_contrib    : {list(sensor_contrib_sorted.items())[:5]}")
    print(f"  report            : {report_path}")

    # Retorno compativel com _upload_run_artifacts (main.py): expoe o output_dir
    # para que csv/ (inclui multivariado_report.json) e figs/ subam como artefatos.
    return {
        "report": report,
        "sensor_outputs": [{"sensor": joint_label, "output_dir": out_dirs["root"]}],
    }
