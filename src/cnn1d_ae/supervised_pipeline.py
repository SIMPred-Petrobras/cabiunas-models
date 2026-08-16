from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config
from .preprocess import (
    build_sensor_dataframe,
    build_group_dataframe,
    build_exclusion_mask,
    clip_outliers,
    normalize_train_only,
    select_feature_columns,
)
from .scoring import (
    map_seq_to_point_anomalies,
    compute_anomaly_rate_per_day,
    build_operational_state,
    eval_alarm_hit_rate,
    compute_normal_alert_rate,
    compute_composite_score,
)
from .plots import plot_series_with_anomalies, plot_series_alarm_anomaly_subplots

_DEFAULT_PROBA_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_DEFAULT_DEBOUNCE_GRID = [1]


def _build_prealarm_labels(
    index: pd.DatetimeIndex, alarm_times: pd.Series, horizon_hours: float, post_exclude_minutes: int
) -> Tuple[pd.Series, pd.Series]:
    """Rotulo supervisionado: y=1 se existe algum alarme em `alarm_times`
    dentro das proximas `horizon_hours` a partir daquele instante -- ou
    seja, X(t) usa so leituras ate t, y(t) e um desfecho *futuro*, o
    contrario do erro de reconstrucao nao-supervisionado (que so aprende
    "normal" e espera que o erro suba perto de uma falha).

    `exclude_post`: janela logo apos cada alarme (mesma duracao de
    EXCLUDE_MINUTES_AROUND_ALARM) -- nao e nem "vai alarmar em breve" nem
    um "normal" limpo (e recuperacao/transiente pos-evento), entao fica de
    fora tanto do treino quanto da avaliacao."""
    horizon = pd.Timedelta(hours=float(horizon_hours))
    post_excl = pd.Timedelta(minutes=int(post_exclude_minutes))
    y = pd.Series(0, index=index, dtype=int)
    exclude_post = pd.Series(False, index=index)
    for t in alarm_times.values:
        t = pd.Timestamp(t)
        y.loc[(index >= t - horizon) & (index < t)] = 1
        exclude_post.loc[(index >= t) & (index <= t + post_excl)] = True
    return y, exclude_post


def run_supervised_group(
    cfg: PipelineConfig,
    df_alarm: pd.DataFrame,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    group: Dict[str, Any],
) -> Dict[str, Any]:
    """Classificador supervisionado de alerta precoce (EXP7 item 4): em vez
    de aprender so "normal" e esperar que o erro de reconstrucao suba perto
    de uma falha (AutoML nao-supervisionado, ver automl_pipeline.py), treina
    um RandomForestClassifier para prever diretamente "existe um alarme nas
    proximas PREDICTION_HORIZON_HOURS?" usando os alarmes reais como rotulo.
    Mesma engenharia de features do EXP7 (select_feature_columns), mesmo
    split OOS, mesma avaliacao (eval_alarm_hit_rate/composite_score) --
    so muda o que e otimizado: antecedencia diretamente, nao reconstrucao.
    Ver docs/analise_automl_exp7_planejamento.md (item 4)."""
    group_name = group["name"]
    sensors = list(group["sensors"])
    eval_sensors = list(group.get("eval_sensors") or sensors)

    out_dirs = ensure_sensor_dirs(cfg, group_name)
    save_run_config(cfg, out_dirs)
    with open(os.path.join(out_dirs["csv"], "group_definition.json"), "w", encoding="utf-8") as f:
        json.dump(group, f, indent=2, ensure_ascii=False)

    df_use, long_gap_mask = build_group_dataframe(cfg, df_feat, df_raw, sensors)

    valid_sensors = [s for s in sensors if float(df_use[s].std()) >= cfg.MIN_STD]
    dropped = set(sensors) - set(valid_sensors)
    if dropped:
        print(f"[WARN] supervised group={group_name}: sensores com std baixo removidos: {dropped}")
    if not valid_sensors:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "all_sensors_low_std"}
    sensors = valid_sensors

    feature_cols = select_feature_columns(cfg, df_use, sensors)
    df_use = df_use[feature_cols]

    state = None
    if cfg.ENABLE_OPERATIONAL_MASK:
        ref_sensor = cfg.OPERATIONAL_REF_SENSOR
        if ref_sensor and ref_sensor not in sensors:
            df_ref, _ = build_sensor_dataframe(cfg, df_feat, df_raw, ref_sensor)
            ref_series = df_ref[ref_sensor]
        else:
            ref_col = ref_sensor if (ref_sensor and ref_sensor in sensors) else sensors[0]
            ref_series = df_use[ref_col]
        state = build_operational_state(
            index=df_use.index, sensor_series=ref_series,
            off_value_quantile=cfg.OFF_VALUE_QUANTILE, off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
            off_long_min_hours=cfg.OFF_LONG_MIN_HOURS, transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        )

    if "Tag" in df_alarm.columns:
        df_alarm_group = df_alarm.loc[df_alarm["Tag"].isin(eval_sensors)].copy()
    else:
        df_alarm_group = df_alarm.copy()
    if "Data da Ocorrencia" in df_alarm_group.columns:
        df_alarm_group = df_alarm_group.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
    alarm_times = (
        df_alarm_group["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_group.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    if len(alarm_times) < 5:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "poucos_alarmes_para_supervisionado"}

    oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE) if cfg.AUTOML_OOS_SPLIT_DATE else None
    if oos_start is None:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "supervisionado_exige_AUTOML_OOS_SPLIT_DATE"}

    y_label, exclude_post = _build_prealarm_labels(
        df_use.index, alarm_times, cfg.PREDICTION_HORIZON_HOURS, cfg.EXCLUDE_MINUTES_AROUND_ALARM,
    )

    long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
    exclude = exclude_post.copy()
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        exclude = exclude | long_gap_mask
    if state is not None:
        exclude = exclude | (state != "on")

    df_use = clip_outliers(df_use, cfg)

    fit_mask = (~exclude) & (df_use.index < oos_start)
    df_fit = df_use.loc[fit_mask]
    y_fit = y_label.loc[fit_mask]
    if len(df_fit) < 50 or y_fit.sum() < 5:
        return {"group": group_name, "sensors": sensors, "skipped": True,
                "reason": "poucos_pontos_de_treino_ou_poucos_positivos"}

    df_fit_z, df_all_z, _, _ = normalize_train_only(cfg, df_fit, df_use)

    x_train = df_fit_z.values.astype(np.float32)
    y_train = y_fit.values.astype(np.int32)
    x_all = df_all_z.values.astype(np.float32)
    all_index = df_all_z.index

    max_train = cfg.SUPERVISED_MAX_TRAIN_SAMPLES
    if max_train and len(x_train) > max_train:
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_neg_keep = max(0, int(max_train) - len(pos_idx))
        if n_neg_keep < len(neg_idx):
            rng = np.random.default_rng(cfg.RANDOM_SEED)
            neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
        x_train = x_train[keep_idx]
        y_train = y_train[keep_idx]

    print(f"[SUPERVISED] group={group_name} treinando RandomForest "
          f"(n_train={len(x_train)}, positivos={int(y_train.sum())})...")
    clf = RandomForestClassifier(
        n_estimators=cfg.SUPERVISED_N_ESTIMATORS,
        max_depth=cfg.SUPERVISED_MAX_DEPTH,
        min_samples_leaf=cfg.SUPERVISED_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    proba_all = clf.predict_proba(x_all)[:, 1]

    eval_mask = pd.Series(all_index >= oos_start, index=all_index)
    if state is not None:
        state = state.reindex(all_index)
    near_alarm_mask = build_exclusion_mask(all_index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start]
    df_point_eval_idx = all_index[eval_mask.values]

    thresholds = cfg.SUPERVISED_PROBA_THRESHOLDS or _DEFAULT_PROBA_THRESHOLDS
    debounces = cfg.AUTOML_DEBOUNCE_GRID or _DEFAULT_DEBOUNCE_GRID

    trials: List[Dict[str, Any]] = []
    best_trial: Dict[str, Any] | None = None
    best_point_df: pd.DataFrame | None = None

    for th in thresholds:
        anomaly_flags = (proba_all >= th).astype(int)
        for debounce in debounces:
            df_point = map_seq_to_point_anomalies(
                anomaly_flags, all_index, time_steps=1,
                point_rule="all_of_window", point_window=int(debounce), point_min_count=int(debounce),
            )
            if state is not None:
                df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
                df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0

            eval_stats = eval_alarm_hit_rate(df_alarm_eval, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
            normal_rate = compute_normal_alert_rate(
                df_point.loc[df_point_eval_idx], near_alarm_mask.loc[df_point_eval_idx]
            )
            score = compute_composite_score(
                detection_rate=eval_stats["hit_rate"] or 0.0,
                normal_alert_rate=normal_rate,
                fp_penalty=cfg.AUTOML_FP_PENALTY,
                min_detection_rate=cfg.AUTOML_MIN_DETECTION_RATE,
            )
            trial = {
                "proba_threshold": th,
                "debounce": int(debounce),
                "anomaly_rate_points_per_day": compute_anomaly_rate_per_day(df_point.loc[df_point_eval_idx]),
                **eval_stats,
                **score,
            }
            trials.append(trial)
            if best_trial is None or trial["composite_score"] > best_trial["composite_score"]:
                best_trial = trial
                best_point_df = df_point

    if best_trial is None:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "no_valid_trials"}

    df_ranking = pd.DataFrame(trials).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_ranking.to_csv(os.path.join(out_dirs["csv"], "supervised_ranking.csv"), index=False)

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv(os.path.join(out_dirs["csv"], "feature_importances.csv"), header=["importance"])

    model_path = os.path.join(out_dirs["best_model"], "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(out_dirs["best_model"], "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in best_trial.items()}, f, indent=2, ensure_ascii=False)

    assert best_point_df is not None
    best_point_df.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    anomalous_times = best_point_df.index[best_point_df["is_anom_point"] == 1]
    for s in sensors:
        safe_name = s.replace("/", "_").replace("\\", "_")
        if "Tag" in df_alarm.columns and "Data da Ocorrencia" in df_alarm.columns:
            s_alarm_times = df_alarm.loc[df_alarm["Tag"] == s, "Data da Ocorrencia"].dropna()
        elif "Data da Ocorrencia" in df_alarm.columns:
            s_alarm_times = df_alarm["Data da Ocorrencia"].dropna()
        else:
            s_alarm_times = pd.Series(dtype="datetime64[ns]")

        plot_series_with_anomalies(
            df_use[s], anomalous_times,
            os.path.join(out_dirs["figs"], f"series_with_anomalies_{safe_name}.png"),
            title=f"Serie + anomalias (Supervisionado/RF) | grupo={group_name} | sensor={s}",
            operational_state=state,
        )
        plot_series_alarm_anomaly_subplots(
            df_use[s], anomalous_times, s_alarm_times,
            os.path.join(out_dirs["figs"], f"series_alarm_anomaly_subplots_{safe_name}.png"),
            title=f"{group_name} | {s} | Supervisionado/RF",
            operational_state=state,
        )

    calibration_report = {
        "group": group_name,
        "sensors": sensors,
        "eval_sensors": eval_sensors,
        "n_sensors": len(sensors),
        "best_model": "random_forest_supervised",
        "prediction_horizon_hours": cfg.PREDICTION_HORIZON_HOURS,
        "proba_threshold": float(best_trial["proba_threshold"]),
        "debounce": int(best_trial["debounce"]),
        "n_trials": len(trials),
        "anomaly_rate_points_per_day": best_trial["anomaly_rate_points_per_day"],
        "operational_mask_enabled": bool(cfg.ENABLE_OPERATIONAL_MASK),
        "operational_ref_sensor": cfg.OPERATIONAL_REF_SENSOR,
        "oos_split_date": cfg.AUTOML_OOS_SPLIT_DATE,
        "oos_validated": True,
        "n_train_points": int(len(x_train)),
        "n_train_positive": int(y_train.sum()),
        "top_10_features": importances.head(10).to_dict(),
        **{k: best_trial[k] for k in ("n_alarms", "alarms_with_detected_anomaly_in_window", "hit_rate",
                                       "composite_score", "balanced_score", "detection_rate", "normal_alert_rate")},
    }
    if state is not None:
        calibration_report["operational_state_counts"] = {
            str(k): int(v) for k, v in state.value_counts().to_dict().items()
        }
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    report = {
        "group": group_name,
        "sensors": sensors,
        "output_dir": out_dirs["root"],
        "model_path": model_path,
        "best_model": "random_forest_supervised",
        "threshold": float(best_trial["proba_threshold"]),
        "n_alarms": best_trial["n_alarms"],
        "alarms_with_detected_anomaly_in_window": best_trial["alarms_with_detected_anomaly_in_window"],
        "hit_rate": best_trial["hit_rate"],
        "composite_score": best_trial["composite_score"],
        "skipped": False,
    }
    with open(os.path.join(out_dirs["csv"], "evaluation_alarm_hit_rate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
