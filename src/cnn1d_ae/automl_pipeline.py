from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config
from .model import build_callbacks
from .preprocess import (
    build_sensor_dataframe,
    build_group_dataframe,
    build_exclusion_mask,
    clip_outliers,
    normalize_train_only,
    TEXTURE_MIN_WINDOW,
)
from .scoring import (
    map_seq_to_point_anomalies,
    compute_anomaly_rate_per_day,
    build_operational_state,
    eval_alarm_hit_rate,
    compute_normal_alert_rate,
    compute_composite_score,
)
from .automl_models import (
    build_dense_autoencoder,
    dense_reconstruction_error,
    fit_ocsvm,
    ocsvm_error,
    fit_isolation_forest,
    isolation_forest_error,
)
from .plots import plot_series_with_anomalies, plot_series_alarm_anomaly_subplots

_DEFAULT_MODELS = ["dense", "ocsvm", "iforest"]
_DEFAULT_PERCENTILES = [95.0, 97.0, 99.0, 99.5]
_DEFAULT_DEBOUNCE_GRID = [1]
_DEFAULT_DENSE_LAYERS = [256, 128]


def _fit_score_dense(cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, n_features: int):
    layers = cfg.AUTOML_DENSE_LAYERS or _DEFAULT_DENSE_LAYERS
    model = build_dense_autoencoder(n_features, layers, cfg.AUTOML_DENSE_DROPOUT, cfg.AUTOML_DENSE_LR)

    n_val = max(1, int(len(x_normal) * cfg.VAL_FRAC))
    x_tr, x_val = x_normal[:-n_val], x_normal[-n_val:]
    model.fit(
        x_tr, x_tr,
        validation_data=(x_val, x_val),
        epochs=cfg.AUTOML_DENSE_EPOCHS,
        batch_size=cfg.AUTOML_DENSE_BATCH_SIZE,
        callbacks=build_callbacks(cfg.AUTOML_DENSE_PATIENCE),
        verbose=0,
    )
    train_err = dense_reconstruction_error(model, x_normal, cfg.AUTOML_DENSE_BATCH_SIZE)
    all_err = dense_reconstruction_error(model, x_all, cfg.AUTOML_DENSE_BATCH_SIZE)
    return train_err, all_err, model, {"dense_layers": layers, "dropout": cfg.AUTOML_DENSE_DROPOUT,
                                        "lr": cfg.AUTOML_DENSE_LR, "epochs": cfg.AUTOML_DENSE_EPOCHS,
                                        "batch_size": cfg.AUTOML_DENSE_BATCH_SIZE}


def _fit_score_ocsvm(cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, n_features: int):
    # OneClassSVM (kernel RBF) escala ~O(n^2)-O(n^3) no numero de amostras de
    # treino -- com centenas de milhares de pontos (datasets maiores/janelas
    # de fit mais longas) o ajuste fica impraticavel. Subamostra so o *fit*;
    # o score (train_err/all_err) continua sendo calculado sobre os dados
    # inteiros, que e barato (so avalia contra os vetores de suporte).
    x_fit = x_normal
    max_train = cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES
    if max_train and len(x_normal) > max_train:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(len(x_normal), size=int(max_train), replace=False)
        x_fit = x_normal[idx]
    clf = fit_ocsvm(x_fit, cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA)
    train_err = ocsvm_error(clf, x_normal)
    all_err = ocsvm_error(clf, x_all)
    return train_err, all_err, clf, {"nu": cfg.AUTOML_OCSVM_NU, "gamma": cfg.AUTOML_OCSVM_GAMMA,
                                      "train_samples": int(len(x_fit))}


def _fit_score_iforest(cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, n_features: int):
    model = fit_isolation_forest(
        x_normal, cfg.AUTOML_IFOREST_CONTAMINATION, cfg.AUTOML_IFOREST_N_ESTIMATORS, cfg.RANDOM_SEED
    )
    train_err = isolation_forest_error(model, x_normal)
    all_err = isolation_forest_error(model, x_all)
    return train_err, all_err, model, {"contamination": cfg.AUTOML_IFOREST_CONTAMINATION,
                                        "n_estimators": cfg.AUTOML_IFOREST_N_ESTIMATORS}


_FITTERS = {"dense": _fit_score_dense, "ocsvm": _fit_score_ocsvm, "iforest": _fit_score_iforest}


def _seed_sweep_iforest(
    cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, all_index: pd.Index,
    state: pd.Series | None, df_alarm_eval: pd.DataFrame, df_point_eval_idx: pd.Index,
    near_alarm_mask: pd.Series, pct: float, debounce: int, n_seeds: int,
) -> List[Dict[str, Any]]:
    """Re-treina o mesmo iforest (mesmo threshold_percentile/debounce do
    melhor trial) com N seeds extras, pra medir o quanto hit_rate/
    normal_alert_rate variam so por causa da aleatoriedade da floresta —
    ver analise_automl_lara.md secao 2 (~+-27pp de ruido de semente na
    pipeline da Lara)."""
    results = []
    for i in range(1, n_seeds + 1):
        seed = cfg.RANDOM_SEED + i
        model = fit_isolation_forest(x_normal, cfg.AUTOML_IFOREST_CONTAMINATION, cfg.AUTOML_IFOREST_N_ESTIMATORS, seed)
        train_err = isolation_forest_error(model, x_normal)
        all_err = isolation_forest_error(model, x_all)
        threshold = float(np.percentile(train_err, pct))
        anomaly_flags = (all_err > threshold).astype(int)
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
        results.append({"seed": seed, "hit_rate": eval_stats["hit_rate"], "normal_alert_rate": normal_rate})
    return results


def _save_model(model_type: str, model_obj: Any, out_dirs: Dict[str, str]) -> str:
    if model_type == "dense":
        path = os.path.join(out_dirs["best_model"], "model.keras")
        model_obj.save(path)
        return path
    path = os.path.join(out_dirs["best_model"], "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    return path


def run_automl_group(
    cfg: PipelineConfig,
    df_alarm: pd.DataFrame,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    group: Dict[str, Any],
) -> Dict[str, Any]:
    """Busca AutoML (dense/ocsvm/iforest x threshold_percentile x debounce) para
    um grupo de sensores, ranqueada por composite_score sobre a nossa propria
    avaliacao de alarme (eval_alarm_hit_rate) — nao a janela de 30 dias/eventos
    nao curados da pipeline original da Lara. Ver analise_automl_lara.md.

    Diferente do CNN-1D (sequencial), os modelos aqui operam ponto-a-ponto:
    cada instante vira uma amostra de treino, sem janelamento temporal.
    """
    group_name = group["name"]
    sensors = list(group["sensors"])
    target_sensor = group.get("target_sensor")
    # eval_sensors: subconjunto de `sensors` cujos alarmes contam na
    # avaliacao (hit_rate/normal_alert_rate/composite_score). Default =
    # todos os `sensors`. Util quando alguns sensores do grupo (ex: canais
    # de vibracao) entram so como feature preditiva, sem que seus proprios
    # alarmes participem do denominador/numerador da avaliacao.
    eval_sensors = list(group.get("eval_sensors") or sensors)

    out_dirs = ensure_sensor_dirs(cfg, group_name)
    save_run_config(cfg, out_dirs)
    with open(os.path.join(out_dirs["csv"], "group_definition.json"), "w", encoding="utf-8") as f:
        json.dump(group, f, indent=2, ensure_ascii=False)

    df_use, long_gap_mask = build_group_dataframe(cfg, df_feat, df_raw, sensors)

    valid_sensors = [s for s in sensors if float(df_use[s].std()) >= cfg.MIN_STD]
    dropped = set(sensors) - set(valid_sensors)
    if dropped:
        print(f"[WARN] automl group={group_name}: sensores com std baixo removidos: {dropped}")
    if not valid_sensors:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "all_sensors_low_std"}
    sensors = valid_sensors
    if target_sensor and target_sensor not in sensors:
        target_sensor = None

    feature_cols = list(sensors)
    if cfg.ENABLE_DERIVED_FEATURES:
        windows = list(cfg.DERIVED_ROLLING_WINDOWS) if cfg.DERIVED_ROLLING_WINDOWS else [cfg.DERIVED_ROLLING_WINDOW]
        suffixes = ["__delta_1"]
        for w in windows:
            w = max(2, int(w))
            suffixes += [f"__roll_med_{w}", f"__roll_std_{w}", f"__trend_{w}"]
            if w >= TEXTURE_MIN_WINDOW:
                suffixes += [f"__roll_kurt_{w}", f"__roll_skew_{w}", f"__crest_{w}"]
        for s in sensors:
            for suffix in suffixes:
                col = f"{s}{suffix}"
                if col in df_use.columns:
                    feature_cols.append(col)
    df_use = df_use[feature_cols]

    # Estado operacional via OPERATIONAL_REF_SENSOR (mesmo mecanismo do
    # CNN-1D). Calculado aqui (antes do fit) para poder excluir periodos
    # fora de "on" do proprio conjunto de treino -- nao so da avaliacao.
    # Sem isso, o modelo aprendia uma mistura de dois regimes bem diferentes
    # (operando vs. parado/frio), o que inflava hit_rate e normal_alert_rate
    # juntos (o "anomalo" virava, em boa parte, so a transicao liga/desliga).
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

    exclude_alarm = build_exclusion_mask(df_use.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    exclude = exclude_alarm.copy()
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
        exclude = exclude | long_gap_mask
    if state is not None:
        exclude = exclude | (state != "on")

    df_normal = df_use.loc[~exclude].copy()
    df_all = df_use.copy()
    if len(df_normal) < 10:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "few_normal_points"}

    df_normal = clip_outliers(df_normal, cfg)
    df_all = clip_outliers(df_all, cfg)

    # Split OOS: se definido, treino (modelo + normalizacao + percentil de
    # threshold) so enxerga dados anteriores a AUTOML_OOS_SPLIT_DATE; a
    # avaliacao (hit_rate/normal_alert_rate/composite_score) so considera o
    # periodo posterior — alarmes e pontos que o modelo nunca viu.
    oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE) if cfg.AUTOML_OOS_SPLIT_DATE else None
    df_normal_fit = df_normal.loc[df_normal.index < oos_start] if oos_start is not None else df_normal
    if len(df_normal_fit) < 10:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "few_normal_points_before_oos_split"}

    df_normal_z, df_all_z, _, _ = normalize_train_only(cfg, df_normal_fit, df_all)

    x_normal = df_normal_z.values.astype(np.float32)
    x_all = df_all_z.values.astype(np.float32)
    n_features = x_normal.shape[1]
    all_index = df_all_z.index
    if oos_start is not None:
        eval_mask = pd.Series(all_index >= oos_start, index=all_index)
    else:
        eval_mask = pd.Series(True, index=all_index)

    # `state` ja foi calculado mais acima (antes do fit) e tem o mesmo indice
    # de df_use == all_index (normalize_train_only preserva index/ordem).
    if state is not None:
        state = state.reindex(all_index)

    near_alarm_mask = build_exclusion_mask(all_index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)

    # Alarmes usados na avaliacao: so os do periodo OOS, se houver split.
    if oos_start is not None:
        df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start]
    else:
        df_alarm_eval = df_alarm_group
    df_point_eval_idx = all_index[eval_mask.values]

    model_types = cfg.AUTOML_MODELS or _DEFAULT_MODELS
    percentiles = cfg.AUTOML_THRESHOLD_PERCENTILES or _DEFAULT_PERCENTILES
    debounces = cfg.AUTOML_DEBOUNCE_GRID or _DEFAULT_DEBOUNCE_GRID

    trials: List[Dict[str, Any]] = []
    best_trial: Dict[str, Any] | None = None
    best_model_obj = None
    best_model_type = None
    best_point_df: pd.DataFrame | None = None

    for model_type in model_types:
        fitter = _FITTERS.get(model_type)
        if fitter is None:
            print(f"[WARN] automl group={group_name}: modelo desconhecido '{model_type}' ignorado")
            continue

        print(f"[AUTOML] group={group_name} model={model_type} — treinando...")
        train_err, all_err, model_obj, model_params = fitter(cfg, x_normal, x_all, n_features)

        for pct in percentiles:
            threshold = float(np.percentile(train_err, pct))
            anomaly_flags = (all_err > threshold).astype(int)

            for debounce in debounces:
                df_point = map_seq_to_point_anomalies(
                    anomaly_flags, all_index, time_steps=1,
                    point_rule="all_of_window", point_window=int(debounce), point_min_count=int(debounce),
                )
                if state is not None:
                    df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
                    df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0

                # Janela de match do hit_rate usa o df_point inteiro (um alarme
                # OOS perto do corte ainda pode casar com pontos um pouco antes
                # dele); o que garante a disciplina OOS e o modelo/threshold so
                # terem visto dados antes do corte, e so contar alarmes OOS.
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
                    "model": model_type,
                    "threshold_percentile": pct,
                    "debounce": int(debounce),
                    "threshold": threshold,
                    "anomaly_rate_points_per_day": compute_anomaly_rate_per_day(df_point.loc[df_point_eval_idx]),
                    **model_params,
                    **eval_stats,
                    **score,
                }
                trials.append(trial)

                if best_trial is None or trial["composite_score"] > best_trial["composite_score"]:
                    best_trial = trial
                    best_model_obj = model_obj
                    best_model_type = model_type
                    best_point_df = df_point

    if best_trial is None:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "no_valid_trials"}

    df_ranking = pd.DataFrame(trials).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_ranking.to_csv(os.path.join(out_dirs["csv"], "automl_ranking.csv"), index=False)

    seed_sweep = None
    if cfg.AUTOML_SEED_SWEEP_N and best_model_type == "iforest":
        extra = _seed_sweep_iforest(
            cfg, x_normal, x_all, all_index, state, df_alarm_eval, df_point_eval_idx, near_alarm_mask,
            best_trial["threshold_percentile"], best_trial["debounce"], cfg.AUTOML_SEED_SWEEP_N,
        )
        runs = [{"seed": cfg.RANDOM_SEED, "hit_rate": best_trial["hit_rate"],
                 "normal_alert_rate": best_trial["normal_alert_rate"]}] + extra
        hit_rates = [r["hit_rate"] for r in runs]
        normal_rates = [r["normal_alert_rate"] for r in runs]
        seed_sweep = {
            "runs": runs,
            "hit_rate_mean": float(np.mean(hit_rates)),
            "hit_rate_std": float(np.std(hit_rates)),
            "hit_rate_min": float(np.min(hit_rates)),
            "hit_rate_max": float(np.max(hit_rates)),
            "normal_alert_rate_mean": float(np.mean(normal_rates)),
            "normal_alert_rate_std": float(np.std(normal_rates)),
        }
        with open(os.path.join(out_dirs["csv"], "seed_sweep.json"), "w", encoding="utf-8") as f:
            json.dump(seed_sweep, f, indent=2, ensure_ascii=False)

    model_path = _save_model(best_model_type, best_model_obj, out_dirs)
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
            df_all[s], anomalous_times,
            os.path.join(out_dirs["figs"], f"series_with_anomalies_{safe_name}.png"),
            title=f"Serie + anomalias (AutoML/{best_model_type}) | grupo={group_name} | sensor={s}",
            operational_state=state,
        )
        plot_series_alarm_anomaly_subplots(
            df_all[s], anomalous_times, s_alarm_times,
            os.path.join(out_dirs["figs"], f"series_alarm_anomaly_subplots_{safe_name}.png"),
            title=f"{group_name} | {s} | AutoML/{best_model_type}",
            operational_state=state,
        )

    calibration_report = {
        "group": group_name,
        "sensors": sensors,
        "eval_sensors": eval_sensors,
        "target_sensor": target_sensor or "global",
        "n_sensors": len(sensors),
        "best_model": best_model_type,
        "threshold": float(best_trial["threshold"]),
        "threshold_percentile": float(best_trial["threshold_percentile"]),
        "debounce": int(best_trial["debounce"]),
        "n_trials": len(trials),
        "anomaly_rate_points_per_day": best_trial["anomaly_rate_points_per_day"],
        "operational_mask_enabled": bool(cfg.ENABLE_OPERATIONAL_MASK),
        "operational_ref_sensor": cfg.OPERATIONAL_REF_SENSOR,
        "oos_split_date": cfg.AUTOML_OOS_SPLIT_DATE,
        "oos_validated": oos_start is not None,
        "n_normal_points_used_for_fit": int(len(df_normal_fit)),
        **{k: best_trial[k] for k in ("n_alarms", "alarms_with_detected_anomaly_in_window", "hit_rate",
                                       "composite_score", "balanced_score", "detection_rate", "normal_alert_rate")},
    }
    if state is not None:
        calibration_report["operational_state_counts"] = {
            str(k): int(v) for k, v in state.value_counts().to_dict().items()
        }
    if seed_sweep is not None:
        calibration_report["seed_sweep"] = seed_sweep
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    report = {
        "group": group_name,
        "sensors": sensors,
        "output_dir": out_dirs["root"],
        "model_path": model_path,
        "best_model": best_model_type,
        "threshold": float(best_trial["threshold"]),
        "n_alarms": best_trial["n_alarms"],
        "alarms_with_detected_anomaly_in_window": best_trial["alarms_with_detected_anomaly_in_window"],
        "hit_rate": best_trial["hit_rate"],
        "composite_score": best_trial["composite_score"],
        "skipped": False,
    }
    with open(os.path.join(out_dirs["csv"], "evaluation_alarm_hit_rate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
