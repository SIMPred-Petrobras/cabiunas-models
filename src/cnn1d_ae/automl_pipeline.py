from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config, load_alert_context_catalog
from .model import build_callbacks
from .preprocess import (
    build_sensor_dataframe,
    build_group_dataframe,
    build_exclusion_mask,
    clip_outliers,
    normalize_train_only,
    select_feature_columns,
    THERMAL_ARRAY_SPREAD_COL,
    BEARING_SPREAD_COL,
    VIBRATION_ENVELOPE_COL,
    ALARM_RECENCY_COL,
    compute_alarm_recency_feature,
)
from .scoring import (
    map_seq_to_point_anomalies,
    compute_anomaly_rate_per_day,
    build_operational_state,
    eval_alarm_hit_rate,
    compute_normal_alert_rate,
    compute_composite_score,
    apply_load_gate,
    compute_volatility_index,
    apply_volatility_gate,
    compute_frozen_sensor_mask,
    apply_frozen_sensor_veto,
    apply_min_duration_filter,
    compute_step_change_index,
    apply_step_change_gate,
    annotate_alert_catalog_context,
    compute_catalog_enrichment_control,
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


def _fit_score_ocsvm(
    cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, n_features: int,
    nu: float | None = None, gamma: Any = None,
):
    # OneClassSVM (kernel RBF) escala ~O(n^2)-O(n^3) no numero de amostras de
    # treino -- com centenas de milhares de pontos (datasets maiores/janelas
    # de fit mais longas) o ajuste fica impraticavel. Subamostra so o *fit*;
    # o score (train_err/all_err) continua sendo calculado sobre os dados
    # inteiros, que e barato (so avalia contra os vetores de suporte).
    nu = cfg.AUTOML_OCSVM_NU if nu is None else nu
    gamma = cfg.AUTOML_OCSVM_GAMMA if gamma is None else gamma
    x_fit = x_normal
    max_train = cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES
    if max_train and len(x_normal) > max_train:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(len(x_normal), size=int(max_train), replace=False)
        x_fit = x_normal[idx]
    clf = fit_ocsvm(x_fit, nu, gamma)
    train_err = ocsvm_error(clf, x_normal)
    all_err = ocsvm_error(clf, x_all)
    return train_err, all_err, clf, {"nu": nu, "gamma": gamma, "train_samples": int(len(x_fit))}


def _ocsvm_param_grid(cfg: PipelineConfig) -> List[Tuple[float, Any]]:
    """Grade de (nu, gamma) a testar para ocsvm. Sem grades explicitas
    definidas, cai no par unico de AUTOML_OCSVM_NU/AUTOML_OCSVM_GAMMA
    (comportamento anterior, inalterado). Ver
    docs/analise_automl_exp9_planejamento.md (item 3)."""
    nus = cfg.AUTOML_OCSVM_NU_GRID or [cfg.AUTOML_OCSVM_NU]
    gammas = cfg.AUTOML_OCSVM_GAMMA_GRID or [cfg.AUTOML_OCSVM_GAMMA]
    return [(nu, gamma) for nu in nus for gamma in gammas]


def _fit_score_iforest(cfg: PipelineConfig, x_normal: np.ndarray, x_all: np.ndarray, n_features: int):
    model = fit_isolation_forest(
        x_normal, cfg.AUTOML_IFOREST_CONTAMINATION, cfg.AUTOML_IFOREST_N_ESTIMATORS, cfg.RANDOM_SEED
    )
    train_err = isolation_forest_error(model, x_normal)
    all_err = isolation_forest_error(model, x_all)
    return train_err, all_err, model, {"contamination": cfg.AUTOML_IFOREST_CONTAMINATION,
                                        "n_estimators": cfg.AUTOML_IFOREST_N_ESTIMATORS}


_FITTERS = {"dense": _fit_score_dense, "ocsvm": _fit_score_ocsvm, "iforest": _fit_score_iforest}


_SEED_SWEEP_SUPPORTED = ("iforest", "ocsvm")


def _refit_with_seed(
    cfg: PipelineConfig, model_type: str, x_normal: np.ndarray, x_all: np.ndarray, seed: int,
    nu: float | None = None, gamma: Any = None,
):
    """Re-treina `model_type` com uma seed especifica. Para `iforest` a
    aleatoriedade vem do proprio ensemble (random_state); para `ocsvm` vem
    da subamostragem do treino quando x_normal > AUTOML_OCSVM_MAX_TRAIN_SAMPLES
    (o algoritmo do SVM em si e deterministico, mas QUAIS pontos entram no
    fit muda com a seed). `nu`/`gamma` sobrescrevem AUTOML_OCSVM_NU/GAMMA --
    necessario porque o melhor trial pode vir de AUTOML_OCSVM_NU_GRID/
    AUTOML_OCSVM_GAMMA_GRID (EXP9 item 3), nao do par unico da config."""
    if model_type == "iforest":
        model = fit_isolation_forest(x_normal, cfg.AUTOML_IFOREST_CONTAMINATION, cfg.AUTOML_IFOREST_N_ESTIMATORS, seed)
        return isolation_forest_error(model, x_normal), isolation_forest_error(model, x_all)
    if model_type == "ocsvm":
        nu = cfg.AUTOML_OCSVM_NU if nu is None else nu
        gamma = cfg.AUTOML_OCSVM_GAMMA if gamma is None else gamma
        x_fit = x_normal
        max_train = cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES
        if max_train and len(x_normal) > max_train:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x_normal), size=int(max_train), replace=False)
            x_fit = x_normal[idx]
        model = fit_ocsvm(x_fit, nu, gamma)
        return ocsvm_error(model, x_normal), ocsvm_error(model, x_all)
    raise ValueError(f"seed sweep nao suportado para model_type={model_type!r}")


def _seed_sweep(
    cfg: PipelineConfig, model_type: str, x_normal: np.ndarray, x_all: np.ndarray, all_index: pd.Index,
    state: pd.Series | None, df_alarm_eval: pd.DataFrame, df_point_eval_idx: pd.Index,
    near_alarm_mask: pd.Series, pct: float, debounce: int, n_seeds: int,
    nu: float | None = None, gamma: Any = None, load_gate_series: pd.Series | None = None,
    volatility_index: pd.Series | None = None, frozen_mask: pd.Series | None = None,
    step_change_index: pd.Series | None = None,
) -> List[Dict[str, Any]]:
    """Re-treina o mesmo modelo (mesmo threshold_percentile/debounce/nu/gamma
    do melhor trial) com N seeds extras, pra medir o quanto hit_rate/
    normal_alert_rate variam so por causa da aleatoriedade do ajuste —
    ver analise_automl_lara.md secao 2 (~+-27pp de ruido de semente na
    pipeline da Lara)."""
    results = []
    for i in range(1, n_seeds + 1):
        seed = cfg.RANDOM_SEED + i
        train_err, all_err = _refit_with_seed(cfg, model_type, x_normal, x_all, seed, nu=nu, gamma=gamma)
        threshold = float(np.percentile(train_err, pct))
        anomaly_flags = (all_err > threshold).astype(int)
        df_point = map_seq_to_point_anomalies(
            anomaly_flags, all_index, time_steps=1,
            point_rule="all_of_window", point_window=int(debounce), point_min_count=int(debounce),
        )
        if state is not None:
            df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
            df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0
        # Filtro de duracao ANTES dos portoes de rampa/volatilidade/veto de
        # congelamento -- mede a persistencia do score BRUTO (pos-mascara
        # apenas), nao o que sobra depois desses portoes fragmentarem um
        # episodio longo em pedacos curtos (ver nota identica mais abaixo,
        # no laco principal de trials).
        if cfg.ENABLE_MIN_DURATION_FILTER:
            df_point = apply_min_duration_filter(df_point, cfg.MIN_DURATION_FILTER_MINUTES)
        if load_gate_series is not None:
            df_point = apply_load_gate(
                df_point, load_gate_series, ramp_max=cfg.LOAD_GATE_RAMP_MAX,
                level_min=cfg.LOAD_GATE_LEVEL_MIN, ramp_halflife_minutes=cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES,
                window_minutes=cfg.LOAD_GATE_WINDOW_MINUTES,
            )
        if volatility_index is not None:
            df_point = apply_volatility_gate(df_point, volatility_index, cfg.VOLATILITY_GATE_THRESHOLD)
        if frozen_mask is not None:
            df_point = apply_frozen_sensor_veto(df_point, frozen_mask)
        if step_change_index is not None:
            df_point = apply_step_change_gate(df_point, step_change_index, cfg.STEP_CHANGE_THRESHOLD)
        eval_stats = eval_alarm_hit_rate(df_alarm_eval, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
        normal_rate = compute_normal_alert_rate(
            df_point.loc[df_point_eval_idx], near_alarm_mask.loc[df_point_eval_idx]
        )
        results.append({"seed": seed, "hit_rate": eval_stats["hit_rate"], "normal_alert_rate": normal_rate})
    return results


def _walkforward_fit_periods(
    cfg: PipelineConfig,
    fitter,
    params: Tuple[float, Any] | None,
    df_normal: pd.DataFrame,
    df_all: pd.DataFrame,
    oos_start: pd.Timestamp,
    n_features: int,
) -> List[Dict[str, Any]]:
    """Re-treina do zero a cada `WALKFORWARD_RETRAIN_FREQ` (janela EXPANSIVA
    -- cada periodo usa TODO o normal disponivel antes dele, nao uma janela
    movel fixa), pontuando so o proximo periodo com o modelo daquele
    momento -- normalizacao (center/scale) tambem recomputada por periodo.
    Fit e feito UMA vez por periodo aqui (independente do percentil/debounce
    testados depois); o chamador aplica o percentil por cima de cada
    `train_err` de periodo pra montar `anomaly_flags`. Ver
    ENABLE_WALKFORWARD_RETRAIN em config.py e docs/analise_automl_exp10.md,
    secao "Retreino walk-forward mensal" (validado: mesmo hit_rate do
    modelo congelado, normal_alert_rate ~19% menor)."""
    data_end = df_all.index.max()
    period_starts = list(pd.date_range(oos_start, data_end, freq=cfg.WALKFORWARD_RETRAIN_FREQ))
    periods: List[Dict[str, Any]] = []
    for i, period_start in enumerate(period_starts):
        period_end = period_starts[i + 1] if i + 1 < len(period_starts) else data_end + pd.Timedelta(seconds=1)
        score_slice = df_all.loc[(df_all.index >= period_start) & (df_all.index < period_end)]
        if score_slice.empty:
            continue
        train_slice = df_normal.loc[df_normal.index < period_start]
        if len(train_slice) < 10:
            continue

        df_normal_z, df_score_z, _center, _scale = normalize_train_only(cfg, train_slice, score_slice)
        x_train = df_normal_z.values.astype(np.float32)
        x_score = df_score_z.values.astype(np.float32)

        if params is not None:
            nu, gamma = params
            train_err, score_err, model_obj, model_params = fitter(cfg, x_train, x_score, n_features, nu=nu, gamma=gamma)
        else:
            train_err, score_err, model_obj, model_params = fitter(cfg, x_train, x_score, n_features)

        periods.append({
            "period_start": period_start, "index": score_slice.index,
            "train_err": train_err, "score_err": score_err,
            "model_obj": model_obj, "model_params": model_params,
        })

    if not periods:
        raise ValueError("ENABLE_WALKFORWARD_RETRAIN: nenhum periodo com dados suficientes para treinar.")
    return periods


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
    if cfg.ENABLE_MIN_DURATION_FILTER and cfg.ENABLE_WALKFORWARD_RETRAIN:
        raise ValueError(
            "ENABLE_MIN_DURATION_FILTER=true com ENABLE_WALKFORWARD_RETRAIN=true nao e suportado -- "
            "3 tentativas independentes (docs/analise_automl_exp10.md) mostraram que filtro de duracao "
            "nao sobrevive a retreino mensal (derruba hit_rate de 92,5% pra 15-40%). Use um ou outro."
        )

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

    # Valores BRUTOS dos sensores do grupo (pre-restricao a feature_cols) --
    # e disso que o veto de sensor congelado precisa (nao das derivadas).
    raw_sensor_values = df_use[sensors].copy()

    # Feature de recencia de alarme de processo (ideia trazida por
    # representante de operacao da Cabiunas -- ver docstring de
    # compute_alarm_recency_feature): precisa de df_alarm, que
    # build_group_dataframe nao recebe, por isso e computada aqui.
    if cfg.ENABLE_ALARM_RECENCY_FEATURE and cfg.ALARM_RECENCY_TAGS:
        if "Tag" not in df_alarm.columns:
            raise ValueError("ENABLE_ALARM_RECENCY_FEATURE=true exige a coluna 'Tag' no alarme.")
        recency_alarm_times = df_alarm.loc[df_alarm["Tag"].isin(cfg.ALARM_RECENCY_TAGS), "Data da Ocorrencia"]
        df_use[ALARM_RECENCY_COL] = compute_alarm_recency_feature(
            df_use.index, recency_alarm_times, cfg.ALARM_RECENCY_HALFLIFE_MINUTES,
        )

    feature_cols = select_feature_columns(cfg, df_use, sensors)
    if cfg.ENABLE_THERMAL_ARRAY_SPREAD and THERMAL_ARRAY_SPREAD_COL in df_use.columns:
        feature_cols += select_feature_columns(cfg, df_use, [THERMAL_ARRAY_SPREAD_COL])
    if cfg.ENABLE_BEARING_SPREAD and BEARING_SPREAD_COL in df_use.columns:
        feature_cols += select_feature_columns(cfg, df_use, [BEARING_SPREAD_COL])
    if cfg.ENABLE_VIBRATION_ENVELOPE and VIBRATION_ENVELOPE_COL in df_use.columns:
        feature_cols += select_feature_columns(cfg, df_use, [VIBRATION_ENVELOPE_COL])
    if cfg.ENABLE_ALARM_RECENCY_FEATURE and ALARM_RECENCY_COL in df_use.columns:
        feature_cols += [ALARM_RECENCY_COL]
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
        secondary_series = None
        if cfg.OFF_TARGET_ABS_THRESHOLD is not None and target_sensor and target_sensor in df_use.columns:
            secondary_series = df_use[target_sensor]
        state = build_operational_state(
            index=df_use.index, sensor_series=ref_series,
            off_value_quantile=cfg.OFF_VALUE_QUANTILE, off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
            off_long_min_hours=cfg.OFF_LONG_MIN_HOURS, transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
            secondary_series=secondary_series, secondary_off_abs_threshold=cfg.OFF_TARGET_ABS_THRESHOLD,
        )

    # Portao de rampa de carga (EXP10b): suprime is_anom_point durante
    # manobra de carga legitima (rampa alta num proxy de carga, ex:
    # T5_AVG_A), sem mexer no threshold do modelo. Motivado por 8 dos 10
    # maiores episodios de falso alerta residual do EXP7 item1+2
    # coincidirem com uma rampa de dezenas de graus/hora + vibracao 3-6x
    # mais volatil -- manobra real, sem alarme, nao degradacao. Janela
    # curta (halflife/window pequenos) escolhida especificamente por
    # preservar os 29/29 casos preditivos reais (janelas longas
    # bloqueavam parte deles). Ja existia para o CNN1D-AE
    # (pipeline.py); portado aqui pro AutoML. Ver
    # docs/analise_automl_exp9_planejamento.md.
    load_gate_series = None
    if cfg.ENABLE_LOAD_GATE:
        gate_sensor = cfg.LOAD_GATE_SENSOR
        if not gate_sensor:
            raise ValueError("ENABLE_LOAD_GATE=true exige LOAD_GATE_SENSOR definido.")
        if gate_sensor not in sensors:
            df_gate, _ = build_sensor_dataframe(cfg, df_feat, df_raw, gate_sensor)
            load_gate_series = df_gate[gate_sensor]
        else:
            load_gate_series = df_use[gate_sensor]

    # Portao de volatilidade (complementar ao de rampa): suprime
    # is_anom_point quando o desvio-padrao movel medio de
    # VOLATILITY_GATE_SENSORS (tipicamente os canais de vibracao) excede
    # VOLATILITY_GATE_THRESHOLD. Motivado por episodios de falso alerta
    # residuais onde a vibracao fica mais volatil e PERMANECE assim durante
    # toda a manobra de carga, nao so na subida (o que o portao de rampa,
    # baseado em taxa de variacao, so cobre parcialmente). Ver
    # docs/analise_automl_exp9_planejamento.md.
    volatility_index = None
    if cfg.ENABLE_VOLATILITY_GATE:
        vol_sensors = cfg.VOLATILITY_GATE_SENSORS
        if not vol_sensors:
            raise ValueError("ENABLE_VOLATILITY_GATE=true exige VOLATILITY_GATE_SENSORS definido.")
        missing_vol = [s for s in vol_sensors if s not in df_use.columns]
        if missing_vol:
            raise ValueError(f"VOLATILITY_GATE_SENSORS fora de `sensors` do grupo: {missing_vol}")
        volatility_index = compute_volatility_index(df_use[vol_sensors], cfg.VOLATILITY_GATE_WINDOW_MINUTES)

    # Veto de sensor congelado: suprime is_anom_point quando qualquer
    # sensor BRUTO do grupo fica com leitura constante por uma janela
    # sustentada (falha de instrumento/comunicacao). Ver
    # docs/analise_automl_exp10.md, secao "Veto de sensor congelado".
    frozen_mask = None
    if cfg.ENABLE_FROZEN_SENSOR_VETO:
        frozen_mask = compute_frozen_sensor_mask(raw_sensor_values, cfg.FROZEN_SENSOR_VETO_WINDOW_MINUTES)

    # Portao de mudanca de nivel (step-change): complementa o portao de
    # rampa (taxa suavizada) para pegar degraus de carga quase
    # instantaneos. Ver docs/analise_automl_exp10.md, secao "Portao de
    # mudanca de nivel (EXP22)".
    step_change_index = None
    if cfg.ENABLE_STEP_CHANGE_GATE:
        step_sensor = cfg.STEP_CHANGE_GATE_SENSOR or cfg.LOAD_GATE_SENSOR
        if not step_sensor:
            raise ValueError("ENABLE_STEP_CHANGE_GATE=true exige STEP_CHANGE_GATE_SENSOR ou LOAD_GATE_SENSOR definido.")
        if step_sensor not in sensors:
            df_step, _ = build_sensor_dataframe(cfg, df_feat, df_raw, step_sensor)
            step_series = df_step[step_sensor]
        else:
            step_series = df_use[step_sensor]
        step_change_index = compute_step_change_index(
            step_series, cfg.STEP_CHANGE_SHORT_WINDOW_MINUTES, cfg.STEP_CHANGE_LONG_WINDOW_MINUTES
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

    # Captura os quantis de clipping calculados sobre df_normal PRE-clip --
    # clip_outliers nao os retorna, e ate aqui nenhum artefato do pipeline
    # os persistia. Necessario para portar o modelo a producao (ver
    # simpred-cabiunas/docs/tc33003a.md secao 5 -- essa lacuna foi
    # descoberta tentando recuperar os artefatos do EXP10c).
    _clip_q_low = df_normal.quantile(cfg.OUTLIER_Q_LOW).to_dict() if cfg.OUTLIER_MODE.lower() == "quantile" else None
    _clip_q_high = df_normal.quantile(cfg.OUTLIER_Q_HIGH).to_dict() if cfg.OUTLIER_MODE.lower() == "quantile" else None

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

    df_normal_z, df_all_z, _normalize_center, _normalize_scale = normalize_train_only(cfg, df_normal_fit, df_all)

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
    if cfg.EXTRA_NEAR_ALARM_TAGS:
        if "Tag" not in df_alarm.columns:
            raise ValueError("EXTRA_NEAR_ALARM_TAGS exige a coluna 'Tag' no alarme.")
        extra_alarm_times = (
            df_alarm.loc[df_alarm["Tag"].isin(cfg.EXTRA_NEAR_ALARM_TAGS), "Data da Ocorrencia"]
            .dropna().sort_values()
        )
        extra_near_mask = build_exclusion_mask(all_index, extra_alarm_times, cfg.EXTRA_NEAR_ALARM_WINDOW_MINUTES)
        near_alarm_mask = near_alarm_mask | extra_near_mask

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
    best_all_err: np.ndarray | None = None
    best_threshold_value: float | None = None

    for model_type in model_types:
        fitter = _FITTERS.get(model_type)
        if fitter is None:
            print(f"[WARN] automl group={group_name}: modelo desconhecido '{model_type}' ignorado")
            continue

        # Para ocsvm, AUTOML_OCSVM_NU_GRID/AUTOML_OCSVM_GAMMA_GRID (se
        # definidos) expandem esse model_type num conjunto de (nu, gamma) a
        # re-treinar, cada um entrando no ranking de trials como uma
        # variante independente -- outros model_types sempre caem no par
        # unico (comportamento anterior). Ver
        # docs/analise_automl_exp9_planejamento.md (item 3).
        param_combos: List[Tuple[float, Any] | None]
        if model_type == "ocsvm":
            param_combos = list(_ocsvm_param_grid(cfg))
        else:
            param_combos = [None]

        for params in param_combos:
            walkforward_periods = None
            if cfg.ENABLE_WALKFORWARD_RETRAIN:
                if oos_start is None:
                    raise ValueError("ENABLE_WALKFORWARD_RETRAIN=true exige AUTOML_OOS_SPLIT_DATE definido.")
                label = f"nu={params[0]} gamma={params[1]}" if params is not None else ""
                print(f"[AUTOML] group={group_name} model={model_type} {label} — walk-forward "
                      f"({cfg.WALKFORWARD_RETRAIN_FREQ}, janela expansiva)...")
                walkforward_periods = _walkforward_fit_periods(
                    cfg, fitter, params, df_normal, df_all, oos_start, n_features
                )
                model_obj = walkforward_periods[-1]["model_obj"]
                model_params = walkforward_periods[-1]["model_params"]
            else:
                if params is not None:
                    nu, gamma = params
                    print(f"[AUTOML] group={group_name} model={model_type} nu={nu} gamma={gamma} — treinando...")
                    train_err, all_err, model_obj, model_params = fitter(cfg, x_normal, x_all, n_features, nu=nu, gamma=gamma)
                else:
                    print(f"[AUTOML] group={group_name} model={model_type} — treinando...")
                    train_err, all_err, model_obj, model_params = fitter(cfg, x_normal, x_all, n_features)

                if cfg.ENABLE_SCORE_EWMA:
                    halflife = pd.Timedelta(cfg.SCORE_EWMA_HALFLIFE)
                    train_err_s = pd.Series(train_err, index=df_normal_fit.index)
                    all_err_s = pd.Series(all_err, index=all_index)
                    train_err = train_err_s.ewm(halflife=halflife, times=train_err_s.index).mean().values
                    all_err = all_err_s.ewm(halflife=halflife, times=all_err_s.index).mean().values

            for pct in percentiles:
                if walkforward_periods is not None:
                    # Um percentil, N modelos: cada periodo usa o proprio
                    # threshold (percentil do PROPRIO train_err daquele
                    # retreino) -- nao um threshold global aplicado depois.
                    # Ver docs/analise_automl_exp10.md: um limiar/filtro
                    # calibrado contra UM modelo nao transferiu pra outros
                    # meses quando testado assim; aqui o percentil e
                    # recalculado a cada periodo desde o inicio, por design.
                    anomaly_flags = np.zeros(len(all_index), dtype=int)
                    threshold = None
                    for p in walkforward_periods:
                        threshold = float(np.percentile(p["train_err"], pct))
                        pos = all_index.get_indexer(p["index"])
                        anomaly_flags[pos] = (p["score_err"] > threshold).astype(int)
                else:
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
                    # Filtro de duracao ANTES dos portoes de rampa/volatilidade/
                    # veto de congelamento -- precisa medir a persistencia do
                    # score BRUTO (so pos-mascara operacional), nao o que sobra
                    # depois desses portoes fragmentarem um episodio real longo
                    # em varios pedacos curtos (cada um mais curto que o corte
                    # de duracao mesmo sem ser ruido). Ordem validada em
                    # docs/analise_automl_exp10.md, secao "EXP20": aplicar
                    # depois dos outros portoes derrubava o hit_rate de 90%
                    # (validado) pra 47,5% (bug de ordem, corrigido aqui).
                    if cfg.ENABLE_MIN_DURATION_FILTER:
                        df_point = apply_min_duration_filter(df_point, cfg.MIN_DURATION_FILTER_MINUTES)
                    if load_gate_series is not None:
                        df_point = apply_load_gate(
                            df_point, load_gate_series, ramp_max=cfg.LOAD_GATE_RAMP_MAX,
                            level_min=cfg.LOAD_GATE_LEVEL_MIN, ramp_halflife_minutes=cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES,
                            window_minutes=cfg.LOAD_GATE_WINDOW_MINUTES,
                        )
                    if volatility_index is not None:
                        df_point = apply_volatility_gate(df_point, volatility_index, cfg.VOLATILITY_GATE_THRESHOLD)
                    if frozen_mask is not None:
                        df_point = apply_frozen_sensor_veto(df_point, frozen_mask)
                    if step_change_index is not None:
                        df_point = apply_step_change_gate(df_point, step_change_index, cfg.STEP_CHANGE_THRESHOLD)

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
                        "walkforward": walkforward_periods is not None,
                        "n_walkforward_periods": len(walkforward_periods) if walkforward_periods is not None else None,
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
                        # Score continuo do trial vencedor (nao so o binario
                        # is_anom_point) -- necessario pra construir canais de
                        # decisao com persistencia/CUSUM proprios (ver
                        # docs/analise_automl_exp10.md, secao "Camada de
                        # decisao multi-canal"). None no caso walk-forward
                        # (score vem por periodo, nao um array unico).
                        best_all_err = None if walkforward_periods is not None else np.asarray(all_err).copy()
                        best_threshold_value = threshold

    if best_trial is None:
        return {"group": group_name, "sensors": sensors, "skipped": True, "reason": "no_valid_trials"}

    df_ranking = pd.DataFrame(trials).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_ranking.to_csv(os.path.join(out_dirs["csv"], "automl_ranking.csv"), index=False)

    seed_sweep = None
    # _seed_sweep/_refit_with_seed re-treinam so no split unico (x_normal/
    # x_all congelados) e nao aplicam o veto de sensor congelado -- nao
    # tem o mesmo significado pro trial vencedor quando ele veio do
    # walk-forward (retreino por periodo). Desligado nesse caso; a
    # variancia de semente do walk-forward exigiria re-rodar todos os
    # periodos por semente (nao implementado, custo N vezes maior).
    if cfg.AUTOML_SEED_SWEEP_N and best_model_type in _SEED_SWEEP_SUPPORTED and not best_trial.get("walkforward"):
        seed_sweep_kwargs = {"load_gate_series": load_gate_series, "volatility_index": volatility_index,
                             "frozen_mask": frozen_mask, "step_change_index": step_change_index}
        if best_model_type == "ocsvm":
            seed_sweep_kwargs["nu"] = best_trial.get("nu")
            seed_sweep_kwargs["gamma"] = best_trial.get("gamma")
        extra = _seed_sweep(
            cfg, best_model_type, x_normal, x_all, all_index, state, df_alarm_eval, df_point_eval_idx, near_alarm_mask,
            best_trial["threshold_percentile"], best_trial["debounce"], cfg.AUTOML_SEED_SWEEP_N,
            **seed_sweep_kwargs,
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

    # Ver nota acima de _clip_q_low/_clip_q_high. Necessario para produzir
    # o predictor.py em simpred-cabiunas -- normalizacao/clipping em
    # producao usam estatisticas fixas de treino, que ate aqui nunca eram
    # persistidas por este pipeline.
    with open(os.path.join(out_dirs["best_model"], "normalization_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": feature_cols,
                "normalize_mode": cfg.NORMALIZE_MODE,
                "normalize_center": {k: float(v) for k, v in _normalize_center.to_dict().items()},
                "normalize_scale": {k: float(v) for k, v in _normalize_scale.to_dict().items()},
                "clip_mode": cfg.OUTLIER_MODE,
                "clip_q_low": {k: float(v) for k, v in (_clip_q_low or {}).items()},
                "clip_q_high": {k: float(v) for k, v in (_clip_q_high or {}).items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    assert best_point_df is not None

    # Anotacao de contexto de alerta (puramente informativa, nao mexe em
    # is_anom_point/hit_rate/normal_alert_rate) -- ver ENABLE_ALERT_CATALOG_CONTEXT
    # em config.py e docs/analise_automl_exp10.md, secao "Cruzamento com
    # catalogo completo". Roda por ultimo, depois de todos os portoes,
    # sobre o df_point ja finalizado do melhor trial.
    if cfg.ENABLE_ALERT_CATALOG_CONTEXT:
        alert_catalog = load_alert_context_catalog(cfg)
        best_point_df = annotate_alert_catalog_context(
            best_point_df, alert_catalog, cfg.ALERT_CONTEXT_WINDOW_HOURS,
        )
        n_anom = int((best_point_df["is_anom_point"] == 1).sum())
        n_explicado = int((best_point_df["alert_confidence"] == "explicado_catalogo").sum())
        print(f"[ALERT-CONTEXT] group={group_name}: {n_explicado}/{n_anom} pontos anomalos "
              f"explicados por catalogo amplo dentro de +-{cfg.ALERT_CONTEXT_WINDOW_HOURS}h")

        # Controle negativo OBRIGATORIO (nao opcional): sem ele, a taxa de
        # "explicado" acima nao tem como ser lida -- pode ser so reflexo de
        # um catalogo denso, nao evidencia real de corroboracao. Ver
        # compute_catalog_enrichment_control em scoring.py e
        # docs/analise_automl_exp10.md, secao "Controle negativo do
        # enriquecimento por catalogo".
        enrichment = compute_catalog_enrichment_control(
            best_point_df, alert_catalog, cfg.ALERT_CONTEXT_WINDOW_HOURS,
            n_samples=cfg.ALERT_CONTEXT_CONTROL_N_SAMPLES,
        )
        with open(os.path.join(out_dirs["csv"], "alert_context_enrichment.json"), "w", encoding="utf-8") as f:
            json.dump(enrichment, f, indent=2, ensure_ascii=False)
        print(f"[ALERT-CONTEXT-CONTROL] group={group_name}: baseline={enrichment['baseline_explained_rate']*100:.1f}% "
              f"anomalo={enrichment['anomaly_explained_rate']*100:.1f}% "
              f"enriquecimento={enrichment['enrichment_factor']:.2f}x")

    best_point_df.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    if best_all_err is not None:
        df_scores = pd.DataFrame({"score": best_all_err, "threshold": best_threshold_value}, index=all_index)
        df_scores.to_csv(os.path.join(out_dirs["csv"], "sequence_scores_all.csv"))

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
