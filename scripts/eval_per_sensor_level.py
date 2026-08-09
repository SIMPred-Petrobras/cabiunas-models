#!/usr/bin/env python3
"""
eval_per_sensor_level.py
Avalia cada sensor individualmente contra seus próprios alarmes.

Novidades:
  --ok_aware    : cluster por sensor usando OK como reset (HIHI→OK→HIHI = 2 incidentes)
  --sticky_hours: após alertar, mantém alerta ativo por N horas (sticky alert)

Uso:
    PYTHONPATH=. python scripts/eval_per_sensor_level.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label inSample_2025 [--ok_aware] [--sticky_hours 8]
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from clearml import Task

SENSORS = [
    "T5_AVG_A",
    "TC382_01_A", "TC382_02_A", "TC382_03_A",
    "TC382_04_A", "TC382_05_A", "TC382_06_A",
    "TV_351X_A",  "TV_351Y_A",
    "TV_352X_A",  "TV_352Y_A",
    "TV_353X_A",  "TV_353Y_A",
    "TV_354X_A",  "TV_354Y_A",
    "TV_355X_A",  "TV_355Y_A",
]

ALARM_CSV_DEFAULT = "../dados/alarmes_selecionados_turbina_a.csv"
SAMPLING_INTERVAL = "5min"
DEBOUNCE_HOURS    = 8.0
HORIZON_HOURS     = 8.0
GAP_HOURS         = 4.0


# ---------------------------------------------------------------------------
# Carregamento de sequence_scores
# ---------------------------------------------------------------------------

def load_mae_series(task: Task, sensors: List[str]) -> Dict[str, pd.Series]:
    arts = task.artifacts
    series: Dict[str, pd.Series] = {}
    for sensor in sensors:
        key = next((k for k in arts if "sequence_scores_all" in k and sensor in k), None)
        if key is None:
            print(f"  [WARN] {sensor}: artifact não encontrado — ignorado")
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path)
        df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
        series[sensor] = df.set_index("seq_start_time")["mae_seq"]
    print(f"  {len(series)}/{len(sensors)} sensores carregados")
    return series


def load_running_masks(task: Task, sensors: List[str]) -> Dict[str, pd.Series]:
    """Carrega máscara 'on' de point_anomalies_all por sensor (resampleada para 5min)."""
    arts = task.artifacts
    masks: Dict[str, pd.Series] = {}
    for sensor in sensors:
        key = next((k for k in arts if "point_anomalies_all" in k and sensor in k), None)
        if key is None:
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path, usecols=["data_datetime", "operational_state"])
        df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["data_datetime"]).set_index("data_datetime")
        # resampla para 5min: True se maioria dos sub-pontos é "on"
        is_on = (df["operational_state"] == "on")
        masks[sensor] = is_on.resample(SAMPLING_INTERVAL).mean() >= 0.5
    return masks


# ---------------------------------------------------------------------------
# EWMA + quantile normalization
# ---------------------------------------------------------------------------

def ewma_quantile(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean().rank(pct=True)


# ---------------------------------------------------------------------------
# Sticky alert: mantém alerta ativo por sticky_hours após último disparo
# ---------------------------------------------------------------------------

def apply_sticky(health: pd.Series, threshold_q: float,
                 sticky_hours: float) -> pd.Series:
    """Retorna série booleana com sticky alert aplicado."""
    alert = (health >= threshold_q).copy()
    if sticky_hours <= 0 or not alert.any():
        return alert
    sticky_td  = pd.Timedelta(hours=sticky_hours)
    idx        = alert.index
    result     = alert.values.copy()
    alert_pos  = np.where(result)[0]
    for pos in alert_pos:
        end_t = idx[pos] + sticky_td
        end_pos = idx.searchsorted(end_t, side="right")
        result[pos:end_pos] = True
    return pd.Series(result, index=idx)


# ---------------------------------------------------------------------------
# Carregamento de alarmes — gap-based e OK-aware
# ---------------------------------------------------------------------------

def _parse_alarm_df(alarm_csv: str) -> Tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(alarm_csv)
    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower()
         or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    cond_col = next((c for c in df.columns if "condi" in c.lower()), None)
    tag_col  = next((c for c in df.columns if "tag" in c.lower()
                     and "alarm" in c.lower()), None)
    if tag_col is None:  # formato novo (base 2025): coluna apenas "Tag"
        tag_col = next((c for c in df.columns if c.strip().lower() == "tag"), None)
    if tag_col is None:
        raise ValueError("Coluna de tag ('Tag Alarme' ou 'Tag') não encontrada.")
    df = df.copy()
    df["_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_time"]).sort_values("_time")
    return df, date_col, cond_col, tag_col


def load_alarms_gap(alarm_csv: str,
                    exclude_conditions: List[str] | None = None) -> Dict[str, List[pd.Timestamp]]:
    """Onset alarms por sensor (exclui OK). Clustering feito depois por gap."""
    df, _, cond_col, tag_col = _parse_alarm_df(alarm_csv)
    if cond_col:
        df = df[df[cond_col].str.upper().fillna("").ne("OK")]
        if exclude_conditions:
            excl = [c.upper() for c in exclude_conditions]
            df = df[~df[cond_col].str.upper().isin(excl)]
    result: Dict[str, List[pd.Timestamp]] = {}
    for sensor in SENSORS:
        rows = df[df[tag_col] == sensor]
        result[sensor] = sorted(pd.to_datetime(rows["_time"], utc=True).tolist())
    return result


def load_alarms_ok_aware(
    alarm_csv: str,
    exclude_conditions: List[str] | None = None,
    min_alarm_duration_minutes: float = 0.0,
) -> Dict[str, List[pd.Timestamp]]:
    """OK-aware: qualquer onset após um OK = novo incidente (reset pelo OK).
    Não usa gap — o próprio OK define o fim do evento.

    min_alarm_duration_minutes: filtra alarmes cujo tempo onset→OK < limite.
    Remove fleeting alarms (ISA-18.2) que ativam e limpam em < N minutos.
    """
    df, _, cond_col, tag_col = _parse_alarm_df(alarm_csv)
    excl = {c.upper() for c in (exclude_conditions or [])}
    result: Dict[str, List[pd.Timestamp]] = {}
    for sensor in SENSORS:
        sensor_df = df[df[tag_col] == sensor].sort_values("_time")
        incidents: List[pd.Timestamp] = []
        onset_time: pd.Timestamp | None = None
        for _, row in sensor_df.iterrows():
            cond = str(row[cond_col]).upper() if cond_col else "CFN"
            if cond == "OK":
                if onset_time is not None and min_alarm_duration_minutes > 0:
                    dur = (row["_time"] - onset_time).total_seconds() / 60.0
                    if dur >= min_alarm_duration_minutes:
                        incidents.append(onset_time)
                elif onset_time is not None:
                    incidents.append(onset_time)
                onset_time = None
            elif cond not in excl:
                if onset_time is None:
                    onset_time = row["_time"]
        # Alerta ainda aberto ao fim do período (sem OK) — não conta como fleeting
        if onset_time is not None:
            incidents.append(onset_time)
        result[sensor] = incidents
    return result


# ---------------------------------------------------------------------------
# Clustering gap-based (usado apenas no modo padrão)
# ---------------------------------------------------------------------------

def cluster_incidents(alarm_times: List[pd.Timestamp],
                      gap_hours: float = GAP_HOURS) -> List[pd.Timestamp]:
    if not alarm_times:
        return []
    s = pd.Series(alarm_times).sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > gap_hours).cumsum()
    return s.groupby(g).first().tolist()


# ---------------------------------------------------------------------------
# Gap-debounce de episódios de alerta (FA)
# ---------------------------------------------------------------------------

def detect_episodes_gap(alert: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)
    on_idx = alert.index[alert]
    if len(on_idx) == 0:
        return []
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cs, ce = on_idx[0], on_idx[0]
    for t in on_idx[1:]:
        if (t - ce) <= debounce:
            ce = t
        else:
            episodes.append((cs, ce))
            cs = ce = t
    episodes.append((cs, ce))
    return episodes


# ---------------------------------------------------------------------------
# Duração mínima: remove episódios mais curtos que min_duration_hours
# ---------------------------------------------------------------------------

def apply_min_duration(
    alert: pd.Series,
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]],
    min_duration_hours: float,
) -> Tuple[pd.Series, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    if min_duration_hours <= 0:
        return alert, episodes
    min_td = pd.Timedelta(hours=min_duration_hours)
    long_eps = [(s0, s1) for s0, s1 in episodes if (s1 - s0) >= min_td]
    if len(long_eps) == len(episodes):
        return alert, episodes
    new_alert = pd.Series(False, index=alert.index)
    for s0, s1 in long_eps:
        new_alert[(alert.index >= s0) & (alert.index <= s1)] = True
    return new_alert, long_eps


# ---------------------------------------------------------------------------
# Avaliação: recall × FA com sweep de threshold
# ---------------------------------------------------------------------------

def best_point_for_sensor(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    sticky_hours: float = 0.0,
    min_duration_hours: float = 0.0,
    n_thresholds: int = 100,
    fa_budget: float = 1.0,
    min_duration_grid: "list | None" = None,
    max_duty_cycle: float = 1.0,
    max_sticky_duty: float = 0.25,
) -> dict:
    """Escolhe o ponto de operação que MAXIMIZA recall sob o teto de FA; no recall
    máximo, desempata por MAIOR lead mediano (antecipação é o valor do preditivo;
    FA é restrição via fa_budget, não critério) e, no empate de lead, por menor FA.
    Quando `min_duration_grid` é fornecido, varre debounce × threshold.

    `max_duty_cycle` (default 1.0 = sem efeito) rejeita thresholds cujo duty-cycle
    bruto (fração do tempo com health>=q, antes do sticky) ultrapasse o teto. A
    FA-por-episódio não enxerga tempo-em-alerta, então sem esse teto a busca escolhe
    o piso q=0.5 (alarme ligado quase sempre). Com o teto, o ponto vira deployável.

    `max_sticky_duty` (default 0.25) limita o tempo-em-alerta REAL (pós-sticky): o
    sticky de 12h multiplica o duty bruto (ex.: 35%→76%) e a busca "compra" recall
    com alerta quase-sempre-ligado sem que a FA-por-episódio denuncie. Também
    devolve `recall_raw` (hits por cruzamento bruto de health>=q na janela, sem
    sticky) — recall que não depende da cauda do sticky de um alerta anterior.
    """
    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])

    md_candidates = list(min_duration_grid) if min_duration_grid else [min_duration_hours]

    best = {"recall": 0.0, "fa_per_day": 0.0, "threshold_q": 0.5, "duty_cycle": 1.0,
            "min_duration_hours": float(md_candidates[0]), "n_incidents": len(incidents)}
    best_recall = -1.0
    best_fa = float("inf")
    best_lead = -1.0

    for md in md_candidates:
        for q in np.linspace(0.50, 0.999, n_thresholds):
            duty = float((health >= q).mean())   # tempo-em-alerta bruto (pré-sticky)
            if duty > max_duty_cycle:
                continue
            alert    = apply_sticky(health, q, sticky_hours)
            duty_sticky = float(alert.mean())
            if duty_sticky > max_sticky_duty:
                continue
            episodes = detect_episodes_gap(alert)
            alert, episodes = apply_min_duration(alert, episodes, md)
            alert_s  = np.array([t.timestamp() for t in health.index[alert]])
            raw_s    = np.array([t.timestamp() for t in health.index[health >= q]])

            n_hit = 0
            n_hit_raw = 0
            leads = []
            for ti in inc_s:
                w = alert_s[(alert_s >= ti - horizon_sec) & (alert_s <= ti)] if alert_s.size else np.array([])
                if w.size:
                    n_hit += 1
                    leads.append((ti - w.min()) / 3600.0)
                if raw_s.size and np.any((raw_s >= ti - horizon_sec) & (raw_s <= ti)):
                    n_hit_raw += 1
            n_fp = sum(
                1 for (s0, s1) in episodes
                if not (np.any(
                    (inc_s - horizon_sec <= s1.timestamp()) &
                    (inc_s >= s0.timestamp())
                ) if inc_s.size else False)
            )

            fa    = n_fp / max(total_days, 1.0)
            recall = n_hit / len(incidents) if incidents else 0.0
            med_lead = float(np.median(leads)) if leads else 0.0

            if fa > fa_budget:
                continue
            if incidents:
                # max recall; empate → maior lead mediano SE a diferença for material
                # (>0.5h; o lead satura no horizonte, então diferenças de segundos
                # são ruído e não justificam pagar mais FA); senão → menor FA
                better = (recall > best_recall) or (
                    recall == best_recall and (
                        (med_lead > best_lead + 0.5) or
                        (abs(med_lead - best_lead) <= 0.5 and fa < best_fa)
                    )
                )
            else:
                # sem incidentes não há recall para satisfazer: preserva o legado
                better = recall > best_recall
            if better:
                best_recall = recall
                best_fa = fa
                best_lead = med_lead
                best = {
                    "recall":      recall,
                    "recall_raw":  n_hit_raw / len(incidents) if incidents else 0.0,
                    "fa_per_day":  fa,
                    "threshold_q": float(q),
                    "duty_cycle":  duty,
                    "duty_sticky": duty_sticky,
                    "min_duration_hours": float(md),
                    "n_incidents": len(incidents),
                    "n_hit":       n_hit,
                    "n_hit_raw":   n_hit_raw,
                    "n_fp":        n_fp,
                    "total_days":  total_days,
                    "median_lead_hours": med_lead,
                    "lead_p25_hours": float(np.percentile(leads, 25)) if leads else 0.0,
                    "lead_p75_hours": float(np.percentile(leads, 75)) if leads else 0.0,
                }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",      required=True)
    parser.add_argument("--label",        default="eval")
    parser.add_argument("--alarm_csv",    default=ALARM_CSV_DEFAULT)
    parser.add_argument("--sensors", nargs="*", default=None,
                        help="Sensores a avaliar (default: os 17 da frota). Necessário p/ "
                             "colunas fora da lista, ex.: 954005_624_TI_0305")
    parser.add_argument("--half_life",    type=float, default=4.0,
                        help="Half-life da EWMA (h) — default global")
    parser.add_argument("--half_life_overrides", nargs="*", default=[],
                        help="Override de half-life por sensor: SENSOR=HORAS "
                             "(ex: TC382_04_A=0.5). Eventos breves (UNDER) pedem hl curto; "
                             "deriva sustentada (térmica) pede hl longo.")
    parser.add_argument("--horizon",      type=float, default=HORIZON_HOURS)
    parser.add_argument("--fa_budget",    type=float, default=1.0)
    parser.add_argument("--max_duty_cycle", type=float, default=1.0,
                        help="teto de duty-cycle (fração tempo-em-alerta); <1 evita o piso q=0.5")
    parser.add_argument("--eval_start",   default=None)
    parser.add_argument("--eval_end",     default=None)
    parser.add_argument("--ok_aware",     action="store_true",
                        help="Usa OK como reset de incidente (HIHI→OK→HIHI = 2 inc)")
    parser.add_argument("--sticky_hours",       type=float, default=0.0,
                        help="Horas que o alerta fica ativo após o último disparo")
    parser.add_argument("--min_duration_hours", type=float, default=0.0,
                        help="Duração mínima (horas) de um episódio de alerta para contar")
    parser.add_argument("--tune_debounce", nargs="*", type=float, default=None,
                        help="Varre debounce junto do threshold p/ minimizar FP sem perder recall. "
                             "Sem valores usa grade [0,0.5,1,2]h; ou passe a grade (ex: --tune_debounce 0 1 3)")
    parser.add_argument("--exclude_conditions", nargs="*", default=[],
                        help="Condições a excluir da avaliação (ex: LOLO)")
    parser.add_argument("--min_alarm_duration_minutes", type=float, default=0.0,
                        help="Filtra alarmes ground-truth com duração < N min (remove fleeting alarms ISA-18.2)")
    parser.add_argument("--mask_off",     action="store_true",
                        help="Zera health score durante operational_state != 'on'")
    parser.add_argument("--exclude_off_alarms", action="store_true",
                        help="Exclui do ground-truth alarmes ocorridos em período OFF "
                             "(operational_state != 'on'), ex: UNDER de termopar frio em turbina desligada")
    parser.add_argument("--out_dir",      default="eval_predictive_out/per_sensor_level")
    args = parser.parse_args()

    if args.sensors:
        # load_alarms_* e load_mae_series leem o global — sobrescrever aqui cobre tudo
        global SENSORS
        SENSORS = list(args.sensors)
        print(f"[SENSORS] Avaliando lista customizada: {SENSORS}")

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    print(f"\n[1/4] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")
    if args.ok_aware:
        print("      Modo: OK-aware clustering")
    if args.sticky_hours > 0:
        print(f"      Sticky alert: {args.sticky_hours}h")

    print(f"\n[2/4] Baixando sequence_scores...")
    mae_dict = load_mae_series(task, SENSORS)

    running_masks: Dict[str, pd.Series] = {}
    if args.mask_off or args.exclude_off_alarms:
        print(f"      Carregando operational_state (mask_off/exclude_off_alarms)...")
        running_masks = load_running_masks(task, SENSORS)

    hl_overrides: Dict[str, float] = {}
    for item in (args.half_life_overrides or []):
        if "=" in item:
            k, v = item.split("=", 1)
            hl_overrides[k.strip()] = float(v)
    print(f"\n[3/4] EWMA (hl={args.half_life}h" +
          (f", overrides={hl_overrides}" if hl_overrides else "") +
          ") + quantile normalization...")
    health_dict = {s: ewma_quantile(mae, hl_overrides.get(s, args.half_life))
                   for s, mae in mae_dict.items()}

    if t0 or t1:
        print(f"      Período: {t0.date() if t0 else 'início'} → {t1.date() if t1 else 'fim'}")

    print(f"\n[4/4] Avaliando sensor a sensor (H={args.horizon}h)...")
    excl_conds = args.exclude_conditions or []
    if excl_conds:
        print(f"      Excluindo condições: {excl_conds}")

    min_alarm_dur = getattr(args, "min_alarm_duration_minutes", 0.0)
    if min_alarm_dur > 0:
        print(f"      Filtro fleeting: alarmes < {min_alarm_dur:.0f}min excluídos do ground-truth")

    if args.ok_aware:
        raw_alarms = load_alarms_ok_aware(args.alarm_csv, excl_conds, min_alarm_dur)
        def get_incidents(sensor, alarms_s):
            return alarms_s
    else:
        raw_alarms = load_alarms_gap(args.alarm_csv, excl_conds)
        def get_incidents(sensor, alarms_s):
            return cluster_incidents(alarms_s)

    debounce_grid = None
    if args.tune_debounce is not None:
        debounce_grid = args.tune_debounce if len(args.tune_debounce) > 0 else [0.0, 0.5, 1.0, 2.0]
        print(f"      Tuning debounce: grade {debounce_grid}h (minimiza FP sem perder recall)")

    rows = []
    for sensor, health in health_dict.items():
        h = health.copy()
        if t0: h = h[h.index >= t0]
        if t1: h = h[h.index <= t1]
        if h.empty:
            continue

        # Zera health score fora de operational_state == 'on'
        if args.mask_off and sensor in running_masks:
            mask = running_masks[sensor].reindex(h.index, method="nearest", tolerance=pd.Timedelta("6min")).fillna(False)
            h = h.where(mask, other=0.0)

        alarms_s = raw_alarms.get(sensor, [])
        alarms_s = [a for a in alarms_s
                    if (t0 is None or a >= t0) and (t1 is None or a <= t1)]

        # Exclui alarmes ocorridos em período OFF (ex: UNDER de termopar frio com
        # a turbina desligada — o detector mascara esse regime, então não devem
        # contar como incidente perdido).
        if args.exclude_off_alarms and sensor in running_masks and alarms_s:
            rm = running_masks[sensor]
            on_at = rm.reindex(pd.DatetimeIndex(alarms_s), method="nearest",
                               tolerance=pd.Timedelta("30min")).fillna(True)
            kept = [a for a, ok in zip(alarms_s, on_at.tolist()) if ok]
            n_off = len(alarms_s) - len(kept)
            if n_off:
                print(f"  [OFF-EXCL] {sensor}: {n_off}/{len(alarms_s)} alarmes em período OFF removidos do ground-truth")
            alarms_s = kept

        incidents = get_incidents(sensor, alarms_s)

        if not incidents:
            print(f"  {sensor}: 0 incidentes — FA/dia medido, recall=N/A")
            result = best_point_for_sensor(h, [], args.horizon,
                                           args.sticky_hours, args.min_duration_hours,
                                           fa_budget=args.fa_budget,
                                           min_duration_grid=debounce_grid,
                                           max_duty_cycle=args.max_duty_cycle)
        else:
            result = best_point_for_sensor(h, incidents, args.horizon,
                                           args.sticky_hours, args.min_duration_hours,
                                           fa_budget=args.fa_budget,
                                           min_duration_grid=debounce_grid,
                                           max_duty_cycle=args.max_duty_cycle)
            print(f"  {sensor}: {len(incidents)} inc | "
                  f"rec={result['recall']:.2f} FA={result['fa_per_day']:.3f} "
                  f"duty={result.get('duty_cycle', float('nan')):.2f} "
                  f"db={result.get('min_duration_hours', 0.0):.1f}h")

        result["sensor"] = sensor
        rows.append(result)

    df_out = pd.DataFrame(rows).set_index("sensor")
    col_order = ["n_incidents", "recall", "fa_per_day", "threshold_q", "duty_cycle",
                 "min_duration_hours", "n_hit", "n_fp", "total_days"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    mode_tag = ("_ok_aware" if args.ok_aware else "") + \
               (f"_sticky{int(args.sticky_hours)}h" if args.sticky_hours > 0 else "") + \
               (f"_mindur{args.min_duration_hours:.1f}h" if args.min_duration_hours > 0 else "") + \
               ("_maskoff" if args.mask_off else "") + \
               ("_offexcl" if args.exclude_off_alarms else "")
    out_label = f"{args.label}{mode_tag}"

    print(f"\n=== RESULTADO POR SENSOR (H={args.horizon}h, {out_label}) ===")
    print(f"{'Sensor':<18} {'Inc':>5} {'Recall':>8} {'FA/dia':>8} {'thr_q':>6} {'duty':>6} {'db(h)':>6}")
    print("─" * 64)
    for sensor, row in df_out.iterrows():
        n   = int(row.get("n_incidents", 0))
        rec = row.get("recall", float("nan"))
        fa  = row.get("fa_per_day", float("nan"))
        thq = row.get("threshold_q", float("nan"))
        duty = row.get("duty_cycle", float("nan"))
        db  = row.get("min_duration_hours", 0.0)
        rec_str = "  N/A" if n == 0 else f"{rec:7.1%}"
        print(f"  {sensor:<16} {n:>5}  {rec_str}  {fa:>8.3f} {thq:>6.3f} {duty:>6.2f} {db:>6.1f}")

    out_csv = os.path.join(args.out_dir, f"per_sensor_eval_{out_label}.csv")
    df_out.to_csv(out_csv)
    print(f"\nSalvo: {out_csv}")


if __name__ == "__main__":
    main()
