"""Reconstrucao independente dos 3 achados do colega sobre o canal 4
(alarme de processo, sem modelo) da `scripts/pipeline_unificada_final.py`,
antes de termos os scripts originais dele (ver docs/analise_automl_exp10.md,
secao "Revisao externa do colega -- canal 4").

Reusa os artefatos JA CALCULADOS em `runs_pipeline_unificada_final/
point_anomalies_final.csv` (saida da propria pipeline unificada, com os
4 canais booleanos ponto-a-ponto) -- nao precisa buscar nada no ClearML
pros canais 1-3. So o catalogo de alarmes completo (canal 4) e lido de
uma copia local em cache do dataset ClearML `a97ba56ba14840fbb1125c2a82f883c9`.

Testes:
  1. Ablacao do canal 4 -- vota so entre os 3 canais modelados (>=2 de 3).
  2. Controle negativo por tag -- taxa de acerto observada vs esperada
     por acaso (duty cycle da propria janela de 24h), teste binomial.
  3. Remocao das tags de "utilidade" (gas/motor de partida) do canal 4.
  4. Varredura fina do filtro de duracao minima na votacao (14-60min).

Uso:
    PYTHONPATH=. python dataset_francisco_lara/validacao_canal4_colega.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.cnn1d_ae.scoring import (
    combine_channels_vote,
    apply_refractory,
    apply_min_duration_filter,
    group_alerts_into_episodes,
    classify_episodes_regua,
    compute_regua_metrics,
    compute_operational_period_days,
)

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs_pipeline_unificada_final")
POINT_CSV = os.path.join(RUN_DIR, "point_anomalies_final.csv")
FALHAS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarmes_francisco_falhas.csv")
# copia local em cache do dataset ClearML a97ba56ba14840fbb1125c2a82f883c9
# (mesmo dataset que scripts/pipeline_unificada_final.py busca via Dataset.get)
ALARM_CATALOG_PATH = os.path.expanduser(
    "~/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9/alarmes_selecionados_turbina_a.csv"
)

ALARM_CHANNEL_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
UTILITY_TAGS = ["PI_6240319_AL", "PAL_6240315", "PAH_6240319"]  # gas/motor de partida (apontadas pelo colega)
ALARM_WINDOW_HOURS = 24.0
REFRACTORY_MINUTES = 48.0 * 60.0
MIN_VOTE_DURATION_MINUTES = 45.0


def alarm_channel_bool(index: pd.DatetimeIndex, alarm_times: pd.Series, window_hours: float) -> pd.Series:
    """Identico ao de scripts/pipeline_unificada_final.py -- duplicado aqui
    de proposito pra nao importar aquele modulo (evita puxar `clearml`
    Task/Dataset so pra usar uma funcao pura)."""
    times = np.sort(pd.DatetimeIndex(pd.Series(alarm_times).dropna()).values.astype("datetime64[ns]"))
    t_arr = index.values.astype("datetime64[ns]")
    out = np.zeros(len(t_arr), dtype=bool)
    if len(times) == 0:
        return pd.Series(out, index=index)
    pos = np.searchsorted(times, t_arr, side="right") - 1
    valid = pos >= 0
    dt_hours = (t_arr[valid] - times[pos[valid]]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    out[valid] = dt_hours <= float(window_hours)
    return pd.Series(out, index=index)


def evaluate(decisao_final: pd.Series, op_state: pd.Series, ft: pd.Series):
    df_eval = pd.DataFrame({"is_anom_point": decisao_final.astype(int), "operational_state": op_state})
    dias_vigiados = compute_operational_period_days(df_eval)
    eps = group_alerts_into_episodes(df_eval["is_anom_point"], merge_gap_minutes=120.0)
    cls = classify_episodes_regua(eps, ft, df_eval["operational_state"], 48.0, 48.0, 2.0)
    metrics = compute_regua_metrics(cls, ft, dias_vigiados)
    return cls, metrics


def detected_failures(cls: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.Timestamp(d) for d in cls.loc[cls["classe"] == "deteccao", "falha_associada"].dropna().unique())


def load_base():
    print("carregando point_anomalies_final.csv (canais 1-3 ja calculados)...", flush=True)
    df = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True, low_memory=False)
    idx = df.index
    canal_temp = df["canal_temperatura"].astype(bool)
    canal_vib = df["canal_vibracao"].astype(bool)
    canal_oleo = df["canal_oleo_pressao"].astype(bool)
    canal_alarme_full = df["canal_alarme_processo"].astype(bool)
    op_state = df["operational_state"]

    falhas = pd.read_csv(FALHAS_CSV)
    falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
    failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
    ft = failure_times.loc[(failure_times >= idx.min()) & (failure_times <= idx.max())]

    print("carregando catalogo completo de alarmes...", flush=True)
    alarm = pd.read_csv(ALARM_CATALOG_PATH)
    alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
    alarm["Tag"] = alarm["Tag Alarme"]
    alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])

    return idx, canal_temp, canal_vib, canal_oleo, canal_alarme_full, op_state, ft, alarm


def run_pipeline(canais: dict, min_votes: int, idx, min_duration_minutes: float, op_state, ft):
    voto = combine_channels_vote(canais, min_votes=min_votes)
    df_voto = pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx)
    voto_filtrado = apply_min_duration_filter(df_voto, min_duration_minutes)["is_anom_point"].astype(bool)
    decisao = apply_refractory(voto_filtrado, refractory_minutes=REFRACTORY_MINUTES)
    cls, metrics = evaluate(decisao, op_state, ft)
    return decisao, cls, metrics


def teste1_ablacao_canal4(canal_temp, canal_vib, canal_oleo, idx, op_state, ft):
    print("\n" + "=" * 70)
    print("TESTE 1 -- ablacao do canal 4 (voto so entre os 3 canais modelados)")
    print("=" * 70)
    _, cls3, metrics3 = run_pipeline(
        {"temperatura": canal_temp, "vibracao": canal_vib, "oleo_pressao": canal_oleo},
        min_votes=2, idx=idx, min_duration_minutes=MIN_VOTE_DURATION_MINUTES, op_state=op_state, ft=ft,
    )
    det3 = detected_failures(cls3)
    faltam = sorted(pd.Timestamp(f) for f in ft if not any(abs((pd.Timestamp(f) - d).total_seconds()) < 60 for d in det3))
    print(f"sem canal 4 (>=2 de 3 canais modelados): {metrics3['falhas_detectadas']}/8, "
          f"FP/mes={metrics3['falso_positivo_por_mes']:.2f}")
    print(f"falhas que dependem do canal 4 para fechar o voto ({len(faltam)}):")
    for f in faltam:
        print(" ", f.strftime("%Y-%m-%d"))
    return metrics3, faltam


def teste2_controle_negativo(alarm, idx, op_state, ft, canal_alarme_full):
    print("\n" + "=" * 70)
    print("TESTE 2 -- controle negativo por tag (enriquecimento vs acaso)")
    print("=" * 70)
    on_mask = (op_state == "on").values
    n_on = int(on_mask.sum())
    period_days = compute_operational_period_days(pd.DataFrame({"operational_state": op_state}, index=idx))
    months = period_days / 30.4375

    PRE_WINDOW_HOURS = 48.0  # mesma janela usada em classify_episodes_regua

    def stats_for(tags):
        alarm_times = alarm.loc[alarm["Tag"].isin(tags), "Data da Ocorrencia"]
        n_ativ = len(alarm_times)
        canal = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS)
        duty = float(canal.values[on_mask].mean())
        # "observado" = tag esteve ativa (proximidade de 24h) em ALGUM ponto
        # das 48h que antecedem a falha -- mesma janela pre_window_hours da
        # regua, nao so no instante exato da falha.
        n = len(ft)
        observado = 0
        for f in ft:
            f = pd.Timestamp(f)
            window = canal.loc[f - pd.Timedelta(hours=PRE_WINDOW_HOURS):f]
            if len(window) and bool(window.any()):
                observado += 1
        # esperado por acaso: duty cycle de uma janela deslizante de 48h
        # (probabilidade de qualquer ponto ter havido >=1 alarme nas 24h+48h
        # anteriores) -- aproximado pelo duty cycle de alarm_channel_bool com
        # janela ALARM_WINDOW_HOURS+PRE_WINDOW_HOURS, avaliado num ponto
        # qualquer da operacao (equivalente a "existe alarme nas ultimas 72h").
        canal_janela_cheia = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS + PRE_WINDOW_HOURS)
        duty_janela = float(canal_janela_cheia.values[on_mask].mean())
        esperado = duty_janela * n
        enriq = (observado / esperado) if esperado > 0 else float("nan")
        p = float(binom.sf(observado - 1, n, duty_janela)) if 0 < duty_janela < 1 else float("nan")
        return {
            "tags": tags, "ativacoes_mes": n_ativ / months, "duty_cycle_on": duty,
            "duty_cycle_janela_72h": duty_janela,
            "observado": observado, "n_eventos": n, "esperado_acaso": esperado,
            "enriquecimento": enriq, "p_valor": p,
        }

    rows = []
    for tag in ALARM_CHANNEL_TAGS:
        rows.append(stats_for([tag]))
    rows.append(stats_for(ALARM_CHANNEL_TAGS))

    print(f"{'tag(s)':<40} {'/mes':>7} {'obs/8':>7} {'esp/8':>7} {'enriq':>7} {'p':>7}")
    for r in rows:
        label = r["tags"][0] if len(r["tags"]) == 1 else "qualquer das 5"
        print(f"{label:<40} {r['ativacoes_mes']:>7.1f} {r['observado']:>4}/{r['n_eventos']:<2} "
              f"{r['esperado_acaso']:>7.2f} {r['enriquecimento']:>6.2f}x {r['p_valor']:>7.3f}")

    duty_full = float(canal_alarme_full.values[on_mask].mean())
    print(f"\nsanity check -- duty cycle do canal_alarme_processo ja calculado na pipeline: {duty_full:.3f} "
          f"(deve bater com a linha 'qualquer das 5' acima)")
    return rows


def teste3_remove_tags_utilidade(canal_temp, canal_vib, canal_oleo, alarm, idx, op_state, ft):
    print("\n" + "=" * 70)
    print("TESTE 3 -- removendo as tags de utilidade (gas/motor de partida) do canal 4")
    print("=" * 70)
    tags_restantes = [t for t in ALARM_CHANNEL_TAGS if t not in UTILITY_TAGS]
    print(f"tags removidas: {UTILITY_TAGS}")
    print(f"tags restantes: {tags_restantes}")
    alarm_times = alarm.loc[alarm["Tag"].isin(tags_restantes), "Data da Ocorrencia"]
    canal_alarme_reduzido = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS)
    on_mask = (op_state == "on").values
    duty = float(canal_alarme_reduzido.values[on_mask].mean())
    print(f"duty cycle do canal reduzido (2 tags): {duty:.3f}")

    _, cls, metrics = run_pipeline(
        {"temperatura": canal_temp, "vibracao": canal_vib, "oleo_pressao": canal_oleo,
         "alarme_processo_reduzido": canal_alarme_reduzido},
        min_votes=2, idx=idx, min_duration_minutes=MIN_VOTE_DURATION_MINUTES, op_state=op_state, ft=ft,
    )
    print(f"resultado: {metrics['falhas_detectadas']}/8, FP/mes={metrics['falso_positivo_por_mes']:.2f}")
    return metrics


def teste4_sweep_duracao(canal_temp, canal_vib, canal_oleo, canal_alarme_full, idx, op_state, ft):
    print("\n" + "=" * 70)
    print("TESTE 4 -- varredura fina do filtro de duracao minima na votacao (14-60min)")
    print("=" * 70)
    canais = {"temperatura": canal_temp, "vibracao": canal_vib, "oleo_pressao": canal_oleo,
              "alarme_processo": canal_alarme_full}
    voto = combine_channels_vote(canais, min_votes=2)
    df_voto = pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx)

    grid = list(range(14, 61, 1))
    resultados = []
    for dur in grid:
        voto_filtrado = apply_min_duration_filter(df_voto, float(dur))["is_anom_point"].astype(bool)
        decisao = apply_refractory(voto_filtrado, refractory_minutes=REFRACTORY_MINUTES)
        cls, metrics = evaluate(decisao, op_state, ft)
        resultados.append((dur, metrics["falhas_detectadas"], metrics["falso_positivo_por_mes"]))
        marca = "" if metrics["falhas_detectadas"] == 8 else "  <<< perde falha(s)"
        print(f"  {dur:>3}min: {metrics['falhas_detectadas']}/8  FP/mes={metrics['falso_positivo_por_mes']:.2f}{marca}")

    quebras = [r for r in resultados if r[1] < 8]
    print(f"\n{len(quebras)} de {len(grid)} valores de duracao (14-60min) NAO fecham 8/8: {[r[0] for r in quebras]}")
    return resultados


def main():
    idx, canal_temp, canal_vib, canal_oleo, canal_alarme_full, op_state, ft, alarm = load_base()
    print(f"periodo: {idx.min()} a {idx.max()}, {len(ft)} falhas catalogadas no periodo")

    out = {}
    out["ablacao"] = teste1_ablacao_canal4(canal_temp, canal_vib, canal_oleo, idx, op_state, ft)
    out["controle_negativo"] = teste2_controle_negativo(alarm, idx, op_state, ft, canal_alarme_full)
    out["remove_utilidade"] = teste3_remove_tags_utilidade(canal_temp, canal_vib, canal_oleo, alarm, idx, op_state, ft)
    out["sweep_duracao"] = teste4_sweep_duracao(canal_temp, canal_vib, canal_oleo, canal_alarme_full, idx, op_state, ft)

    out_path = os.path.join(RUN_DIR, "validacao_canal4_colega.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nresumo salvo em {out_path}")


if __name__ == "__main__":
    main()
