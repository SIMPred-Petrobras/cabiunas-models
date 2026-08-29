"""v10: fecha as duas lacunas conhecidas do v9 em relacao a formulacao
ATUAL do Francisco (`DOCUMENTACAO_DETECTOR_TC33003A.pdf`), que ainda
deixavam um gap residual (v9: 5/9 eventos a 1,51 FP/mes; config exata
dele no nosso dataset: 3/9 a 0,69 FP/mes -- ver
docs/analise_pca_monitoramento_sistema.md, secao "v9").

1. **Faixa fisica FIXA em vez de clip por quantil.** v6-v9 usavam
   `clip_outliers` (quantil 0,1%/99,9%, reajustado a cada mes/split).
   Ele usa limites fisicos absolutos e globais, aplicados uma unica vez
   sobre a serie inteira -- fora da faixa vira NaN, nao e comprimido:
       temperatura: -15 a 900 C
       pressao/diferencial: -1,5 a 120
   Isso e mais fiel (nao redefine o corte mes a mes) e menos agressivo
   nas caudas (nao comprime anomalia real, so descarta leitura
   fisicamente impossivel -- ex. termopar saturado no fim de escala).
2. **Veto de sensor congelado.** Se o desvio-padrao de um sensor fica
   zero por 30min (mesma janela do `sustain`), a familia dele (e os
   sinais derivados que usam esse sensor) tem o score anulado (flag
   forcado a 0) naquele instante -- instrumento travado nao e anomalia
   do equipamento, e antes virava falso positivo. Auto-calibrado: um
   sensor que fica congelado mais de 5% do tempo em operacao ("stable",
   aqui aproximado por `is_on`) e constante por natureza e fica fora do
   teste (ex.: `PI_0319`, pressao do gas de partida).
   Mapeamento familia -> sinal: veto de temperatura afeta os sinais
   `temperatura` e `mancal_spread` (deriva de sensores de temperatura);
   veto de pressao afeta `pressao_oleo` e `selagem_z` (deriva de
   `PDIT_0305`, um sensor de pressao).

Resto identico ao v9 (votacao 2-de-4, EWMA 2h uniforme, percentil 99,9,
grade 2min, exclude_alarm_h=1h, exclude_days=0).

Uso:
    PYTHONPATH=. python scripts/pca_monitoramento_sistema/reproducao_francisco_v10_veto_e_faixa_fisica.py
"""
import gc
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from clearml import Task, Dataset

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.preprocess import build_group_dataframe, select_feature_columns, normalize_train_only
from src.cnn1d_ae.scoring import build_operational_state

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"  # "Cabiunas full 2024-2026 30s"
CLEARML_PROJECT_NAME = "TesteMLCab"
CLEARML_DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

task = Task.init(
    project_name=CLEARML_PROJECT_NAME,
    task_name="pca-walkforward::monitoramento_sistema_v10_veto_e_faixa_fisica",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(CLEARML_DOCKER_IMAGE)

print("Resolvendo dataset ClearML...", flush=True)
_dataset_root = Dataset.get(dataset_id=CLEARML_DATASET_ID).get_local_copy()
RAW_CSV = os.path.join(_dataset_root, "sensores_full_2024_2026_30s.csv")
ALARM_CSV = os.path.join(_dataset_root, "alarmes_selecionados_turbina_a.csv")

TEMPERATURE_SENSORS = [
    "954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
    "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
    "954005_624_TI_0301",
    "TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A",
    "TC382_06_A", "T5_AVG_A",
]
PRESSURE_SENSORS = [
    "954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
    "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
    "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
    "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301",
]
MANCAL_TARGET = "954005_624_TI_0305"
MANCAL_SIBLINGS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]
SELAGEM_TARGET = "954005_624_PDIT_0305"
ALL_NEEDED_SENSORS = sorted(set(TEMPERATURE_SENSORS) | set(PRESSURE_SENSORS))

# -------- parametros da formulacao ATUAL (DOCUMENTACAO_DETECTOR_TC33003A) ---
GRID_RESAMPLE = "2min"
EWMA_HALFLIFE = "2h"       # igual pros 4 sinais
THRESHOLD_PCT = 99.9       # percentil direto do score suavizado do baseline, igual pros 4 sinais
SUSTAIN_MINUTES = 30
CONFIRM_MIN = 2            # votacao N-de-4 generica
PCA_VARIANCE = 0.95
EXCLUDE_ALARM_H = 1.0      # blackout no baseline ao redor de QUALQUER ativacao de alarme

# -------- v10: faixa fisica fixa (nao clip por quantil) ---------------------
PHYSICAL_RANGE_TEMP = (-15.0, 900.0)
PHYSICAL_RANGE_PRESS = (-1.5, 120.0)

# -------- v10: veto de sensor congelado (por familia) ------------------------
FROZEN_WINDOW_MIN = 30.0
FROZEN_NATURAL_THRESHOLD = 0.05  # sensor congelado por natureza (ex. PI_0319) fica fora do teste

# -------- ground-truth curado (algoritmo dele, replicado no nosso dataset) --
MIN_STOP_HOURS = 2.0
STOP_ALARM_BEFORE_MIN = 60
STOP_ALARM_AFTER_MIN = 30
EVENT_GAP_HOURS = 24
DETECTION_WINDOW_HOURS = 48
ALARM_LEVEL_PATTERN = r"TRIP|MT\.\s?ALTA|M\.\s?ALTA|MT\.\s?BX|M\.\s?BX|MT\.\s?BAIXA|M\.\s?BAIXA"

# -------- baseline: janela em horas ELEGIVEIS, nao calendario --------------
WINDOW_HOURS_ELIGIBLE = 3000.0
MIN_HOURS_ELIGIBLE = 100.0
EXCLUDE_DAYS_BEFORE_EVENT = 0  # DESLIGADO -- igual a config real (excl0d)

RANDOM_SEED = 42
BURN_IN_MONTHS = 2
FAST_MONTHS_LIMIT = int(os.environ.get("FAST_MONTHS_LIMIT", "0")) or None

_config = {
    "grid_resample": GRID_RESAMPLE, "ewma_halflife": EWMA_HALFLIFE,
    "threshold_pct": THRESHOLD_PCT, "sustain_minutes": SUSTAIN_MINUTES,
    "confirm_min": CONFIRM_MIN, "pca_variance": PCA_VARIANCE,
    "exclude_alarm_h": EXCLUDE_ALARM_H,
    "min_stop_hours": MIN_STOP_HOURS, "event_gap_hours": EVENT_GAP_HOURS,
    "detection_window_hours": DETECTION_WINDOW_HOURS,
    "window_hours_eligible": WINDOW_HOURS_ELIGIBLE,
    "min_hours_eligible": MIN_HOURS_ELIGIBLE,
    "exclude_days_before_event": EXCLUDE_DAYS_BEFORE_EVENT,
    "physical_range_temp": PHYSICAL_RANGE_TEMP, "physical_range_press": PHYSICAL_RANGE_PRESS,
    "frozen_window_min": FROZEN_WINDOW_MIN, "frozen_natural_threshold": FROZEN_NATURAL_THRESHOLD,
    "clearml_dataset_id": CLEARML_DATASET_ID,
}
task.connect(_config, name="reproducao_francisco_v10_config")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

cfg = PipelineConfig()
cfg.TIME_COL = "data_datetime"
cfg.TRAIN_SOURCE = "raw"
cfg.ENABLE_DERIVED_FEATURES = False  # sensor bruto limpo, igual v7 (ja validado como melhor)
cfg.DERIVED_ROLLING_WINDOWS = [12, 120, 480, 2880]
cfg.TEXTURE_SENSORS = []
cfg.OUTLIER_MODE = "quantile"
cfg.OUTLIER_Q_LOW = 0.001
cfg.OUTLIER_Q_HIGH = 0.999
cfg.NORMALIZE_MODE = "robust"
cfg.INTERPOLATE_LIMIT = 3

print("Lendo dados brutos...", flush=True)
cols = [cfg.TIME_COL, "RUNNING_A"] + ALL_NEEDED_SENSORS
df_raw = pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)
df_raw[cfg.TIME_COL] = pd.to_datetime(df_raw[cfg.TIME_COL], errors="coerce")
for c in ["RUNNING_A"] + ALL_NEEDED_SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("float32")
df_raw = df_raw.dropna(subset=[cfg.TIME_COL]).drop_duplicates(subset=[cfg.TIME_COL]).sort_values(cfg.TIME_COL).reset_index(drop=True)
print("shape bruto (30s):", df_raw.shape, flush=True)

print(f"Reamostrando pra grade de {GRID_RESAMPLE} (mediana)...", flush=True)
df_raw = df_raw.set_index(cfg.TIME_COL).resample(GRID_RESAMPLE).median().reset_index()
STEP_SECONDS = float(pd.Timedelta(GRID_RESAMPLE).total_seconds())
running_a_raw = df_raw.set_index(cfg.TIME_COL)["RUNNING_A"]
print("shape reamostrado:", df_raw.shape, "step_s:", STEP_SECONDS, flush=True)

print("Aplicando faixa fisica FIXA (fora dela vira NaN, nao clip por quantil)...", flush=True)
for c in TEMPERATURE_SENSORS:
    df_raw[c] = df_raw[c].where(df_raw[c].between(*PHYSICAL_RANGE_TEMP))
for c in PRESSURE_SENSORS:
    df_raw[c] = df_raw[c].where(df_raw[c].between(*PHYSICAL_RANGE_PRESS))

print("Construindo familia 'temperatura' (14 sensores)...", flush=True)
df_temp_use, _ = build_group_dataframe(cfg, df_raw, df_raw, TEMPERATURE_SENSORS)
feat_temp = select_feature_columns(cfg, df_temp_use, TEMPERATURE_SENSORS)
df_temp_feat = df_temp_use[feat_temp]
mancal_spread = df_temp_use[MANCAL_TARGET] - df_temp_use[MANCAL_SIBLINGS].median(axis=1)

print("Construindo familia 'pressao_oleo' (12 sensores)...", flush=True)
df_press_use, _ = build_group_dataframe(cfg, df_raw, df_raw, PRESSURE_SENSORS)
feat_press = select_feature_columns(cfg, df_press_use, PRESSURE_SENSORS)
df_press_feat = df_press_use[feat_press]
selagem_raw = df_press_use[SELAGEM_TARGET]
del df_raw
gc.collect()

FULL_INDEX = df_temp_use.index
assert FULL_INDEX.equals(df_press_use.index)
running_a = running_a_raw.reindex(FULL_INDEX).ffill().bfill()
is_running_raw = running_a >= 0.5  # "in_operation" simples, igual ao dele -- so p/ stops/eventos

print("Calculando estado operacional (on/off/transiente, nosso portao fino)...", flush=True)
operational_state = build_operational_state(
    index=FULL_INDEX, sensor_series=running_a,
    off_value_quantile=0.05, off_abs_threshold=0.5, off_long_min_hours=4.0,
    transient_padding_minutes=60, transient_diff_quantile=0.99,
)
is_on = (operational_state == "on")
print("fracao 'on':", is_on.mean(), flush=True)

FROZEN_WINDOW_SAMPLES = max(2, int(FROZEN_WINDOW_MIN * 60 / STEP_SECONDS))


def compute_family_veto(df_family: pd.DataFrame, sensors: list, stable_mask: pd.Series) -> pd.Series:
    """True onde algum sensor da familia esta congelado (desvio-padrao
    zero por FROZEN_WINDOW_MIN). Sensores congelados por natureza (mais
    de FROZEN_NATURAL_THRESHOLD do tempo 'on') ficam fora do teste."""
    rolled_std = df_family[sensors].rolling(FROZEN_WINDOW_SAMPLES, min_periods=FROZEN_WINDOW_SAMPLES).std()
    frozen = (rolled_std == 0)
    frac = frozen.loc[stable_mask.values].mean()
    naturais = frac[frac > FROZEN_NATURAL_THRESHOLD].index.tolist()
    if naturais:
        print(f"  [veto] constantes por natureza (fora do teste): {naturais}", flush=True)
    uteis = [s for s in sensors if s not in naturais]
    if not uteis:
        return pd.Series(False, index=df_family.index)
    return frozen[uteis].any(axis=1).fillna(False)


print("Calculando veto de sensor congelado (30min, por familia)...", flush=True)
veto_temp = compute_family_veto(df_temp_use, TEMPERATURE_SENSORS, is_on)
veto_press = compute_family_veto(df_press_use, PRESSURE_SENSORS, is_on)
print(f"  veto_temp ativo em {veto_temp.mean():.3%} do tempo | "
      f"veto_press ativo em {veto_press.mean():.3%} do tempo", flush=True)
VETO_BY_SIGNAL = {
    "temperatura": veto_temp, "pressao_oleo": veto_press,
    "mancal_spread": veto_temp, "selagem_z": veto_press,
}

# ------------------------------------------------ ground-truth curado (#1) --
print("Derivando paradas reais (RUNNING_A 1->0 >= %.0fh)..." % MIN_STOP_HOURS, flush=True)
off = ~is_running_raw
block = (off != off.shift(fill_value=False)).cumsum()
stops = []
for _, g in off[off].groupby(block[off]):
    t0, t1 = g.index[0], g.index[-1]
    horas = (t1 - t0).total_seconds() / 3600.0
    if horas >= MIN_STOP_HOURS:
        stops.append({"inicio": t0, "fim": t1, "horas": horas})
print(f"  {len(stops)} paradas reais >= {MIN_STOP_HOURS}h", flush=True)

alarm_all = pd.read_csv(ALARM_CSV)
if "Data da Ocorrência" in alarm_all.columns and "Data da Ocorrencia" not in alarm_all.columns:
    alarm_all["Data da Ocorrencia"] = alarm_all["Data da Ocorrência"]
if "Tag Alarme" in alarm_all.columns and "Tag" not in alarm_all.columns:
    alarm_all["Tag"] = alarm_all["Tag Alarme"]
alarm_all["Data da Ocorrencia"] = pd.to_datetime(alarm_all["Data da Ocorrencia"], errors="coerce")
alarm_all = alarm_all.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
alarm_all["ativado"] = alarm_all["Status"].astype(str).str.startswith("ACT")
alarm_all["nivel"] = alarm_all["Descrição Alarme"].astype(str).str.upper().str.contains(ALARM_LEVEL_PATTERN, regex=True)
all_alarm_times = alarm_all[alarm_all["ativado"]].set_index("Data da Ocorrencia").index
lvl_times = alarm_all[alarm_all["ativado"] & alarm_all["nivel"]].set_index("Data da Ocorrencia").index
print(f"  {len(all_alarm_times)} ativacoes ACT no catalogo, {len(lvl_times)} de nivel (TRIP/MT.ALTA/...)", flush=True)

antes = pd.Timedelta(minutes=STOP_ALARM_BEFORE_MIN)
depois = pd.Timedelta(minutes=STOP_ALARM_AFTER_MIN)
trips = []
for s in stops:
    janela = lvl_times[(lvl_times >= s["inicio"] - antes) & (lvl_times <= s["inicio"] + depois)]
    if len(janela):
        trips.append(s["inicio"])
trips = sorted(set(trips))
print(f"  {len(trips)} paradas coincidem com alarme de nivel (= trips)", flush=True)

events = []
gap = pd.Timedelta(hours=EVENT_GAP_HOURS)
for t in trips:
    if events and (t - events[-1]["fim"]) <= gap:
        events[-1]["fim"] = t
    else:
        events.append({"inicio": t, "fim": t})
print(f"  {len(events)} eventos fisicos (trips agrupados por <{EVENT_GAP_HOURS}h)", flush=True)
for e in events:
    print("   -", e["inicio"], flush=True)

lvl_times_on = [t for t in lvl_times if bool(is_running_raw.asof(t))] if len(lvl_times) else []
exclusion_events = sorted(set(e["inicio"] for e in events) | set(lvl_times_on))
print(f"  {len(exclusion_events)} eventos usados p/ limpar o baseline (trips + alarme de nivel em operacao)", flush=True)

# ------------------------------------ elegibilidade do baseline -------------
keep = pd.Series(True, index=FULL_INDEX)
if EXCLUDE_DAYS_BEFORE_EVENT > 0:
    delta_excl = pd.Timedelta(days=EXCLUDE_DAYS_BEFORE_EVENT)
    for ev in exclusion_events:
        keep.loc[(FULL_INDEX >= ev - delta_excl) & (FULL_INDEX <= ev)] = False

if EXCLUDE_ALARM_H > 0 and len(all_alarm_times):
    # distancia ao alarme mais proximo por busca binaria (igual ao dele) --
    # a matriz completa (amostras x alarmes) nao cabe em memoria.
    marcas = np.sort(all_alarm_times.values.astype("datetime64[ns]"))
    alvo = FULL_INDEX.values.astype("datetime64[ns]")
    pos_a = np.searchsorted(marcas, alvo)
    esquerda = marcas[np.clip(pos_a - 1, 0, len(marcas) - 1)]
    direita = marcas[np.clip(pos_a, 0, len(marcas) - 1)]
    dist = np.minimum(np.abs(alvo - esquerda), np.abs(alvo - direita))
    near_alarm = dist <= np.timedelta64(int(EXCLUDE_ALARM_H * 3600), "s")
    keep &= ~pd.Series(near_alarm, index=FULL_INDEX)

eligible_mask = is_on.values & keep.values
eligible_idx = FULL_INDEX[eligible_mask]
print(f"fracao elegivel p/ baseline (on & fora de exclusao): {eligible_mask.mean():.3f} "
      f"({len(eligible_idx)} amostras elegiveis no total)", flush=True)

K_WINDOW = int(WINDOW_HOURS_ELIGIBLE * 3600 / STEP_SECONDS)
K_MIN = int(MIN_HOURS_ELIGIBLE * 3600 / STEP_SECONDS)
print(f"janela de baseline: {WINDOW_HOURS_ELIGIBLE}h elegiveis = {K_WINDOW} amostras "
      f"(piso {MIN_HOURS_ELIGIBLE}h = {K_MIN} amostras)", flush=True)

months = pd.period_range(FULL_INDEX.min(), FULL_INDEX.max(), freq="M")
print("total de meses no periodo:", len(months), flush=True)

# Os 4 sinais tratados de forma IGUAL: "pca" reduz multivariado via PCA-Q;
# "raw" e o proprio valor (com sinal) do sensor/spread univariado -- ambos
# passam pelo MESMO mecanismo depois (EWMA -> percentil do baseline -> sustentacao).
SIGNAL_SPECS = {
    "temperatura": {"kind": "pca", "feat": df_temp_feat},
    "pressao_oleo": {"kind": "pca", "feat": df_press_feat},
    "mancal_spread": {"kind": "raw", "raw": mancal_spread},
    "selagem_z": {"kind": "raw", "raw": selagem_raw},
}
HALFLIFE = pd.Timedelta(EWMA_HALFLIFE)
DEBOUNCE = int(SUSTAIN_MINUTES * 60 / STEP_SECONDS)

raw_flags = {name: pd.Series(0, index=FULL_INDEX, dtype="int8") for name in SIGNAL_SPECS}
scored_mask = pd.Series(False, index=FULL_INDEX)

n_scored = 0
n_truncados = 0
for i, m in enumerate(months):
    if i < BURN_IN_MONTHS:
        continue
    if FAST_MONTHS_LIMIT and n_scored >= FAST_MONTHS_LIMIT:
        print(f"[{m}] parado (FAST_MONTHS_LIMIT={FAST_MONTHS_LIMIT})", flush=True)
        break

    month_start = m.start_time
    month_end = m.end_time + pd.Timedelta(seconds=int(STEP_SECONDS))

    pos = int(np.searchsorted(eligible_idx.values, np.datetime64(month_start), side="left"))
    train_idx_dt = eligible_idx[max(0, pos - K_WINDOW):pos]
    n_train = len(train_idx_dt)
    truncado = n_train < K_WINDOW
    if truncado:
        n_truncados += 1
    eval_idx = (FULL_INDEX >= month_start) & (FULL_INDEX < month_end)
    n_eval = int(eval_idx.sum())
    if n_train < K_MIN or n_eval < 20:
        print(f"[{m}] pulado (n_train={n_train} n_eval={n_eval})", flush=True)
        continue

    flag_counts = {}
    for name, spec in SIGNAL_SPECS.items():
        if spec["kind"] == "pca":
            df_feat = spec["feat"]
            df_train = df_feat.loc[train_idx_dt]
            df_eval = df_feat.loc[eval_idx]
            # v10: sem clip_outliers por quantil -- a faixa fisica fixa
            # (aplicada uma vez, globalmente, logo apos ler os dados brutos)
            # ja tratou os valores fisicamente impossiveis.
            df_train_z, df_eval_z, _, _ = normalize_train_only(cfg, df_train, df_eval)
            X_train = np.nan_to_num(df_train_z.values, nan=0.0, posinf=0.0, neginf=0.0)
            X_eval = np.nan_to_num(df_eval_z.values, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_SEED)
            X_train_lat = pca.fit_transform(X_train)
            err_train = np.mean((X_train - pca.inverse_transform(X_train_lat)) ** 2, axis=1)
            X_eval_lat = pca.transform(X_eval)
            err_eval = np.mean((X_eval - pca.inverse_transform(X_eval_lat)) ** 2, axis=1)

            s_train = pd.Series(err_train, index=train_idx_dt)
            eval_dt = df_feat.index[eval_idx]
            s_eval = pd.Series(err_eval, index=eval_dt)
        else:  # "raw" -- sensor/spread univariado, mesmo mecanismo, sem PCA
            raw_series = spec["raw"]
            s_train = raw_series.loc[train_idx_dt]
            eval_dt = raw_series.loc[eval_idx].index
            s_eval = raw_series.loc[eval_idx]

        # suavizacao (EWMA 2h) aplicada IGUAL ao treino e ao teste, depois
        # limiar = percentil 99,9 do baseline JA suavizado (mesma ordem do dele).
        s_train_sm = s_train.ewm(halflife=HALFLIFE, times=s_train.index).mean()
        s_eval_sm = s_eval.ewm(halflife=HALFLIFE, times=s_eval.index).mean()
        thr = float(np.percentile(s_train_sm.values, THRESHOLD_PCT))
        flag = (s_eval_sm > thr).astype("int8")
        flag.index = eval_dt

        # v10: veto de sensor congelado -- anula o flag onde a familia
        # (ou o sensor de origem, pros sinais univariados) esta travada.
        veto_series = VETO_BY_SIGNAL[name].reindex(flag.index).fillna(False)
        flag = (flag.astype(bool) & ~veto_series).astype("int8")

        raw_flags[name].loc[flag.index] = flag.values
        flag_counts[name] = int(flag.sum())

    scored_mask.loc[eval_idx] = True
    n_scored += 1
    print(f"[{m}] n_train={n_train}({'trunc' if truncado else 'full'}) n_eval={n_eval} flags_brutos={flag_counts}", flush=True)

print(f"\nmeses truncados (baseline < {WINDOW_HOURS_ELIGIBLE}h elegiveis): {n_truncados}/{n_scored}", flush=True)

print("\nAplicando sustentacao (%d min, all-of-window) por sinal..." % SUSTAIN_MINUTES, flush=True)
sustained = {}
for name in SIGNAL_SPECS:
    hits = raw_flags[name]
    sust = hits.rolling(DEBOUNCE, min_periods=DEBOUNCE).sum() >= DEBOUNCE
    sust = sust.fillna(False) & scored_mask & is_on
    sustained[name] = sust
    print(f"  [{name}] pontos sustentados: {int(sust.sum())} de {int(scored_mask.sum())} avaliados", flush=True)

vote_count = sum(sustained[n].astype(int) for n in SIGNAL_SPECS)
alert_votacao = (vote_count >= CONFIRM_MIN) & scored_mask & is_on  # RESULTADO PRINCIPAL (formulacao atual)

first_scored_month = months[BURN_IN_MONTHS].start_time
vigiando = scored_mask & is_on
dias_vigiados_total = float(vigiando.sum()) * STEP_SECONDS / 86400.0


def extract_episodes(alert: pd.Series) -> list[pd.Timestamp]:
    a = alert.fillna(False)
    starts = a & ~a.shift(fill_value=False)
    return list(a.index[starts.values])


def evaluate_style_dele(name: str, alert: pd.Series):
    """FP/deteccao no estilo do Francisco: contra eventos curados, com
    exclusao de 48h ao redor de qualquer evento curado OU parada real."""
    episodios = extract_episodes(alert)
    detected, leads, matched = {}, {}, set()
    janela = pd.Timedelta(hours=DETECTION_WINDOW_HOURS)
    for ev in events:
        w0 = ev["inicio"] - janela
        inside = [ep for ep in episodios if w0 <= ep <= ev["inicio"]]
        if inside:
            lead_h = (ev["inicio"] - inside[0]).total_seconds() / 3600.0
            detected[str(ev["inicio"])] = True
            leads[str(ev["inicio"])] = round(lead_h, 1)
            matched.update(inside)

    fps = []
    for ep in episodios:
        if ep in matched:
            continue
        horizon = ep + janela
        near_evento = any(ep <= e["inicio"] <= horizon or e["inicio"] <= ep <= e["inicio"] + janela for e in events)
        near_parada = any(ep <= s["inicio"] <= horizon or s["inicio"] <= ep <= s["inicio"] + janela for s in stops)
        if not (near_evento or near_parada):
            fps.append(ep)

    fp_mes = len(fps) / max(dias_vigiados_total, 1.0) * 30
    lead_medio = float(np.mean(list(leads.values()))) if leads else None
    print(f"\n=== {name} (estilo Francisco: eventos curados) ===")
    print(f"eventos detectados: {len(detected)}/{len(events)}  lead medio: {lead_medio}")
    print(f"episodios de alerta: {len(episodios)}  FP: {len(fps)}  FP/mes: {fp_mes:.2f}")
    return {
        "eventos_detectados": len(detected), "eventos_total": len(events),
        "lead_medio_h": lead_medio if lead_medio is not None else 0.0,
        "n_episodios": len(episodios), "n_fp": len(fps), "fp_por_mes": float(fp_mes),
    }


print("\n\n########## RESULTADO ESTILO FRANCISCO (eventos curados) ##########")
for name, sig in list(sustained.items()) + [("votacao_2de4_ATUAL", alert_votacao)]:
    res = evaluate_style_dele(name, sig)
    task.get_logger().report_scalar(title="fr_eventos_detectados_pct", series=name,
                                     value=100.0 * res["eventos_detectados"] / max(res["eventos_total"], 1), iteration=0)
    task.get_logger().report_scalar(title="fr_lead_medio_h", series=name, value=res["lead_medio_h"], iteration=0)
    task.get_logger().report_scalar(title="fr_fp_por_mes", series=name, value=res["fp_por_mes"], iteration=0)

for vote_min in [2, 3, 4]:
    sig = (vote_count >= vote_min) & scored_mask & is_on
    name = f"votacao_{vote_min}de4"
    res = evaluate_style_dele(name, sig)
    task.get_logger().report_scalar(title="fr_eventos_detectados_pct", series=name,
                                     value=100.0 * res["eventos_detectados"] / max(res["eventos_total"], 1), iteration=0)
    task.get_logger().report_scalar(title="fr_lead_medio_h", series=name, value=res["lead_medio_h"], iteration=0)
    task.get_logger().report_scalar(title="fr_fp_por_mes", series=name, value=res["fp_por_mes"], iteration=0)

# ---------------- avaliacao no NOSSO estilo (catalogo completo), p/ continuidade
from src.cnn1d_ae.scoring import map_seq_to_point_anomalies, eval_alarm_hit_rate
from src.cnn1d_ae.preprocess import build_exclusion_mask

EXCLUDE_MIN_ALARM = 1440
near_any_alarm_full_catalog = build_exclusion_mask(FULL_INDEX, pd.Series(all_alarm_times), EXCLUDE_MIN_ALARM)
scored_alarms_full = alarm_all[alarm_all["ativado"] & (alarm_all["Data da Ocorrencia"] >= first_scored_month)]
fp_mask_full = (~near_any_alarm_full_catalog) & scored_mask.values & is_on.values

ep_gaps = scored_alarms_full["Data da Ocorrencia"].diff().dt.total_seconds().fillna(99999) / 60.0
ep_id = (ep_gaps > 60).cumsum()
episodes_cat = scored_alarms_full.groupby(ep_id)["Data da Ocorrencia"].agg(["min", "max"])
n_episodes_cat = len(episodes_cat)


def evaluate_style_nosso(name: str, is_anom_point: pd.Series):
    point_df_local = pd.DataFrame({"is_anom_point": is_anom_point.astype(int)})
    fp_rate = point_df_local.loc[fp_mask_full, "is_anom_point"].mean()
    hits = 0
    for _, row in episodes_cat.iterrows():
        t0 = row["min"] - pd.Timedelta(minutes=EXCLUDE_MIN_ALARM)
        t1 = row["max"] + pd.Timedelta(minutes=EXCLUDE_MIN_ALARM)
        if point_df_local.loc[t0:t1, "is_anom_point"].sum() > 0:
            hits += 1
    hit_rate_ep = (hits / n_episodes_cat * 100) if n_episodes_cat else 0.0
    print(f"[{name}] (estilo nosso, catalogo completo) FP%={fp_rate*100:.3f}  hit_rate_episodio={hit_rate_ep:.1f}%")
    return fp_rate, hit_rate_ep


print("\n\n########## RESULTADO ESTILO NOSSO (catalogo completo, p/ continuidade v2-v8) ##########")
for name, sig in list(sustained.items()) + [("votacao_2de4_ATUAL", alert_votacao)]:
    fp_rate, hit_ep = evaluate_style_nosso(name, sig)
    task.get_logger().report_scalar(title="fp_rate_geral_pct_pontos", series=name, value=float(fp_rate) * 100, iteration=0)
    task.get_logger().report_scalar(title="hit_rate_por_episodio_pct", series=name, value=hit_ep, iteration=0)
for vote_min in [2, 3, 4]:
    sig = (vote_count >= vote_min) & scored_mask & is_on
    name = f"votacao_{vote_min}de4"
    fp_rate, hit_ep = evaluate_style_nosso(name, sig)
    task.get_logger().report_scalar(title="fp_rate_geral_pct_pontos", series=name, value=float(fp_rate) * 100, iteration=0)
    task.get_logger().report_scalar(title="hit_rate_por_episodio_pct", series=name, value=hit_ep, iteration=0)

print("\nOK - fim do script v10 (veto de sensor congelado + faixa fisica fixa)", flush=True)
task.mark_completed(status_message="v10: veto de sensor congelado + faixa fisica fixa concluido com sucesso.")
task.close()
