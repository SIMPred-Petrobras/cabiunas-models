"""
sensor_preprocessing.py — Pré-processamento universal para sensores
Versão: v6-ngp

Novidades em relação à v5-auditado:

  [1] REGIME_COL configurável — RUNNING_A ou NGP_A
      Na hora de rodar o CNN1D_AE_universal.py você escolhe qual
      variável define o regime operacional:

        --regime_col RUNNING_A          (padrão, comportamento v5)
        --regime_col NGP_A              (usa velocidade da turbina)
        --regime_col NGP_A --ngp_min 60 (treina apenas com NGP >= 60%)

      A variável de regime pode ser qualquer coluna numérica do CSV.
      O preprocessing exporta regime_bin, regime_steady_mask e
      regime_transition_mask prontos para uso.

  [2] LIMIARES FÍSICOS por sensor (SENSOR_LIMITS)
      Baseados nos documentos oficiais:
        - TV_351/352/353 (mancais 1-3 turbina GG): H=114, HH=125 µm
        - TV_354/355     (mancais 4-5 turbina PT): H=50,  HH=64  µm
        - TC382_*/T5_AVG (temperatura T5):
            NGP > 60%: H=760, HH=788 °C
            NGP < 60%: H=649, HH=677 °C  (limiar dinâmico)
        - TI mancais compressor:  H=120, HH=125 °C
        - TI T7 exaustão:         H=580, HH=600 °C
        - PI_0315 gás combustível: LL=15.8, L=16.9, H=21.1, HH=21.8 kgf/cm2
        - PDI_0317 delta-P gás:   L=0.11, H=1.76 kgf/cm2
        - PDIT_0305 selagem:      H=2.5, HH=6.0 kgf/cm2
        - PDI_0301 filtro:        H=1.75 kgf/cm2
        - PDI_0302 selagem prim:  L=1.1 kgf/cm2
        - PI_5134001 ar barreira: L=4.0 kgf/cm2

  [3] SCORING EM 3 NÍVEIS exportado como função
      get_physical_level(sensor, value, ngp=None) → "normal" | "alerta" | "critico"
      Usado pelo CNN1D_AE_universal.py para enriquecer o CSV de anomalias.

  [4] Compatibilidade total com v5
      Todos os EXPORTS da v5 mantidos. Novos itens adicionados:
        regime_bin, regime_col_used, get_physical_level,
        sensor_limits, ngp_series

  Configuração de regime no sensor_preprocessing.py (constante CONFIG):
    REGIME_COL   = "RUNNING_A"   # ou "NGP_A"
    NGP_MIN_PCT  = 60.0          # limiar mínimo de NGP para steady state
    NGP_COL      = "NGP_A"       # coluna do NGP no CSV (para limiares dinâmicos)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


# =========================================================
# CONFIG PRINCIPAL — ajuste caminhos aqui
# =========================================================

# ── ID do dataset no ClearML (CABIUNAS_DATA) ──────────────
# Lido de os.environ["CLEARML_DATASET_ID"] (definido pelo CNN1D_AE_universal.py
# antes de exec_module). Se None, usa os caminhos locais abaixo.
CLEARML_DATASET_ID = os.getenv("CLEARML_DATASET_ID", None)

# ── Caminhos locais (usados se CLEARML_DATASET_ID for None) ──
DATA_CSV   = r"dados_novos_30s_juntos/dados_geral_2025_30s.csv"
ALARM_CSV  = r"alarmes_filtrados/ocorrencias_alarmes_sensores_2025.csv"
OUTPUT_DIR = r"OUTPUT_PREPROCESSING_V6"

RUNNING_COL = "RUNNING_A"
NGP_COL     = "NGP_A"          # coluna do NGP no CSV (se disponível)
NGP_MIN_PCT = float(os.getenv("NGP_MIN_PCT", "60.0"))  # NGP mínimo para steady state

# ── Escolha do regime ─────────────────────────────────────
# Lido de os.environ["REGIME_COL"] (definido pelo CNN1D_AE_universal.py).
# "RUNNING_A" → usa RUNNING_A == 1 (padrão) | "NGP_A" → usa NGP_A >= NGP_MIN_PCT
REGIME_COL = os.getenv("REGIME_COL", "RUNNING_A")

# ── Download automático do dataset ClearML ────────────────
def _download_clearml_dataset(dataset_id: str) -> str:
    """
    Baixa o dataset CABIUNAS_DATA do servidor ClearML e retorna
    o caminho local onde os arquivos foram salvos.
    Funciona em qualquer máquina com acesso ao servidor CICA.
    """
    try:
        from clearml import Dataset
        print(f"[CLEARML-DATA] Baixando dataset id={dataset_id}...")
        dataset = Dataset.get(dataset_id=dataset_id)
        local_path = dataset.get_local_copy()
        print(f"[CLEARML-DATA] Dataset disponível em: {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Falha ao baixar dataset ClearML id={dataset_id}: {e}\n"
            f"Verifique se o clearml está configurado e o servidor CICA está acessível."
        )

# Resolve caminhos: ClearML ou local
if CLEARML_DATASET_ID:
    _dataset_root = _download_clearml_dataset(CLEARML_DATASET_ID)
    import pathlib as _pl
    # Busca os arquivos dentro do dataset baixado
    _root = _pl.Path(_dataset_root)
    _csv_candidates = list(_root.rglob("dados_geral_2025_30s.csv"))
    _alarm_candidates = list(_root.rglob("ocorrencias_alarmes_sensores_2025.csv"))
    if not _csv_candidates:
        raise FileNotFoundError(
            f"dados_geral_2025_30s.csv não encontrado no dataset {CLEARML_DATASET_ID}.\n"
            f"Conteúdo: {list(_root.rglob('*.csv'))}"
        )
    DATA_CSV  = str(_csv_candidates[0])
    ALARM_CSV = str(_alarm_candidates[0]) if _alarm_candidates else ALARM_CSV
    print(f"[CLEARML-DATA] DATA_CSV  = {DATA_CSV}")
    print(f"[CLEARML-DATA] ALARM_CSV = {ALARM_CSV}")

# =========================================================
# PARÂMETROS PADRÃO
# =========================================================
RUNNING_STEADY_VALUES_DEFAULT   = {1}
TRANSITION_MASK_MINUTES_DEFAULT = 720
GRAD_STD_MULT_DEFAULT           = 10.0
GRAD_SUPPRESS_MINUTES_DEFAULT   = 60
EXCLUDE_ALARM_MINUTES           = 420

# =========================================================
# LIMIARES FÍSICOS POR SENSOR  ← NOVO v6
# Fonte: Turbinas_Limites_Alertas_TAGs_das_variaveis.xlsx
#        limites_sensores_filtrados_2025.csv
# =========================================================
SENSOR_LIMITS = {
    # ── TV_* Mancais 1-3 (turbina GG): H=114 µm, HH=125 µm ──────────
    "TV_351X_A": {"desc": "Vibração Mancal 1 eixo X", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 1, "grupo_vib": "mancal_1_3"},
    "TV_351Y_A": {"desc": "Vibração Mancal 1 eixo Y", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 1, "grupo_vib": "mancal_1_3"},
    "TV_352X_A": {"desc": "Vibração Mancal 2 eixo X", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 2, "grupo_vib": "mancal_1_3"},
    "TV_352Y_A": {"desc": "Vibração Mancal 2 eixo Y", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 2, "grupo_vib": "mancal_1_3"},
    "TV_353X_A": {"desc": "Vibração Mancal 3 eixo X", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 3, "grupo_vib": "mancal_1_3"},
    "TV_353Y_A": {"desc": "Vibração Mancal 3 eixo Y", "unit": "µm",
                  "LL": None, "L": None, "H": 114, "HH": 125,
                  "mancal": 3, "grupo_vib": "mancal_1_3"},
    # ── TV_* Mancais 4-5 (turbina PT): H=50 µm, HH=64 µm ────────────
    "TV_354X_A": {"desc": "Vibração Mancal 4 eixo X", "unit": "µm",
                  "LL": None, "L": None, "H": 50, "HH": 64,
                  "mancal": 4, "grupo_vib": "mancal_4_5"},
    "TV_354Y_A": {"desc": "Vibração Mancal 4 eixo Y", "unit": "µm",
                  "LL": None, "L": None, "H": 50, "HH": 64,
                  "mancal": 4, "grupo_vib": "mancal_4_5"},
    "TV_355X_A": {"desc": "Vibração Mancal 5 eixo X", "unit": "µm",
                  "LL": None, "L": None, "H": 50, "HH": 64,
                  "mancal": 5, "grupo_vib": "mancal_4_5"},
    "TV_355Y_A": {"desc": "Vibração Mancal 5 eixo Y", "unit": "µm",
                  "LL": None, "L": None, "H": 50, "HH": 64,
                  "mancal": 5, "grupo_vib": "mancal_4_5"},
    # ── TC382/T5: limiares dinâmicos por NGP ─────────────────────────
    # NGP > 60%: H=760, HH=788 | NGP < 60%: H=649, HH=677
    "TC382_01_A": {"desc": "Temperatura T5 ponto 1", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "TC382_02_A": {"desc": "Temperatura T5 ponto 2", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "TC382_03_A": {"desc": "Temperatura T5 ponto 3", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "TC382_04_A": {"desc": "Temperatura T5 ponto 4", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "TC382_05_A": {"desc": "Temperatura T5 ponto 5", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "TC382_06_A": {"desc": "Temperatura T5 ponto 6", "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    "T5_AVG_A":   {"desc": "Temperatura T5 média",   "unit": "°C",
                   "LL": None, "L": None, "H": 760, "HH": 788,
                   "H_low_ngp": 649, "HH_low_ngp": 677, "ngp_threshold": 60},
    # ── TI compressor (mancais) ───────────────────────────────────────
    "954005_624_TI_0301": {"desc": "Temp Mancal Escora Ativo",  "unit": "°C",
                           "LL": None, "L": None, "H": 120, "HH": 125},
    "954005_624_TI_0303": {"desc": "Temp Mancal Escora Inativo","unit": "°C",
                           "LL": None, "L": None, "H": 120, "HH": 125},
    "954005_624_TI_0305": {"desc": "Temp Mancal Radial LNA",    "unit": "°C",
                           "LL": None, "L": None, "H": 120, "HH": 125},
    "954005_624_TI_0307": {"desc": "Temp Mancal Radial LA",     "unit": "°C",
                           "LL": None, "L": None, "H": 120, "HH": 125},
    # ── TI T7 exaustão ────────────────────────────────────────────────
    "954005_624_TI_0315": {"desc": "Temp T7 exaustão ponto 1",  "unit": "°C",
                           "LL": None, "L": None, "H": 580, "HH": 600},
    "954005_624_TI_0317": {"desc": "Temp T7 exaustão ponto 2",  "unit": "°C",
                           "LL": None, "L": None, "H": 580, "HH": 600},
    # ── TI entrada de ar ─────────────────────────────────────────────
    "954005_624_TI_0325": {"desc": "Temp entrada ar",           "unit": "°C",
                           "LL": None, "L": None, "H": 35, "HH": 40},
    # ── PI gás combustível ────────────────────────────────────────────
    "954005_624_PI_0315": {"desc": "Pressão gás combustível",   "unit": "kgf/cm2",
                           "LL": 15.8, "L": 16.9, "H": 21.1, "HH": 21.8},
    "954005_624_PI_0319": {"desc": "Pressão gás motor partida", "unit": "kgf/cm2",
                           "LL": None, "L": None, "H": 16.5, "HH": None},
    # ── PDI / PDIT compressor ─────────────────────────────────────────
    "954005_624_PDI_0317":  {"desc": "Delta-P gás combustível e PCD","unit": "kgf/cm2",
                             "LL": None, "L": 0.11, "H": 1.76, "HH": None},
    "954005_624_PDIT_0305": {"desc": "Pressão vazamento gás selagem","unit": "kgf/cm2",
                             "LL": None, "L": None, "H": 2.5, "HH": 6.0},
    "954005_624_PDI_0301":  {"desc": "Delta-P filtro gás selagem",   "unit": "kgf/cm2",
                             "LL": None, "L": None, "H": 1.75, "HH": None},
    "954005_624_PDI_0302":  {"desc": "Delta-P gás selagem primária", "unit": "kgf/cm2",
                             "LL": None, "L": 1.1, "H": None, "HH": None},
    # ── PI ar de barreira ─────────────────────────────────────────────
    "PI_5134001":           {"desc": "Pressão header ar barreira",   "unit": "kgf/cm2",
                             "LL": None, "L": 4.0, "H": None, "HH": None},
}


# =========================================================
# CONFIGURAÇÃO POR GRUPO (igual v5)
# =========================================================
GROUP_TC = {
    "sensors": [
        "TC382_01_A", "TC382_02_A", "TC382_03_A",
        "TC382_04_A", "TC382_05_A", "TC382_06_A",
        "T5_AVG_A",
    ],
    "running_steady_values":   {1},
    "transition_mask_minutes": 240,
    "grad_std_mult":           12.0,
    "grad_suppress_minutes":   90,
}

GROUP_TV = {
    "sensors": [
        "TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A",
        "TV_353X_A", "TV_353Y_A", "TV_354X_A", "TV_354Y_A",
        "TV_355X_A", "TV_355Y_A",
    ],
    "running_steady_values":   {0, 1},
    "transition_mask_minutes": 0,
    "grad_std_mult":           15.0,
    "grad_suppress_minutes":   30,
}

GROUP_PI_HIGHSPIKE = {
    "sensors": [
        "954005_624_PI_0340",
        "954005_624_PI_0339",
        "954005_624_PI_0308",
    ],
    "running_steady_values":   {1},
    "transition_mask_minutes": 720,
    "grad_std_mult":           20.0,
    "grad_suppress_minutes":   30,
}

GROUP_PDI_LOWAMP = {
    "sensors": [
        "954005_624_PDI_0302",
        "954005_624_PDI_0317",
        "954005_624_PDI_0338",
        "954005_624_PDI_0301",
        "954005_624_PDIT_0305",
    ],
    "running_steady_values":   {1},
    "transition_mask_minutes": 720,
    "grad_std_mult":           20.0,
    "grad_suppress_minutes":   30,
}

GROUP_TI = {
    "sensors": [
        "954005_624_TI_0325",
        "954005_624_TI_0315",
        "954005_624_TI_0317",
        "954005_624_TI_0305",
        "954005_624_TI_0307",
        "954005_624_TI_0303",
        "954005_624_TI_0301",
    ],
    "running_steady_values":   {1},
    "transition_mask_minutes": 720,
    "grad_std_mult":           10.0,
    "grad_suppress_minutes":   60,
}

ALL_GROUPS = [GROUP_TC, GROUP_TV, GROUP_PI_HIGHSPIKE, GROUP_PDI_LOWAMP, GROUP_TI]

GROUP_NAMES = {
    id(GROUP_TC):           "TC/T5",
    id(GROUP_TV):           "TV",
    id(GROUP_PI_HIGHSPIKE): "PI_highspike",
    id(GROUP_PDI_LOWAMP):   "PDI_lowamp",
    id(GROUP_TI):           "TI",
}

SENSOR_TO_GROUP = {}
for _g in ALL_GROUPS:
    for _s in _g["sensors"]:
        SENSOR_TO_GROUP[_s] = _g

SENSOR_CLASSES = {
    "HSX_6240001A":        "binary",
    "954005_624_PI_0315":  "low_signal",
    "PI_5134001":          "low_signal",
}

VIBRATION_SENSORS = set(GROUP_TV["sensors"])

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# 1. LEITURA
# ─────────────────────────────────────────────────────────
print("[LOAD] Lendo dados...")

df_raw = pd.read_csv(DATA_CSV)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw = (df_raw.dropna(subset=["data_datetime"])
                .sort_values("data_datetime")
                .reset_index(drop=True)
                .set_index("data_datetime"))

if os.path.exists(ALARM_CSV):
    df_alarm = pd.read_csv(ALARM_CSV, sep=";")
    for col_cand in ["data_datetime", "Data da Ocorrencia"]:
        if col_cand in df_alarm.columns:
            df_alarm["data_datetime"] = pd.to_datetime(
                df_alarm[col_cand], format="mixed", dayfirst=True, errors="coerce"
            )
            break
    df_alarm = (df_alarm.dropna(subset=["data_datetime"])
                        .sort_values("data_datetime")
                        .reset_index(drop=True))
    if "Tag Alarme" not in df_alarm.columns:
        df_alarm["Tag Alarme"] = ""
    if "Status" not in df_alarm.columns:
        df_alarm["Status"] = "ACT/UNACK"
    print(f"[LOAD] Alarmes: {len(df_alarm)} | "
          f"ACT/UNACK: {(df_alarm['Status']=='ACT/UNACK').sum()} | "
          f"Tags: {df_alarm['Tag Alarme'].nunique()}")
else:
    df_alarm = pd.DataFrame(columns=["data_datetime", "Tag Alarme", "Status"])
    print("[LOAD] Sem arquivo de alarmes.")

sensors = [c for c in df_raw.columns if c != RUNNING_COL]
print(f"[LOAD] Sensores: {len(sensors)} | Linhas: {len(df_raw):,} | "
      f"{df_raw.index.min()} → {df_raw.index.max()}")

# Verifica se NGP está disponível
ngp_available = NGP_COL in df_raw.columns
if ngp_available:
    print(f"[LOAD] NGP_A disponível ✓ | faixa: {df_raw[NGP_COL].min():.1f}% – {df_raw[NGP_COL].max():.1f}%")
else:
    print(f"[LOAD] NGP_A NÃO encontrado no CSV — limiares dinâmicos T5 usarão regime fixo")


# ─────────────────────────────────────────────────────────
# 2. RUNNING_A (sempre extraído antes da interpolação)
# ─────────────────────────────────────────────────────────
print("\n[RUNNING] Extraindo RUNNING_A e NGP_A...")

running_raw = df_raw[RUNNING_COL].copy()
running_bin_series = running_raw.round().astype(int).isin(RUNNING_STEADY_VALUES_DEFAULT).astype(int)
running_bin_series.name = "running_bin"

print(f"[RUNNING] RUNNING=1: {running_bin_series.sum():,} ({100*running_bin_series.mean():.1f}%) | "
      f"Modos: {sorted(running_raw.round().astype(int).unique())}")

# NGP série (antes da interpolação)
if ngp_available:
    ngp_raw = df_raw[NGP_COL].copy()
    ngp_bin_series = (ngp_raw >= NGP_MIN_PCT).astype(int)
    ngp_bin_series.name = "ngp_bin"
    print(f"[NGP]    NGP>={NGP_MIN_PCT:.0f}%: {ngp_bin_series.sum():,} "
          f"({100*ngp_bin_series.mean():.1f}%)")
else:
    ngp_raw        = pd.Series(np.nan, index=df_raw.index, name=NGP_COL)
    ngp_bin_series = running_bin_series.copy()
    ngp_bin_series.name = "ngp_bin"


# ─────────────────────────────────────────────────────────
# 3. INTERPOLAÇÃO
# ─────────────────────────────────────────────────────────
print("\n[INTERP] Interpolando sensores...")

_cols_to_interp = [c for c in df_raw.columns
                   if c not in [RUNNING_COL, NGP_COL]]
df = df_raw[_cols_to_interp].copy()
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.interpolate(method="time", limit_direction="both").ffill().bfill()

# NGP interpolado (para limiares dinâmicos em scoring)
if ngp_available:
    ngp_interp = df_raw[NGP_COL].copy()
    ngp_interp = pd.to_numeric(ngp_interp, errors="coerce")
    ngp_interp = ngp_interp.interpolate(method="time", limit_direction="both").ffill().bfill()
else:
    ngp_interp = pd.Series(np.nan, index=df.index, name=NGP_COL)

print(f"[INTERP] NaN restantes: {df.isna().sum().sum()}")


# ─────────────────────────────────────────────────────────
# 4. REGIME BIN — RUNNING_A ou NGP_A  ← NOVO v6
# ─────────────────────────────────────────────────────────
print(f"\n[REGIME] Usando regime_col='{REGIME_COL}'...")

if REGIME_COL == NGP_COL and ngp_available:
    regime_bin = ngp_bin_series.reindex(df.index, fill_value=0)
    regime_label = f"NGP>={NGP_MIN_PCT:.0f}%"
elif REGIME_COL == RUNNING_COL:
    regime_bin = running_bin_series.reindex(df.index, fill_value=0)
    regime_label = "RUNNING_A=1"
else:
    print(f"[REGIME] WARN: '{REGIME_COL}' não reconhecido ou indisponível. "
          f"Usando RUNNING_A como fallback.")
    regime_bin = running_bin_series.reindex(df.index, fill_value=0)
    regime_label = "RUNNING_A=1 (fallback)"

print(f"[REGIME] {regime_label}: {regime_bin.sum():,} pts ({100*regime_bin.mean():.1f}%)")

# Sempre manter também o RUNNING_A puro (para supressão de TV_*)
running_bin_idx = running_bin_series.reindex(df.index, fill_value=0)
ngp_idx         = ngp_interp.reindex(df.index, fill_value=np.nan)


# ─────────────────────────────────────────────────────────
# 5. MÁSCARAS DE TRANSIÇÃO
# ─────────────────────────────────────────────────────────
print("\n[MASK] Construindo máscaras de transição...")

state_changes = regime_bin.diff().fillna(0).abs() > 0
change_times  = regime_bin.index[state_changes].tolist()
print(f"[MASK] Mudanças de estado ({regime_label}): {len(change_times)}")

_transition_cache = {}

def build_transition_mask(minutes):
    if minutes in _transition_cache:
        return _transition_cache[minutes]
    if minutes == 0:
        mask = pd.Series(False, index=df.index)
        _transition_cache[0] = mask
        return mask
    delta = pd.Timedelta(minutes=minutes)
    mask  = pd.Series(False, index=df.index)
    for t in change_times:
        mask.loc[(mask.index >= t - delta) & (mask.index <= t + delta)] = True
    _transition_cache[minutes] = mask
    print(f"  trans_{minutes}min: {mask.sum():,} pts ({100*mask.mean():.1f}%)")
    return mask

trans_720_idx = build_transition_mask(720).reindex(df.index, fill_value=False)
trans_240_idx = build_transition_mask(240).reindex(df.index, fill_value=False)
trans_0_idx   = build_transition_mask(0).reindex(df.index, fill_value=False)

steady_tc  = (regime_bin == 1) & (~trans_240_idx)
steady_tv  = pd.Series(True, index=df.index)   # TV: todos os estados
steady_def = (regime_bin == 1) & (~trans_720_idx)

print(f"\n[MASK] Steady state:")
print(f"  TC/T5  (regime=1, trans=240min): {steady_tc.sum():,} ({100*steady_tc.mean():.1f}%)")
print(f"  TV_*   (todos os estados):        {steady_tv.sum():,} ({100*steady_tv.mean():.1f}%)")
print(f"  Padrão (regime=1, trans=720min): {steady_def.sum():,} ({100*steady_def.mean():.1f}%)")


def get_steady_mask(sensor):
    if sensor in VIBRATION_SENSORS:
        return steady_tv
    grp  = SENSOR_TO_GROUP.get(sensor, {})
    mins = grp.get("transition_mask_minutes", TRANSITION_MASK_MINUTES_DEFAULT)
    if mins == 240:
        return steady_tc
    if mins == 0:
        return steady_tv
    return steady_def


def get_transition_mask_for_sensor(sensor):
    if sensor in VIBRATION_SENSORS:
        return trans_0_idx
    grp  = SENSOR_TO_GROUP.get(sensor, {})
    mins = grp.get("transition_mask_minutes", TRANSITION_MASK_MINUTES_DEFAULT)
    if mins == 240:
        return trans_240_idx
    if mins == 0:
        return trans_0_idx
    return trans_720_idx


# ─────────────────────────────────────────────────────────
# 6. MÁSCARA DE ALARMES
# ─────────────────────────────────────────────────────────
alarm_mask_idx = pd.Series(False, index=df.index)
if len(df_alarm) > 0:
    delta_alarm = pd.Timedelta(minutes=EXCLUDE_ALARM_MINUTES)
    for t in df_alarm["data_datetime"]:
        alarm_mask_idx.loc[
            (alarm_mask_idx.index >= t - delta_alarm) &
            (alarm_mask_idx.index <= t + delta_alarm)
        ] = True
    print(f"\n[ALARM] Excluídos: {alarm_mask_idx.sum():,} ({100*alarm_mask_idx.mean():.1f}%)")

global_exclude_base = (regime_bin != 1) | trans_720_idx | alarm_mask_idx


def get_global_exclude(sensor):
    if sensor in VIBRATION_SENSORS:
        return alarm_mask_idx.copy()
    trans = get_transition_mask_for_sensor(sensor)
    return (regime_bin != 1) | trans | alarm_mask_idx


# ─────────────────────────────────────────────────────────
# 7. MÁSCARA DE GRADIENTE POR SENSOR
# ─────────────────────────────────────────────────────────
print("\n[GRAD] Calculando máscaras de gradiente por sensor...")

sensor_profiles       = {}
grad_masks_per_sensor = {}

for s in sensors:
    sc  = SENSOR_CLASSES.get(s, "default")
    grp = SENSOR_TO_GROUP.get(s, {})
    g_mult     = grp.get("grad_std_mult",         GRAD_STD_MULT_DEFAULT)
    g_suppress = grp.get("grad_suppress_minutes", GRAD_SUPPRESS_MINUTES_DEFAULT)
    t_mins     = grp.get("transition_mask_minutes", TRANSITION_MASK_MINUTES_DEFAULT)
    r_steady   = grp.get("running_steady_values", RUNNING_STEADY_VALUES_DEFAULT)
    steady_s   = get_steady_mask(s)

    group_name = "default"
    for g in ALL_GROUPS:
        if s in g["sensors"]:
            group_name = GROUP_NAMES[id(g)]
            break

    lim = SENSOR_LIMITS.get(s, {})

    if sc == "binary":
        grad_masks_per_sensor[s] = pd.Series(False, index=df.index)
        sensor_profiles[s] = {
            "class": "binary", "group": "binary",
            "grad_std_mult": None, "grad_suppress_minutes": None,
            "transition_mask_minutes": None,
            "running_steady_values": list(RUNNING_STEADY_VALUES_DEFAULT),
            "grad_mu_r1": 0, "grad_std_r1": 0,
            "grad_threshold": None, "n_grad_spikes": 0,
            "limits": lim,
        }
        continue

    grad = df[s].diff().abs().fillna(0)

    if sc == "low_signal":
        grad_masks_per_sensor[s] = pd.Series(False, index=df.index)
        gmu  = float(grad[steady_s].mean())
        gstd = float(grad[steady_s].std())
        sensor_profiles[s] = {
            "class": "low_signal", "group": "low_signal",
            "grad_std_mult": None, "grad_suppress_minutes": None,
            "transition_mask_minutes": t_mins,
            "running_steady_values": list(r_steady),
            "grad_mu_r1": gmu, "grad_std_r1": gstd,
            "grad_threshold": None, "n_grad_spikes": 0,
            "limits": lim,
        }
        continue

    grad_steady = grad[steady_s]
    gmu    = float(grad_steady.mean())
    gstd   = float(grad_steady.std())
    thresh = gmu + g_mult * gstd

    spike_times = grad.index[grad > thresh].tolist()
    n_spikes    = len(spike_times)

    delta_grad = pd.Timedelta(minutes=g_suppress)
    gmask      = pd.Series(False, index=df.index)
    for t in spike_times:
        gmask.loc[(gmask.index >= t - delta_grad) & (gmask.index <= t + delta_grad)] = True

    grad_masks_per_sensor[s] = gmask
    sensor_profiles[s] = {
        "class":  "default",
        "group":  group_name,
        "grad_std_mult":          g_mult,
        "grad_suppress_minutes":  g_suppress,
        "transition_mask_minutes": t_mins,
        "running_steady_values":  list(r_steady),
        "grad_mu_r1":   gmu,
        "grad_std_r1":  gstd,
        "grad_threshold": float(thresh),
        "n_grad_spikes":  n_spikes,
        "pct_masked_by_grad": float(gmask.mean()),
        "limits": lim,
    }

print("\n  Spikes após calibração por grupo:")
for gname, gsensors in [
    ("TC/T5",        GROUP_TC["sensors"]),
    ("TV_*",         GROUP_TV["sensors"]),
    ("PI_highspike", GROUP_PI_HIGHSPIKE["sensors"]),
    ("PDI_lowamp",   GROUP_PDI_LOWAMP["sensors"]),
    ("TI",           GROUP_TI["sensors"]),
]:
    total = sum(sensor_profiles.get(s, {}).get("n_grad_spikes", 0) for s in gsensors)
    print(f"    {gname:<15} total_spikes={total:,}")


# ─────────────────────────────────────────────────────────
# 8. NORMALIZAÇÃO E SÉRIES DE TREINO POR SENSOR
# ─────────────────────────────────────────────────────────
print("\n[NORM] Calculando normalização por sensor...")

normalization        = {}
df_normal_per_sensor = {}

for s in sensors:
    gmask     = grad_masks_per_sensor[s]
    exclude_s = get_global_exclude(s) | gmask.reindex(df.index, fill_value=False)
    normal_s  = df.loc[~exclude_s, s]

    if len(normal_s) < 10:
        normal_s = df[s] if s in VIBRATION_SENSORS else df.loc[regime_bin == 1, s]
        print(f"  [WARN] {s}: fallback ({len(normal_s)} pts)")

    mu_s  = float(normal_s.mean())
    std_s = float(normal_s.std()) if normal_s.std() > 0 else 1.0

    normalization[s]        = {"mean": mu_s, "std": std_s, "n_normal": len(normal_s)}
    df_normal_per_sensor[s] = (normal_s - mu_s) / std_s

    sensor_profiles[s]["norm_mean"]    = mu_s
    sensor_profiles[s]["norm_std"]     = std_s
    sensor_profiles[s]["n_normal_pts"] = len(normal_s)
    sensor_profiles[s]["pct_normal"]   = round(100 * len(normal_s) / len(df), 2)

df_all_z = pd.DataFrame(
    {s: (df[s] - normalization[s]["mean"]) / normalization[s]["std"] for s in sensors},
    index=df.index
)

df_grad_z = pd.DataFrame(index=df.index)
for s in sensors:
    steady_s = get_steady_mask(s)
    g  = df[s].diff().abs().fillna(0)
    gm = float(g[steady_s].mean())
    gs = float(g[steady_s].std()) if g[steady_s].std() > 0 else 1.0
    df_grad_z[s] = (g - gm) / gs


# ─────────────────────────────────────────────────────────
# 9. FUNÇÃO DE NÍVEL FÍSICO  ← NOVO v6
# ─────────────────────────────────────────────────────────

def get_physical_level(sensor: str, value: float, ngp: float = None) -> str:
    """
    Retorna o nível físico de um valor para um sensor.

    Parâmetros:
      sensor : nome do sensor
      value  : valor bruto (na unidade original do sensor)
      ngp    : NGP atual em % (opcional — usado para limiares dinâmicos T5)

    Retorna:
      "normal"   → abaixo de todos os limiares
      "alerta"   → entre L e H, ou entre H e HH
      "critico"  → acima de HH ou abaixo de LL
      "sem_limite" → sensor sem limites definidos
    """
    lim = SENSOR_LIMITS.get(sensor)
    if not lim:
        return "sem_limite"

    # Limiares dinâmicos para T5 (função do NGP)
    H  = lim.get("H")
    HH = lim.get("HH")
    L  = lim.get("L")
    LL = lim.get("LL")

    if "ngp_threshold" in lim and ngp is not None and not np.isnan(ngp):
        if ngp < lim["ngp_threshold"]:
            H  = lim.get("H_low_ngp",  H)
            HH = lim.get("HH_low_ngp", HH)

    # Verificação crítico primeiro
    if HH is not None and value >= HH:
        return "critico"
    if LL is not None and value <= LL:
        return "critico"

    # Alerta
    if H is not None and value >= H:
        return "alerta"
    if L is not None and value <= L:
        return "alerta"

    return "normal"


def get_physical_level_series(sensor: str) -> pd.Series:
    """
    Aplica get_physical_level a toda a série temporal do sensor.
    Retorna pd.Series com valores "normal" | "alerta" | "critico" | "sem_limite".
    Usa NGP interpolado quando disponível.
    """
    values = df[sensor]
    result = []
    for i, (t, v) in enumerate(values.items()):
        ngp_val = float(ngp_idx.iloc[i]) if ngp_available else None
        result.append(get_physical_level(sensor, v, ngp=ngp_val))
    return pd.Series(result, index=values.index, name=f"{sensor}_level")


# ─────────────────────────────────────────────────────────
# 10. FUNÇÕES MODULARES (mantidas da v5)
# ─────────────────────────────────────────────────────────

def get_multi_normal_mask(sensors_list):
    """Intersecção conservadora das máscaras de treino para múltiplos sensores."""
    exclude_combined = pd.Series(False, index=df.index)
    for s in sensors_list:
        exc_s = get_global_exclude(s)
        gmask = grad_masks_per_sensor.get(s, pd.Series(False, index=df.index))
        exclude_combined = exclude_combined | exc_s | gmask.reindex(df.index, fill_value=False)
    valid_mask = ~exclude_combined
    print(f"[MULTI_MASK] {len(sensors_list)} sensores → {valid_mask.sum():,} pts "
          f"({100*valid_mask.mean():.1f}%)")
    return valid_mask


def get_feature_array(sensors_list, mode="zscore"):
    """Monta DataFrame de features. mode: zscore | zscore+grad | raw"""
    valid_modes = {"zscore", "zscore+grad", "raw"}
    if mode not in valid_modes:
        raise ValueError(f"mode '{mode}' inválido. Escolha: {valid_modes}")
    if mode == "zscore":
        feat = df_all_z[sensors_list].copy()
        feat.columns = [f"{s}__zscore" for s in sensors_list]
    elif mode == "zscore+grad":
        z_part = df_all_z[sensors_list].copy()
        z_part.columns = [f"{s}__zscore" for s in sensors_list]
        g_part = df_grad_z[sensors_list].copy()
        g_part.columns = [f"{s}__grad" for s in sensors_list]
        feat = pd.concat([z_part, g_part], axis=1)
    else:
        feat = df[sensors_list].copy()
        feat.columns = [f"{s}__raw" for s in sensors_list]
    print(f"[FEATURES] mode='{mode}' | n_features={feat.shape[1]} | shape={feat.shape}")
    return feat


def get_context_sensors_by_correlation(target_sensor, n_top=4, min_corr=0.5):
    """Sugere sensores de contexto por correlação de Pearson em steady state."""
    steady_pts = get_steady_mask(target_sensor)
    df_steady  = df_all_z.loc[steady_pts]
    corr_s     = df_steady.corr()[target_sensor].drop(target_sensor).abs()
    skip = {"binary", "low_signal"}
    if target_sensor in VIBRATION_SENSORS:
        valid = [s for s in corr_s.index
                 if s in VIBRATION_SENSORS and SENSOR_CLASSES.get(s, "default") not in skip]
    else:
        valid = [s for s in corr_s.index if SENSOR_CLASSES.get(s, "default") not in skip]
    top = (corr_s[valid][corr_s[valid] >= min_corr]
           .sort_values(ascending=False)
           .head(n_top))
    result = top.index.tolist()
    print(f"[CORR] Contexto para '{target_sensor}':")
    for s in result:
        print(f"  {s:<40} corr={corr_s[s]:.3f}")
    if not result:
        print(f"  Nenhum sensor com corr >= {min_corr}.")
    return result


# ─────────────────────────────────────────────────────────
# 11. SUPRESSÃO (scoring)
# ─────────────────────────────────────────────────────────

def get_suppression_mask(sensor):
    """True = ponto deve ser suprimido no scoring."""
    sc = SENSOR_CLASSES.get(sensor, "default")
    if sc == "binary":
        return pd.Series(True, index=df.index)
    gmask = grad_masks_per_sensor.get(sensor, pd.Series(False, index=df.index))
    return get_global_exclude(sensor) | gmask.reindex(df.index, fill_value=False)


# ─────────────────────────────────────────────────────────
# 12. SALVAR PERFIS
# ─────────────────────────────────────────────────────────
profile_path = os.path.join(OUTPUT_DIR, "sensor_profiles_v6.json")
with open(profile_path, "w", encoding="utf-8") as f:
    # Serializar lim (converte sets para listas)
    profiles_serial = {}
    for s, p in sensor_profiles.items():
        pp = dict(p)
        if "running_steady_values" in pp and isinstance(pp["running_steady_values"], set):
            pp["running_steady_values"] = sorted(list(pp["running_steady_values"]))
        profiles_serial[s] = pp
    json.dump(profiles_serial, f, indent=2, ensure_ascii=False)

norm_path = os.path.join(OUTPUT_DIR, "normalization_v6.json")
with open(norm_path, "w", encoding="utf-8") as f:
    json.dump(normalization, f, indent=2, ensure_ascii=False)

limits_path = os.path.join(OUTPUT_DIR, "sensor_limits.json")
with open(limits_path, "w", encoding="utf-8") as f:
    json.dump(SENSOR_LIMITS, f, indent=2, ensure_ascii=False)

print(f"\n[SAVE] Perfis:       {profile_path}")
print(f"[SAVE] Normalização: {norm_path}")
print(f"[SAVE] Limites:      {limits_path}")


# ─────────────────────────────────────────────────────────
# 13. EXPORTAÇÕES
# ─────────────────────────────────────────────────────────
EXPORTS = {
    # Dados
    "df":                   df,
    "df_all_z":             df_all_z,
    "df_grad_z":            df_grad_z,
    "df_normal_per_sensor": df_normal_per_sensor,

    # Regime — NOVO v6
    "running_bin":          running_bin_idx,        # sempre RUNNING_A
    "regime_bin":           regime_bin,             # regime ativo (RUNNING ou NGP)
    "regime_col_used":      REGIME_COL,
    "regime_label":         regime_label,
    "ngp_series":           ngp_idx,                # NGP interpolado (ou NaN se indisponível)
    "ngp_available":        ngp_available,
    "NGP_MIN_PCT":          NGP_MIN_PCT,

    # Máscaras
    "transition_mask":      trans_720_idx,
    "alarm_mask":           alarm_mask_idx,
    "global_exclude_base":  global_exclude_base,
    "grad_masks":           grad_masks_per_sensor,

    # Funções de máscara
    "suppression_mask_fn":     get_suppression_mask,
    "get_steady_mask":         get_steady_mask,
    "get_transition_mask":     get_transition_mask_for_sensor,
    "get_global_exclude":      get_global_exclude,

    # Funções de features
    "get_multi_normal_mask":              get_multi_normal_mask,
    "get_feature_array":                  get_feature_array,
    "get_context_sensors_by_correlation": get_context_sensors_by_correlation,

    # Limiares físicos — NOVO v6
    "sensor_limits":          SENSOR_LIMITS,
    "get_physical_level":     get_physical_level,
    "get_physical_level_series": get_physical_level_series,

    # Metadados
    "sensors":              sensors,
    "sensor_profiles":      sensor_profiles,
    "normalization":        normalization,
    "sensor_classes":       SENSOR_CLASSES,
    "sensor_to_group":      SENSOR_TO_GROUP,
    "vibration_sensors":    VIBRATION_SENSORS,

    # Parâmetros
    "TRANSITION_MASK_MINUTES_DEFAULT": TRANSITION_MASK_MINUTES_DEFAULT,
    "EXCLUDE_ALARM_MINUTES":           EXCLUDE_ALARM_MINUTES,
}


# ─────────────────────────────────────────────────────────
# 14. SUMÁRIO DE SAÚDE
# ─────────────────────────────────────────────────────────
print(f"\n[DONE] Pré-processamento v6-ngp completo.")
print(f"  Regime ativo : {regime_label}")
print(f"  NGP disponível: {ngp_available}")
print(f"  Sensores: {len(sensors)} | Período: {df.index.min()} → {df.index.max()}")
print(f"  Pontos:   {len(df):,}")

print("\n[HEALTH] Sumário por sensor:")
hdr = (f"  {'Sensor':<35} {'Grupo':>14} {'n_normal':>10} "
       f"{'%normal':>8} {'mult':>6} {'spikes':>8} {'H':>7} {'HH':>7}")
print(hdr)
print("  " + "-" * len(hdr))
for s in sensors:
    p     = sensor_profiles[s]
    gname = p.get("group", "default")
    n_n   = p.get("n_normal_pts", 0)
    pct_n = p.get("pct_normal", 0)
    g_m   = p.get("grad_std_mult")
    g_s   = f"{g_m:.0f}x" if g_m else "N/A"
    nsp   = p.get("n_grad_spikes", 0)
    lim   = SENSOR_LIMITS.get(s, {})
    h_s   = str(lim.get("H", "-"))
    hh_s  = str(lim.get("HH", "-"))
    print(f"  {s:<35} {gname:>14} {n_n:>10,} {pct_n:>7.1f}% "
          f"{g_s:>6} {nsp:>8,} {h_s:>7} {hh_s:>7}")
