"""Configuração central do projeto: caminhos, tags, sentinelas, eventos de trip.

Espelha o padrão config-driven do projeto Transpetro-modelos: tudo que é
específico do equipamento vive aqui; os módulos são genéricos.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- caminhos
# A raiz do projeto é deduzida da posição deste arquivo (src/cabiunas_pdm/config.py),
# para o código rodar em qualquer checkout. Variáveis de ambiente sobrescrevem:
#   CABIUNAS_PDM_ROOT — raiz do repositório
#   CABIUNAS_RAW      — pasta com os arquivos brutos do PI (só os scripts de
#                       ingestão local precisam dela; o fluxo ClearML não usa)
PROJECT = Path(os.environ.get("CABIUNAS_PDM_ROOT", Path(__file__).resolve().parents[2]))
CABIUNAS = Path(os.environ.get("CABIUNAS_RAW", PROJECT.parent))
DATA = PROJECT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

ABR26_PI = CABIUNAS / "ABR_26" / "PI"
TAGSSEL_30S = ABR26_PI / "Turbina A - TagsSelecionadas - Completo - 30s - Timestamp UTC-3"
TAGSSEL_2MIN = ABR26_PI / "Turbina A - TagsSelecionadas - 2025 - 2min - Timestamp UTC-3"
PORTAL_UTC = ABR26_PI / "Turbina A - PortalIntegridade - Completo - Timestamp UTC"
ALARMES_XLSX = CABIUNAS / "ABR_26" / "Alarmes Selecionados Turbina A.xlsx"

# arquivos interpolated extras (fora do ABR_26) que cobrem trips / baselines
EXTRA_INTERPOLATED_UTC = [
    CABIUNAS / "Dados_v2/Dados/PI/2024/01_2024_interpolated_PortalIntegridade.xlsx",
    CABIUNAS / "Out_25/Dados/PI/2025/01_2025_interpolated_PortalIntegridade.xlsx",
    CABIUNAS / "Nov_25/Dados/PI/OLD/2025/04_2025_interpolated_PortalIntegridade.xlsx",
]

# Raízes onde procurar os mensais *interpolated_PortalIntegridade.xlsx (86 tags,
# timestamp UTC). Ordem = prioridade: o primeiro que tiver o mês é o usado.
# Estes arquivos são a ÚNICA fonte local de NGP_A/NCPSR_A — as TagsSelecionadas
# (38 tags) não incluem sinais de rotação.
PORTAL_ROOTS = [
    PORTAL_UTC,
    CABIUNAS / "Nov_25/Dados/PI",
    CABIUNAS / "Out_25/Dados/PI",
    CABIUNAS / "Dados_v2/Dados/PI",
]

# ---------------------------------------------------------------- grade / fuso
GRID = "2min"          # grade canônica
UTC_OFFSET_H = 3       # PortalIntegridade é UTC; canônico é UTC-3 (Brasil sem DST desde 2019)
PREFIX = "bapiha02-"   # prefixo das colunas nos xlsx

# ---------------------------------------------------------------- tags (38 da seleção)
TAG_RUNNING = "RUNNING_A"
TAG_MAINT = "HSX_6240001A"
# Decisão operacional: NGP_A é a fonte de verdade para máquina em operação.
# O limiar deve ser calibrado no dataset que o contenha e registrado antes do
# treino; RUNNING_A permanece apenas como sinal auxiliar/histórico.
TAG_OPERABILITY = "NGP_A"
# Limiar calibrado em 04/08/2026 sobre 302.915 amostras de 2 min com NGP válido
# (fev/2022–abr/2025, ver reports/DIAGNOSTICO_OPERABILIDADE.md):
#   - distribuição é bimodal: 64% em NGP≈0 (parado) e 36% em 88–99% (operando);
#   - vale empírico vazio entre 39,6% e 43,6% → limiar no centro;
#   - concordância com RUNNING_A = 99,98%, sem nenhum caso de RUNNING=1 com
#     NGP abaixo do limiar (NGP é superset de RUNNING, não perde operação);
#   - qualquer valor entre 30% e 70% dá a mesma máscara (±0,04%), o que torna a
#     escolha insensível ao ponto exato — daí a margem de ~35 pp para cada lado.
# PENDENTE: confirmação formal com a equipe de operação de Cabiúnas.
NGP_OPERATIONAL_THRESHOLD: float | None = 40.0
# Faixa de carga nominal observada em operação (p01=87,9% / mediana=92,1%).
# Usar como filtro adicional apenas se o baseline precisar de carga homogênea;
# como máscara principal cortaria rampas legítimas (278 amostras).
NGP_NOMINAL_LOAD = 85.0

# Sinais de rotação/carga que só existem nos arquivos PortalIntegridade (86 tags).
# NGP_A  — velocidade da turbina geradora de gás (%), sinal de operabilidade
# NPT_A  — velocidade da turbina de potência (%)
# NCPSR_A— rotação do compressor (rpm)
# TM_TORQUE_A — torque (kgf·m), útil como proxy de carga
OPERABILITY_TAGS = ["NGP_A", "NPT_A", "NCPSR_A", "TM_TORQUE_A"]
STARTUP_EXCLUDE_NGP = "2h"   # transiente pós-partida quando a máscara vem do NGP

TEMPERATURE_TAGS = [
    "954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
    "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
    "954005_624_TI_0301",
    "TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A",
    "TC382_06_A", "T5_AVG_A",
]
PRESSURE_TAGS = [
    "954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
    "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
    "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
    "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301",
]
VIBRATION_TAGS = [
    "TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
    "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A",
]
SENSOR_TAGS = TEMPERATURE_TAGS + PRESSURE_TAGS + VIBRATION_TAGS
ALL_TAGS = SENSOR_TAGS + [TAG_MAINT, TAG_RUNNING]
# Colunas do canônico: sensores + discretas + operabilidade (quando disponível).
CANONICAL_TAGS = ALL_TAGS + OPERABILITY_TAGS

# ---------------------------------------------------------------- limpeza
# Faixas físicas plausíveis por família; fora delas => NaN (sentinelas de
# interpolação como -40.51, -19.06, -11.02 caem aqui).
PHYSICAL_RANGE = {
    "temperature": (-15.0, 900.0),
    "pressure": (-1.5, 120.0),
    "vibration": (0.0, 200.0),
}
FREEZE_WINDOW = "30min"   # janela p/ detectar sensor congelado (std == 0 em operação)
STARTUP_EXCLUDE = "2h"    # transiente pós-partida excluído do "normal estável"


def family(tag: str) -> str:
    if tag in TEMPERATURE_TAGS:
        return "temperature"
    if tag in PRESSURE_TAGS:
        return "pressure"
    if tag in VIBRATION_TAGS:
        return "vibration"
    return "discrete"


# ---------------------------------------------------------------- eventos (trips confirmados)
@dataclass(frozen=True)
class Trip:
    """Parada 1→0 do RUNNING_A coincidente com alarme LL/HH (análise 2026-07)."""
    when: str                 # timestamp local UTC-3
    alarm: str
    mechanism: str            # oleo | mancal | selagem
    fine_data: bool           # há série interpolada 30s/2min cobrindo o evento?
    sensors: tuple[str, ...] = field(default_factory=tuple)  # sensores do mecanismo


# sensores por mecanismo (tags da seleção de 38)
OIL_SENSORS = ("954005_624_PI_0308", "954005_624_PI_0339", "954005_624_PI_0340",
               "954005_624_TI_0325", "954005_624_PDI_0338")
BEARING_SENSORS = ("954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
                   "954005_624_TI_0301", "TV_353X_A", "TV_353Y_A")
SEAL_SENSORS = ("954005_624_PDIT_0305", "954005_624_PDI_0302", "954005_624_PDI_0301",
                "954005_624_PI_0307")

TRIPS = [
    Trip("2023-11-05 15:05", "PALL_6240309", "oleo", True, OIL_SENSORS),
    Trip("2023-11-14 15:04", "PALL_6240309", "oleo", True, OIL_SENSORS),
    Trip("2024-01-17 07:57", "TAHH_6240305", "mancal", True, BEARING_SENSORS),
    Trip("2025-02-27 08:38", "PDAHH6240305", "selagem", False, SEAL_SENSORS),
    Trip("2025-02-27 17:33", "PDAHH6240305", "selagem", False, SEAL_SENSORS),
    Trip("2025-03-17 18:15", "TAHH_6240305", "mancal", False, BEARING_SENSORS),
    Trip("2025-03-18 11:16", "TAHH_6240305", "mancal", False, BEARING_SENSORS),
    # 04_2025_interpolated só tem dados reais até 04/04 13:42 ('No Data' depois)
    # => estes dois trips também ficam sem série fina local.
    Trip("2025-04-07 21:18", "TAHH_6240305", "mancal", False, BEARING_SENSORS),
    Trip("2025-04-11 17:03", "TAHH_6240305", "mancal", False, BEARING_SENSORS),
    # trips adicionais achados no cruzamento paradas(série fina 2min/30s)×alarmes
    # (análise 18/07/2026) — ambos óleo lubrificante, COM série fina local:
    Trip("2025-11-04 06:24", "PALL_6240309+PALL_6240340", "oleo", True, OIL_SENSORS),
    Trip("2026-02-26 15:34", "PALL_6240309+PALL_6240340", "oleo", True, OIL_SENSORS),
]

# ---------------------------------------------------------------- política de detecção
# Prioridade do projeto: MINIMIZAR FALSOS POSITIVOS (decisão 2026-07-18).
# Ponto de operação calibrado em held-out (mar–abr/2026) e no trip de 17/01/2024:
#   - alarme  : PCA-recon da família de temperaturas, score/p99 suavizado (EWMA 1h),
#               thr 2.0, sustentado 30 min  -> ~0,8 FP/mês, lead ~17 h
#   - early   : spread mancal (TI_0305 − mediana TI_0301/0303/0307), |z|>3 EWMA 30 min,
#               sustentado 30 min           -> ~0,8 FP/mês, lead ~46 h
#   - ALERTA CONFIRMADO = os dois sinais ativos -> FP conjunto ~0
#   - vibração: NÃO usar como alarme até definir re-fit por regime (instável em held-out)
DETECTION_POLICY = dict(
    family_score=dict(family="temperature", ewma_halflife="1h",
                      threshold_x_p99=2.0, sustain_min=30),
    early_warning=dict(feature="spread_mancal_0305", ewma_halflife="30min",
                       z_threshold=3.0, sustain_min=30),
    confirm_with_both=True,
    quarantine_families=["vibration"],
    max_fp_episodes_per_month=1.0,   # constraint dura p/ seleção de modelos (AutoML)
)

# períodos de operação estável com série fina, por regime (pré/pós grande parada)
STABLE_PERIODS = {
    "regime_2022": [("2022-06-01", "2022-10-15")],
    "regime_2024": [("2024-01-05", "2024-01-31")],
    "regime_2025_26": [("2025-01-01", "2025-01-31"), ("2025-04-01", "2025-04-27"),
                       ("2025-07-01", "2025-07-31"), ("2026-01-01", "2026-04-30")],
}
