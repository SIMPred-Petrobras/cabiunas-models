"""AutoML FP-first do detector de anomalias do TC-330.03A, executável no ClearML.

Busca o ponto de operação que **maximiza a antecipação das falhas sujeito a um
teto de falso positivo** (padrão: 1 episódio/mês), com protocolo estritamente
temporal (walk-forward, re-baseline mensal) sobre a série publicada no ClearML.

Por que este arquivo é autocontido
----------------------------------
Os workers do ClearML rodam em outra máquina e recebem apenas este arquivo, sem
o pacote ``src/cabiunas_pdm`` instalado. Toda a lógica necessária (limpeza,
operabilidade, scorers, avaliação) está aqui. Os parâmetros que também existem na
biblioteca são referenciados nos comentários para manter as duas versões
conscientes uma da outra — ``src/cabiunas_pdm/replay.py`` importa este arquivo por
caminho justamente para que o desenho use o mesmo cálculo da busca.

Uso local (teste rápido):
    python scripts/automl_clearml.py --mode quick --local

Uso remoto (deixa a task rodando na fila do ClearML):
    python scripts/automl_clearml.py --mode full --remote --queue default
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- dados
DATASET_ID = "68b25f9db0b8471a90b8100800d26e9a"   # Cabiunas brutos 2025-2026 alarmes mapeados
SENSORS_FILE = "sensores_brutos_2025_2026_30s.csv"
ALARMS_FILE = "alarmes_mapeados_colunas.csv"
RUNNING_TAG = "RUNNING_A"
MAINT_TAG = "HSX_6240001A"
RUNNING_THRESHOLD = 0.5      # RUNNING_A é interpolado: '== 1' perde as transições
STARTUP_EXCLUDE = "2h"

# Segundo marcador de operação: a temperatura de exaustão da turbina (T5)
# ----------------------------------------------------------------------
# Sugestão da equipe (26/08/2026), medida sobre os 16 meses da grade de 2 min: o
# T5_AVG_A é fortemente bimodal e separa "turbina em carga" de "turbina apagada"
# melhor que o próprio RUNNING_A. Com RUNNING_A = 1 a mediana é 634,7 °C (p5 =
# 573,2); com RUNNING_A = 0 é 32,1 °C (p95 = 81,2). Entre 300 °C e 550 °C há
# ~2.000 das 349.200 amostras — o limiar cai num vale praticamente vazio, então
# qualquer valor de 400 a 550 dá quase o mesmo corte (70 h de diferença).
#
# O que o porteiro remove que o RUNNING_A não removia: 159,2 h marcadas como
# operação (1,79% do tempo), sendo
#   * 60 de 91 trechos nas primeiras 2 h após uma partida — rampa de partida;
#   * 44,1 h em 17/08/2025 com exaustão média de 437,8 °C — outro estado de
#     operação, não carga plena;
#   * ~100 h entre 19/08 e 24/08/2025 com RUNNING_A = 1 e exaustão a ~30 °C, isto
#     é, o tag afirmando operação com a turbina fria. Esse é o trecho que produz
#     os dois maiores falsos positivos da série (104,5 h e 14,0 h).
# O caminho inverso não existe na prática: RUNNING_A = 0 com T5 alto são 11
# amostras isoladas (0,4 h), então o T5 não serve para remendar as quedas de
# telemetria do RUNNING_A — só para desqualificar operação aparente.
# Silêncio do ALERTA após cada partida (eixo (a) da decisão de 26/08/2026)
# ----------------------------------------------------------------------
# Diferente de ``STARTUP_EXCLUDE``, que age no baseline E na pontuação: aqui só o
# ALERTA é suprimido, e as horas suprimidas saem do denominador de FP/mês — o
# tempo em que ninguém está vigiando não conta como oportunidade de errar.
# A separação existe porque as duas coisas têm risco diferente: ampliar
# ``STARTUP_EXCLUDE`` encolhe o baseline e desloca os limiares (foi o que afundou
# o porteiro de exaustão), enquanto o silêncio do alerta não toca no score.
# Medido com máscara no alerta: 24 h mantém 6/8 e leva o alarme falso de 21,1 para
# 13,8 h/mês (-35%); 36 h perde a detecção de 29/04/2025, que vem 28,7 h depois de
# uma partida. Ou seja, 24 h é a borda, não um valor com folga.
POST_START_SILENCE = "0h"

EXHAUST_TAG = "T5_AVG_A"
EXHAUST_ON_C = 500.0
# Onde o marcador age:
#   "operation" — entra na definição de operabilidade (afeta baseline, score,
#                 dias avaliados, paradas e, por consequência, o catálogo de trips);
#   "stable"    — só limpa o que treina e o que é pontuado, preservando a
#                 definição de parada/falha que já foi validada com a operação;
#   "off"       — só RUNNING_A (padrão).
#
# MEDIDO em 26/08/2026, antes de mudar o padrão: o porteiro PIORA o resultado.
# Mesma arquitetura (PCA, 2 min, baseline de 3.000 h), varrendo limiar (percentil
# e k-ésimo maior), EWMA e persistência — 180 configurações nos dois mundos:
#   sem porteiro : melhor 6/8 dentro do teto, FP 0,94/mês, 22,0 h/mês de alarme falso
#   com porteiro : melhor 5/8 dentro do teto, FP 0,96/mês, ~50 h/mês
# A causa está no limiar, não na limpeza: as 155 h removidas concentram a cauda
# do score do baseline, então o p99,9 desaba — em 12/2025 o limiar de temperatura
# cai a 3% do valor anterior, e o de pressão a 0,3% em 04/2025. Um punhado de
# amostras contaminadas vinha sustentando o limiar de cinco meses seguidos. Com o
# baseline limpo o detector fica muito mais sensível, e no mesmo percentil isso
# vira falso positivo. Trocar para "k_maiores" não recupera.
# Fica implementado e DESLIGADO: a ideia é boa e a contaminação é real, mas para
# aproveitá-la é preciso recalibrar o limiar numa varredura completa, não trocar
# só a operabilidade.
EXHAUST_GATE = "off"
# Duração mínima de um trecho "frio" para valer como desqualificação. Os 91
# trechos de T5 baixo com RUNNING_A = 1 têm duração mediana de 0,07 h (4 min) e
# 60 deles começam nas 2 h seguintes a uma partida: são a rampa de partida, que
# já tem tratamento próprio (``STARTUP_EXCLUDE``). Mascarar essas dezenas de
# trechos curtos fragmenta a máscara sem remover contaminação relevante. Com o
# piso, o marcador age apenas onde há um regime frio sustentado — as ~100 h de
# 19-24/08/2025 com o tag afirmando operação e a exaustão a 30 °C, e as 44,1 h de
# 17/08/2025 a 437,8 °C.
EXHAUST_MIN_H = 1.0

# famílias de sensores (espelham config.TEMPERATURE_TAGS / PRESSURE_TAGS)
FAMILIES: dict[str, list[str]] = {
    "temperatura": [
        "954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
        "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
        "954005_624_TI_0301", "TC382_01_A", "TC382_02_A", "TC382_03_A",
        "TC382_04_A", "TC382_05_A", "TC382_06_A", "T5_AVG_A",
    ],
    "pressao_oleo": [
        "954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
        "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
        "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
        "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301",
    ],
}
# Vibração fica em QUARENTENA (decisão do projeto): o normal dela não se sustenta
# entre meses e gera falso positivo crescente. Habilitar só com --with-vibration.
VIBRATION = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
             "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]

# sinal univariado de diagnóstico: mancal que costuma falhar contra os irmãos
BEARING_TARGET = "954005_624_TI_0305"
BEARING_PEERS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]

# 4º sinal, dedicado à selagem (acrescentado em 15/08/2026)
# ------------------------------------------------------
# Medição que motivou: abrindo o erro de reconstrução por sensor antes da falha
# de selagem de 27/02/2025, o PDIT_0305 respondia por 53% do erro — e mesmo assim
# não houve alerta, porque o score da família de pressão é a MÉDIA sobre 12
# sensores e um sensor gritando vira sussurro depois de dividido por 12. Este
# sinal tira o PDIT_0305 da média e o trata como o spread do mancal: z robusto
# contra o próprio baseline.
SEAL_TARGET = "954005_624_PDIT_0305"

PHYSICAL_RANGE = {"temperatura": (-15.0, 900.0), "pressao_oleo": (-1.5, 120.0),
                  "vibracao": (0.0, 200.0)}
RANGE_BY_COLUMN = {c: PHYSICAL_RANGE[fam] for fam, cols in FAMILIES.items() for c in cols}
RANGE_BY_COLUMN.update({c: PHYSICAL_RANGE["vibracao"] for c in VIBRATION})

# ------------------------------------------------------- definição de falha
# Até 14/08/2026 os trips eram uma LISTA FIXA, herdada de uma análise feita sobre
# os mensais locais — que têm dado real só até 04/04/2025. A varredura sistemática
# sobre a série do ClearML achou duas falhas que faltavam, inclusive a maior de
# todas (29/04/2025, mancal, 23 dias de máquina parada). Agora a lista é derivada
# dos dados, com o critério abaixo:
#
#   trip  = parada real (RUNNING_A 1→0 que dura >= MIN_STOP_HOURS)
#           COINCIDENTE com alarme de nível na janela [-1 h, +30 min]
#   falha = trips a menos de EVENT_GAP um do outro, contados como um evento
#
# A conjunção é indispensável: `TAHH_6240305` sozinho tem 114 ativações na
# planilha e só 6 coincidem com a máquina caindo — nas outras ele dispara com o
# equipamento já parado, como eco da parada.
MIN_STOP_HOURS = 2.0     # abaixo disso é queda de telemetria do RUNNING_A, não parada
                         # (medido: 65 das 135 transições na grade de 30 s duram <3 min)
STOP_ALARM_BEFORE = "1h"
STOP_ALARM_AFTER = "30min"
EVENT_GAP = "24h"

# Nível do alarme pela DESCRIÇÃO, porque é a única coluna que o dataset deste
# script traz (a tag com o nível está na outra versão do dataset, de 594 MB).
# Validado contra a tag real nas 6.851 linhas da planilha: **100% de concordância,
# zero divergências**. "Mt.Alta"/"M.Bx" é o nível de intertravamento; "Alta"/"Baixa"
# sozinho é aviso.
ALARM_LEVEL_PATTERN = r"TRIP|MT\.\s?ALTA|M\.\s?ALTA|MT\.\s?BX|M\.\s?BX|MT\.\s?BAIXA|M\.\s?BAIXA"

# mecanismo pela coluna de sensor que o alarme de nível referencia
MECHANISM_BY_SENSOR = {
    "954005_624_TI_0305": "mancal", "954005_624_TI_0303": "mancal",
    "954005_624_TI_0307": "mancal", "954005_624_TI_0301": "mancal",
    "954005_624_PI_0339": "oleo", "954005_624_PI_0340": "oleo",
    "954005_624_TI_0325": "oleo", "954005_624_PI_0309": "oleo",
    "954005_624_PDIT_0305": "selagem", "954005_624_PDI_0302": "selagem",
}

DETECTION_WINDOW = "48h"     # janela antes do trip em que o alerta conta como detecção
EPISODE_GAP = "2h"           # alertas separados por menos que isso são o mesmo episódio

# Alarmes de UTILIDADE: suprimento de gás, a montante do compressor. São 60% das
# 420 ativações que ocorrem em operação (PI_0319: 182, PI_0315: 71). Não são
# degradação do equipamento — ficam separados nas contas para que o resultado do
# detector não seja julgado (nem elogiado) por eles.
UTILITY_TAGS = ["954005_624_PI_0319", "954005_624_PI_0315"]
ALARM_EPISODE_GAP = "24h"    # ativações a menos disso são o mesmo episódio de alarme
# ±3 h e não mais: com ±12 h a união das janelas cobre 47% do tempo em operação e
# quase todo falso positivo cairia ali por acaso. Com ±3 h a cobertura é 14,9%, então
# um alerta nessa faixa é 3,4× mais provável que o azar.
ALARM_EXPLAIN_WINDOW = "3h"

# Limpeza do baseline (decisão de 12/08/2026, critério da equipe)
# ---------------------------------------------------------------
# O baseline é "tudo que estava operando nos últimos N dias". Se um trip aconteceu
# dentro dessa janela, a degradação que levou a ele entra no treino como se fosse
# normal — e o modelo aprende a degradação como saudável. Medido: com baseline de
# 28 dias, 5 das 15 janelas mensais contêm um trip.
#
# Correção: excluir do baseline os dias que antecedem cada evento conhecido. É o
# mesmo princípio do `exclusion_days_before` já validado no projeto Transpetro.
# Os eventos considerados são os trips confirmados MAIS as ativações de alarme de
# nível TRIP ocorridas com a máquina em operação (as que dispararam com a máquina
# já parada são consequência da parada, não sintoma — 91% dos casos).
EXCLUSION_ALARM_HINT = "TRIP"     # marca de nível no texto da descrição do alarme


# ------------------------------------------------------------------ scorers
class FamilyScorer:
    """Modelo de normal de uma família: ``fit`` no baseline, ``score`` normalizado.

    O score é dividido pelo p99 do próprio baseline, então 1,0 = borda do normal
    aprendido, independentemente do modelo e da escala dos sensores.
    """

    def __init__(self, kind: str, seed: int = 0) -> None:
        self.kind = kind
        self.seed = seed

    def fit(self, frame: pd.DataFrame) -> "FamilyScorer":
        from sklearn.preprocessing import RobustScaler
        data = frame.dropna()
        if len(data) < 500:
            raise ValueError(f"baseline curto: {len(data)} amostras")
        self.columns = list(data.columns)
        self.scaler = RobustScaler().fit(data)
        scaled = self.scaler.transform(data)
        self.model = self._build(scaled.shape[1])
        # Subamostra APENAS o ajuste, e apenas para os modelos que não escalam.
        # Determinístico (linspace, não sorteio) para a busca ser reprodutível.
        teto = self.MAX_FIT_SAMPLES.get(self.kind)
        if teto and len(scaled) > teto:
            passos = np.linspace(0, len(scaled) - 1, teto).astype(int)
            treino = scaled[passos]
        else:
            treino = scaled
        if self.kind.startswith("ae"):
            self.model.fit(treino, treino)       # autoencoder: alvo = entrada
        else:
            self.model.fit(treino)
        # Guarda o score do próprio baseline: o limiar é um QUANTIL desta
        # distribuição (após a mesma suavização aplicada ao teste), o que torna
        # a busca válida para qualquer modelo — inclusive os de score limitado,
        # como IsolationForest, em que múltiplos de p99 são inalcançáveis.
        self.baseline_score = pd.Series(self._raw(scaled), index=data.index)
        return self

    # Arquiteturas disponíveis. O critério de inclusão é escalar: o baseline vai
    # de 9 mil a 460 mil amostras, e o ajuste é a etapa cara da busca (30 por
    # chave de cache). Modelos com custo quadrático em amostras — OneClassSVM
    # exato, LocalOutlierFactor, KernelPCA — ficaram FORA de propósito: com
    # 100 mil pontos de treino e 1,4 M de teste eles não terminam.
    #
    #   pca        erro de reconstrução (estatística Q / SPE do monitoramento
    #              de processo). Barato e é o que já vinha sendo usado.
    #   pca_t2     T² de Hotelling nos escores latentes do PCA. Complementa o
    #              anterior: Q vê o que sai do subespaço normal, T² vê o que
    #              anda longe dentro dele. Par clássico em processo industrial.
    #   mahal      Mahalanobis com covariância encolhida (Ledoit-Wolf). Barato,
    #              e é a referência honesta contra a qual um autoencoder tem
    #              que provar que vale a complexidade.
    #   gmm        mistura de gaussianas, log-verossimilhança negativa. Pega
    #              normal MULTIMODAL, que é o caso aqui: a máquina opera em mais
    #              de um regime de carga.
    #   iforest    isolamento por particionamento aleatório.
    #   ocsvm_sgd  fronteira de uma classe, versão linear/SGD com aproximação de
    #              kernel de Nyström — O(n), ao contrário do OneClassSVM exato.
    #   ae         autoencoder denso, gargalo em n/4.
    #   ae_deep    mais profundo e com gargalo mais apertado (n/6).
    #   ae_wide    uma camada larga (2n) e gargalo em n/3.
    ARQUITETURAS = ("pca", "pca_t2", "mahal", "gmm", "iforest", "ocsvm_sgd",
                    "ae", "ae_deep", "ae_wide")

    # Ajustar em subamostra quando o baseline é grande: para estes modelos o
    # custo cresce mais que linear e o ganho de precisão satura muito antes.
    # A subamostra é SEMPRE do baseline (nunca do teste), então não há vazamento.
    MAX_FIT_SAMPLES = {"gmm": 60_000, "ocsvm_sgd": 60_000, "iforest": 120_000}

    def _build(self, n_features: int):
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.kernel_approximation import Nystroem
        from sklearn.mixture import GaussianMixture
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline

        if self.kind in ("pca", "pca_t2"):
            return PCA(n_components=0.95, svd_solver="full",
                       random_state=self.seed)
        if self.kind == "mahal":
            return LedoitWolf(store_precision=True)
        if self.kind == "gmm":
            # 3 componentes: a carga tem mais de um patamar, mas com 12-14
            # sensores e covariância cheia mais componentes ficam mal estimados.
            return GaussianMixture(n_components=3, covariance_type="full",
                                   reg_covar=1e-4, max_iter=100,
                                   random_state=self.seed)
        if self.kind == "iforest":
            return IsolationForest(n_estimators=200, contamination="auto",
                                   random_state=self.seed, n_jobs=-1)
        if self.kind == "ocsvm_sgd":
            return make_pipeline(
                Nystroem(gamma=1.0 / max(n_features, 1), n_components=100,
                         random_state=self.seed),
                SGDOneClassSVM(nu=0.05, random_state=self.seed))
        if self.kind in ("ae", "ae_deep", "ae_wide"):
            if self.kind == "ae":
                hidden = (max(8, n_features // 2), max(4, n_features // 4),
                          max(8, n_features // 2))
            elif self.kind == "ae_deep":
                hidden = (max(12, n_features), max(8, n_features // 2),
                          max(3, n_features // 6), max(8, n_features // 2),
                          max(12, n_features))
            else:
                hidden = (max(24, n_features * 2), max(5, n_features // 3),
                          max(24, n_features * 2))
            return MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                early_stopping=True, n_iter_no_change=5,
                                max_iter=60, random_state=self.seed)
        raise ValueError(f"modelo desconhecido: {self.kind}")

    def _raw(self, scaled: np.ndarray) -> np.ndarray:
        """Score bruto, sempre no sentido "maior = mais anômalo"."""
        if self.kind == "pca":
            back = self.model.inverse_transform(self.model.transform(scaled))
            return np.mean((scaled - back) ** 2, axis=1)
        if self.kind == "pca_t2":
            # T² de Hotelling: distância no subespaço latente, normalizada pela
            # variância explicada de cada componente.
            latente = self.model.transform(scaled)
            return np.sum(latente ** 2 / self.model.explained_variance_, axis=1)
        if self.kind == "mahal":
            return self.model.mahalanobis(scaled)
        if self.kind.startswith("ae"):
            return np.mean((self.model.predict(scaled) - scaled) ** 2, axis=1)
        if self.kind == "ocsvm_sgd":
            return -self.model.decision_function(scaled)
        return -self.model.score_samples(scaled)   # iforest, gmm

    def score(self, frame: pd.DataFrame) -> pd.Series:
        data = frame[self.columns]
        ok = data.notna().all(axis=1).to_numpy()
        out = np.full(len(data), np.nan)
        if ok.any():
            out[ok] = self._raw(self.scaler.transform(data[ok]))
        return pd.Series(out, index=data.index)


# --------------------------------------------------------------------- dados
class DataBundle:
    """Série limpa do ClearML, em uma ou mais grades temporais."""

    def __init__(self, dataset_id: str = DATASET_ID, local_csv: Path | None = None,
                 with_vibration: bool = False, single_model: bool = False) -> None:
        self.dataset_id = dataset_id
        self.local_csv = local_csv
        self.families = dict(FAMILIES)
        if with_vibration:
            self.families["vibracao"] = VIBRATION
        # --single-model: um multivariado sobre TODOS os sensores, sem os sinais
        # derivados. É a variante de controle — serve para medir quanto a estrutura
        # de 4 sinais realmente ganha, em vez de afirmar que ganha.
        self.single_model = single_model
        self.derived_signals = not single_model
        if single_model:
            todos = [c for cols in self.families.values() for c in cols]
            self.families = {"tudo": todos}
        self._grids: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._alarms: pd.DataFrame | None = None
        self._trips: list[tuple[str, str]] | None = None

    def _csv_path(self) -> Path:
        if self.local_csv:
            return Path(self.local_csv)
        from clearml import Dataset
        root = Path(Dataset.get(dataset_id=self.dataset_id).get_local_copy())
        matches = list(root.rglob(SENSORS_FILE)) or list(root.rglob("*sensores*.csv"))
        if not matches:
            raise FileNotFoundError(f"{SENSORS_FILE} não encontrado em {root}")
        return matches[0]

    def load(self) -> "DataBundle":
        raw = pd.read_csv(self._csv_path(), engine="pyarrow")
        time_col = raw.columns[0]
        raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
        raw = raw.dropna(subset=[time_col]).set_index(time_col).sort_index()

        numeric = raw.apply(pd.to_numeric, errors="coerce")   # status do PI -> NaN
        # faixa por coluna: no modo --single-model as famílias são fundidas em uma,
        # e o nome do grupo deixaria de bater com PHYSICAL_RANGE
        for column, (low, high) in RANGE_BY_COLUMN.items():
            if column in numeric:
                numeric[column] = numeric[column].where(numeric[column].between(low, high))
        self.raw = numeric
        print(f"[dados] {len(numeric):,} amostras | "
              f"{numeric.index.min()} → {numeric.index.max()}", flush=True)
        return self

    # ------------------------------------------------------ alarmes e falhas
    def alarms(self) -> pd.DataFrame:
        """Planilha de alarmes do dataset, com ``ativado`` e ``nivel`` marcados."""
        if getattr(self, "_alarms", None) is not None:
            return self._alarms
        pasta = self._csv_path().parent
        arquivos = list(pasta.rglob(ALARMS_FILE)) or list(pasta.rglob("*alarmes*.csv"))
        if not arquivos:
            print("[aviso] alarmes não encontrados; a lista de falhas ficará vazia")
            self._alarms = pd.DataFrame()
            return self._alarms
        alarmes = pd.read_csv(arquivos[0])
        quando = alarmes.columns[0]
        alarmes[quando] = pd.to_datetime(alarmes[quando], errors="coerce")
        alarmes = alarmes.dropna(subset=[quando]).rename(columns={quando: "quando"})
        alarmes["ativado"] = alarmes["Status"].astype(str).str.startswith("ACT")
        alarmes["nivel"] = (alarmes["Descrição Alarme"].astype(str).str.upper()
                            .str.contains(ALARM_LEVEL_PATTERN, regex=True))
        self._alarms = alarmes.set_index("quando").sort_index()
        return self._alarms

    def stops(self, freq: str = "2min") -> pd.DataFrame:
        """Paradas reais: transição operando→parado que dura >= ``MIN_STOP_HOURS``."""
        _, oper = self.grid(freq)
        operando = oper["in_operation"].astype(bool)
        transicoes = (~operando) & operando.shift(fill_value=False)
        linhas = []
        for inicio in transicoes[transicoes].index:
            adiante = operando.loc[inicio:]
            voltou = adiante[adiante]
            fim = voltou.index[0] if len(voltou) else adiante.index[-1]
            horas = (fim - inicio).total_seconds() / 3600
            if horas >= MIN_STOP_HOURS:
                linhas.append({"inicio": inicio, "fim": fim, "horas": round(horas, 1)})
        return pd.DataFrame(linhas)

    def trips(self, freq: str = "2min") -> list[tuple[str, str]]:
        """Paradas reais que coincidem com alarme de nível — a lista de falhas."""
        if self._trips is not None:
            return self._trips
        alarmes = self.alarms()
        if alarmes.empty:
            return []
        nivel = alarmes[alarmes["ativado"] & alarmes["nivel"]]
        antes, depois = pd.Timedelta(STOP_ALARM_BEFORE), pd.Timedelta(STOP_ALARM_AFTER)
        achados = []
        for _, parada in self.stops(freq).iterrows():
            janela = nivel.loc[parada["inicio"] - antes:parada["inicio"] + depois]
            if janela.empty:
                continue
            sensores = janela["Tag Alarme"].astype(str)
            mecanismos = [MECHANISM_BY_SENSOR[s] for s in sensores
                          if s in MECHANISM_BY_SENSOR]
            achados.append((f"{parada['inicio']:%Y-%m-%d %H:%M}",
                            mecanismos[0] if mecanismos else "indefinido"))
        self._trips = achados
        return achados

    def alarm_events_in_operation(self, nivel: bool = True) -> list[pd.Timestamp]:
        """Ativações de alarme ocorridas com a máquina **operando**.

        Com ``nivel=True`` (padrão) devolve só as de nível, que complementam a
        lista de falhas na limpeza do baseline. Com ``nivel=False`` devolve todas
        as ativações em operação (420 no período) — o conjunto de validação
        secundária: episódios anômalos que não viraram parada, mas em que o score
        deveria subir.
        """
        alarmes = self.alarms()
        if alarmes.empty:
            return []
        candidatos = alarmes[alarmes["ativado"]]
        if nivel:
            candidatos = candidatos[candidatos["nivel"]]
        _, oper = self.grid("2min")
        operando = oper["in_operation"]
        return [t for t in candidatos.index
                if operando.index[0] <= t <= operando.index[-1]
                and bool(operando.asof(t))]

    def alarm_episodes(self, gap: str = ALARM_EPISODE_GAP,
                       freq: str = "2min") -> list[dict]:
        """Episódios de alarme ocorridos **em operação**, agrupados por proximidade.

        Conjunto de avaliação secundária: são 132 episódios (agrupando por 24 h)
        contra 8 falhas catalogadas — 16× mais evidência. Um detector que sobe o
        score antes de episódios anômalos que *não* viraram parada está funcionando,
        mesmo que a métrica de 8 eventos não consiga mostrar isso.
        """
        alarmes = self.alarms()
        if alarmes.empty:
            return []
        ativos = alarmes[alarmes["ativado"]]
        _, oper = self.grid(freq)
        operando = oper["in_operation"]
        dentro = [t for t in ativos.index
                  if operando.index[0] <= t <= operando.index[-1] and bool(operando.asof(t))]
        if not dentro:
            return []
        ativos = ativos.loc[dentro]
        marcas = pd.Series(ativos.index, index=ativos.index)
        grupo = (marcas.diff() > pd.Timedelta(gap)).cumsum()
        episodios = []
        for _, trecho in ativos.groupby(grupo):
            tags = sorted(set(trecho["Tag Alarme"].astype(str)))
            episodios.append({
                "inicio": trecho.index[0], "fim": trecho.index[-1],
                "tags": tags, "n": len(trecho),
                # utilidade só se TODAS as tags do episódio forem de gás
                "utilidade": all(t in UTILITY_TAGS for t in tags),
            })
        return episodios

    def exclusion_events(self) -> list[pd.Timestamp]:
        """Eventos cujos dias anteriores devem sair do baseline.

        São as falhas detectadas pela varredura mais as ativações de alarme de
        nível ocorridas **em operação** (as que dispararam com a máquina já parada
        são consequência da parada, não sintoma — 94% dos casos).
        """
        falhas = [pd.Timestamp(when) for when, _ in self.trips()]
        return sorted(set(falhas) | set(self.alarm_events_in_operation()))

    def frozen_veto(self, freq: str, minutos: int = 30,
                    limite_natural: float = 0.05) -> pd.DataFrame:
        """Por família: True onde algum sensor **dela** está congelado.

        Instrumento travado em um valor não é anomalia do equipamento — é falha de
        instrumentação, e antes era contada como anomalia. O veto anula o score da
        família nesses instantes. (Leitura fora da faixa física já vira NaN na
        carga, e NaN já zera o score.)

        Dois cuidados aprendidos medindo:

        * o veto é **por família** — um sensor de pressão travado não pode anular
          o score de temperatura;
        * sensores **constantes por natureza** ficam de fora do teste. O
          `PI_0319` (pressão do gás de partida) fica parado 11,8% do tempo em
          operação porque é assim que ele funciona; incluí-lo vetava 14,5% de todo
          o tempo útil. Fica de fora quem passa de ``limite_natural`` do tempo
          congelado — critério auto-calibrado, sem lista fixa.
        """
        chave = f"veto_{freq}_{minutos}"
        if chave in self._grids:
            return self._grids[chave]
        frame, oper = self.grid(freq)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        janela = max(int(minutos * 60 / step), 2)
        operando = oper["stable"]
        vetos = {}
        for familia, colunas in self.families.items():
            cols = [c for c in colunas if c in frame.columns]
            parado = (frame[cols].rolling(janela, min_periods=janela).std() == 0)
            fracao = parado[operando].mean()
            naturais = list(fracao[fracao > limite_natural].index)
            if naturais:
                print(f"[veto] {familia}: {naturais} são constantes por natureza "
                      f"({', '.join(f'{fracao[c]:.1%}' for c in naturais)}) e ficam fora do teste")
            uteis = [c for c in cols if c not in naturais]
            vetos[familia] = parado[uteis].any(axis=1).fillna(False) if uteis \
                else pd.Series(False, index=frame.index)
        veto = pd.DataFrame(vetos)
        self._grids[chave] = veto
        return veto

    def grid(self, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(sensores, operabilidade) reamostrados; cacheado por frequência."""
        if freq in self._grids:
            return self._grids[freq]
        frame = self.raw if freq in ("30s", "30S") else self.raw.resample(freq).median()
        running = frame[RUNNING_TAG]
        in_operation = running.ge(RUNNING_THRESHOLD).fillna(False)
        em_carga = exhaust_on_load(frame)
        if EXHAUST_GATE == "operation":
            in_operation = in_operation & em_carga
        starts = in_operation & ~in_operation.shift(fill_value=False)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        window = max(int(pd.Timedelta(STARTUP_EXCLUDE).total_seconds() / step), 1)
        stable = in_operation & ~starts.rolling(window, min_periods=1).max().astype(bool)
        if EXHAUST_GATE == "stable":
            stable = stable & em_carga
        oper = pd.DataFrame({"in_operation": in_operation, "stable": stable}, index=frame.index)
        self._grids[freq] = (frame, oper)
        return self._grids[freq]


# --------------------------------------------------------------- operabilidade
def post_start_mask(oper: pd.DataFrame, index: pd.DatetimeIndex,
                    silence: str | None = None) -> pd.Series:
    """True onde o alerta está silenciado por proximidade de uma partida.

    A partida é a transição para ``in_operation``, e não para ``stable``: quem
    define o transiente é a máquina ligando, não o fim do descarte de partida.
    """
    janela = pd.Timedelta(silence or POST_START_SILENCE)
    if janela <= pd.Timedelta(0):
        return pd.Series(False, index=index)
    operando = oper["in_operation"].astype(bool)
    partidas = operando & ~operando.shift(fill_value=False)
    silenciado = pd.Series(False, index=operando.index)
    for inicio in partidas[partidas].index:
        silenciado.loc[inicio:inicio + janela] = True
    return silenciado.reindex(index, fill_value=False)


def exhaust_on_load(frame: pd.DataFrame) -> pd.Series:
    """Turbina em carga pela temperatura de exaustão (ver ``EXHAUST_TAG``).

    Ausência de leitura é tratada como "não sei" e mantém o que o RUNNING_A diz:
    o marcador só serve para DESQUALIFICAR operação aparente, nunca para negar
    operação por falta de dado. Na série de 16 meses isso muda 9 amostras.
    """
    if EXHAUST_TAG not in frame.columns:
        print(f"[aviso] {EXHAUST_TAG} ausente: o porteiro de exaustão fica inativo")
        return pd.Series(True, index=frame.index)
    t5 = frame[EXHAUST_TAG]
    em_carga = (t5.ge(EXHAUST_ON_C) | t5.isna()).fillna(True)
    if not EXHAUST_MIN_H:
        return em_carga
    passo = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
    minimo = pd.Timedelta(hours=EXHAUST_MIN_H)
    frio = ~em_carga
    bloco = (frio != frio.shift(fill_value=False)).cumsum()
    for _, trecho in frio[frio].groupby(bloco[frio]):
        if (len(trecho) * passo) < minimo.total_seconds():
            em_carga.loc[trecho.index] = True     # curto: é rampa, não regime frio
    return em_carga


# ------------------------------------------------------------------- eventos
def merge_events(trips: list[tuple[str, str]], gap: str = "24h") -> list[dict]:
    """Agrupa trips próximos em um evento físico (evita contar a mesma falha 2x)."""
    ordered = sorted((pd.Timestamp(when), mech) for when, mech in trips)
    events: list[dict] = []
    for when, mech in ordered:
        if events and (when - events[-1]["fim"]) <= pd.Timedelta(gap):
            events[-1]["fim"] = when
            events[-1]["trips"].append(str(when))
            continue
        events.append({"inicio": when, "fim": when, "mecanismo": mech,
                       "trips": [str(when)]})
    return events


# ---------------------------------------------------------------- avaliação
# Limite de SILÊNCIO do detector, em horas de OPERAÇÃO sem emitir nenhum alerta.
# Calibrado em 19/08/2026: a cadência natural dos 132 episódios de alarme tem
# mediana de 50 h e p90 de 138 h de operação, mas um detector preso ao teto de
# 1 FP/mês só pode emitir ~12 alertas/ano, o que dá ~660 h de espaçamento médio.
# O limite fica em ~3x isso: acima daí o detector não está calado por a máquina
# estar saudável, está cego. O campeão da task 61b6b109 fica 3.186 h em silêncio.
SILENCE_LIMIT_H = 2000.0
# Corte que separa a série em duas metades para reportar detecção distribuída.
# 5 dos 8 eventos ficam antes, 3 depois.
EVAL_SPLIT = "2025-10-01"
# Linhas enviadas ao gráfico de tabela do ClearML. A grade toda vai no artefato;
# aqui o teto existe só para o evento não estourar o limite de 15 MB do servidor.
TABLE_PLOT_ROWS = 300


@dataclass(frozen=True)
class BaselinePolicy:
    """Como escolher o trecho de histórico que treina o modelo de cada mês.

    Substitui o par ``{movel, acumulativo}``, que era um caso degenerado desta
    política: móvel de N horas é ``window_hours=N``; acumulativo é tudo ``None``.

    Os três limites são independentes e o mais restritivo vence:

    ``window_hours``
        horas de operação ELEGÍVEL (já descontadas parada, transiente de partida
        e as exclusões de evento/alarme). É a unidade de exposição correta:
        medido em 19/08/2026, um mês de calendário entrega de 133,8 h a 691,1 h
        (5,17x), e a escassez é ENDÓGENA às falhas — corr(falhas nos 45 d
        anteriores, horas elegíveis nos 30 d) = -0,61. Contar em meses encolhe o
        baseline exatamente nos retreinos que vêm depois de um reparo.
    ``window_days``
        dias de CALENDÁRIO. Existe para reproduzir a parametrização em meses e
        poder comparar as duas de frente, não porque seja preferível.
    ``max_age_days``
        teto de idade. Necessário porque uma janela em horas recua muito mais no
        calendário do que se imagina: 400 h elegíveis exigem recuar de 16,8 a
        64,0 dias dependendo do retreino (3,8x), e sazonalidade e reparo agem em
        tempo de calendário, não em hora de máquina.
    """

    window_hours: float | None = None
    window_days: float | None = None
    max_age_days: float | None = None
    # piso: abaixo disso o retreino é marcado inválido em vez de treinar com
    # amostra insuficiente. 100 h de operação a 2 min = 3.000 amostras.
    min_hours: float = 100.0

    @property
    def label(self) -> str:
        if self.window_hours is None and self.window_days is None:
            return "acum"
        partes = []
        if self.window_hours is not None:
            partes.append(f"{self.window_hours:g}h")
        if self.window_days is not None:
            partes.append(f"{self.window_days:g}d")
        if self.max_age_days is not None:
            partes.append(f"idade{self.max_age_days:g}d")
        return "+".join(partes)

    @property
    def modo_legado(self) -> str:
        """Rótulo compatível com os resultados anteriores (notebook 02)."""
        if self.window_hours is None and self.window_days is None:
            return "acumulativo"
        if self.window_hours is not None and self.window_days is None:
            return "hibrido" if self.max_age_days is not None else "movel"
        return "dias"

    def select(self, elegiveis: pd.DatetimeIndex, start: pd.Timestamp,
              step_s: float) -> tuple[pd.DatetimeIndex, dict]:
        """Trecho de treino para o retreino de ``start``, e o diagnóstico dele.

        Vetorizado: cada limite vira um índice por busca binária sobre o eixo
        de tempo elegível, e o corte final é o máximo deles. Nenhum laço por
        amostra — o eixo tem 1,4 M pontos na grade de 30 s.
        """
        marcas = elegiveis.values
        fim = int(np.searchsorted(marcas, np.datetime64(start), side="right"))
        inicio = 0
        if self.window_hours is not None:
            inicio = max(inicio, fim - int(self.window_hours * 3600 / step_s))
        for dias in (self.window_days, self.max_age_days):
            if dias is not None:
                corte = np.datetime64(start - pd.Timedelta(days=float(dias)))
                inicio = max(inicio, int(np.searchsorted(marcas, corte, side="left")))
        inicio = max(inicio, 0)
        escolhidos = elegiveis[inicio:fim]

        horas = len(escolhidos) * step_s / 3600
        pedido = self.window_hours
        diag = {
            "horas": horas,
            "amostras": len(escolhidos),
            "valido": horas >= self.min_hours,
            # truncado = o histórico não deu para preencher a janela pedida.
            # Importa muito: com 3.000 h, 6 dos 15 retreinos de 2025/26 ficam
            # truncados e ali o braço "janela longa" vira acumulativo disfarçado.
            "truncado": bool(pedido is not None and horas < pedido * 0.999),
            "span_dias": ((escolhidos[-1] - escolhidos[0]).total_seconds() / 86400
                          if len(escolhidos) > 1 else 0.0),
        }
        return escolhidos, diag


@dataclass(frozen=True)
class Trial:
    model: str
    grid: str
    # Política de baseline (ver BaselinePolicy). Substituiu o par
    # (baseline_hours, baseline_mode): a busca de 31.104 configs varreu só 300 h
    # e 400 h — 1,33x de amplitude — e pulou para o acumulativo (~8.000 h). A
    # década inteira no meio ficou intocada, e é onde está o resultado: uma
    # janela de 3.000 h dá 6/8 eventos com 0,94 FP/mês, contra 1,54 FP/mês do
    # melhor 6/8 da grade antiga.
    baseline: BaselinePolicy
    ewma: str
    # dias antes de cada evento conhecido removidos do baseline (0 = sem limpeza)
    exclude_days: int
    # horas em torno de CADA ativação de alarme removidas do baseline (0 = nenhuma).
    # Custo medido: ~5% do baseline com 1 h.
    exclude_alarm_h: float
    # Limiar do baseline suavizado. Um único valor aplica-se a todos os sinais;
    # uma tupla define um limiar por sinal, na ordem de SIGNAL_ORDER — isso abre
    # a fronteira, porque as famílias têm estabilidade muito diferente.
    # O SIGNIFICADO depende de threshold_kind:
    #   "percentil"  -> 99.9 = p99,9 do baseline (parametrização histórica)
    #   "k_maiores"  -> 30 = o 30º maior score do baseline
    # Motivo de existir o segundo: medido em 19/08/2026, em janela curta os
    # percentis de 99,9 a 99,995 são o MESMO limiar (o máximo fica só 1,79%
    # acima do p99,9), então 4 dos 6 níveis da grade eram um só. Já "k-ésimo
    # maior" mantém o orçamento de excedências fixo qualquer que seja a janela,
    # o que torna políticas de tamanhos diferentes comparáveis.
    threshold: float | tuple[float, ...]
    sustain: str
    confirm: int          # nº de sinais simultâneos exigidos (1=atenção, 2=confirmado)
    # tempo mínimo que o alerta CONFIRMADO precisa ficar de pé para ser emitido.
    # Sem isso, a detecção de 17/03/2025 durava 9 minutos — antecedência de 29 h
    # no papel, invisível em um plantão.
    min_alert: str = "0min"
    # ver o comentário de `threshold`: "percentil" ou "k_maiores"
    threshold_kind: str = "percentil"

    def label(self) -> str:
        thr = (self.threshold if isinstance(self.threshold, float)
               else "/".join(str(t) for t in self.threshold))
        sigla = "q" if self.threshold_kind == "percentil" else "k"
        return (f"{self.model}|{self.grid}|b{self.baseline.label}"
                f"|excl{self.exclude_days}d|alm{self.exclude_alarm_h}h"
                f"|ewma{self.ewma}|{sigla}{thr}|sust{self.sustain}"
                f"|conf{self.confirm}|min{self.min_alert}")

    def threshold_for(self, signal: str, order: list[str]) -> float:
        if isinstance(self.threshold, (int, float)):
            return float(self.threshold)
        return float(self.threshold[order.index(signal) % len(self.threshold)])


@dataclass
class TrialResult:
    trial: dict
    eventos_detectados: int
    eventos_total: int
    lead_medio_h: float | None
    leads: dict
    fp_episodios: int
    fp_por_mes: float
    # horas totais de alerta falso por mês: um episódio de 10 min e um de uma
    # semana contam igual na métrica de episódios, e não são a mesma coisa para
    # quem está de plantão
    fp_horas_por_mes: float
    # (c) episódios que coincidem com alarme ativo: não são falso positivo nem
    # detecção — são "explicados". Reportados à parte, sem mexer no fp_episodios,
    # para que os números anteriores continuem comparáveis.
    fp_explicados: int = 0
    fp_por_mes_liquido: float = 0.0
    # (2) avaliação secundária: episódios de alarme em operação antecipados
    alarmes_antecipados: int = 0
    alarmes_total: int = 0
    alarmes_utilidade_antecipados: int = 0
    alarmes_utilidade_total: int = 0
    dias_avaliados: float = 0.0
    episodios_totais: int = 0
    baseline_amostras: int = 0    # tamanho médio do baseline (mede o custo da exclusão)
    # --- diagnósticos de política de baseline (19/08/2026) ---
    baseline_horas_medio: float = 0.0      # horas de operação elegível por retreino
    baseline_span_dias_medio: float = 0.0  # quanto isso recua no CALENDÁRIO
    retreinos_truncados: int = 0           # histórico não deu para encher a janela
    retreinos_invalidos: int = 0           # abaixo de min_hours
    # --- detecção distribuída no tempo ---
    # Reportado por metade porque a média esconde o essencial: a config campeã da
    # busca antiga detecta 5 dos 5 eventos da 1a metade e 0 dos 3 da 2a.
    det_1a_metade: int = 0
    det_2a_metade: int = 0
    # --- liveness, medido na série de ALERTAS (não usa rótulo de evento) ---
    # Este é o teste que pega o detector que emudece. Os 8 rótulos não servem:
    # medido em 19/08/2026, dentro do teto de 1 FP/mês a máscara "detectou algo
    # na 2a metade" é BIT-A-BIT idêntica a "detectou 26/02/2026" — zero das
    # 11.395 configs aprovadas pegam 04/11/2025 ou 09/12/2025. Qualquer critério
    # baseado nos eventos é, na prática, um teste de um único evento.
    maior_silencio_h: float = 0.0          # horas de OPERAÇÃO sem nenhum alerta
    trimestres_com_alerta: int = 0
    trimestres_avaliados: int = 0
    aprovado: bool = False
    # aprovado exige o teto de FP; vivo exige também não estar cego
    vivo: bool = False


class WalkForwardEvaluator:
    """Protocolo temporal: re-baseline a cada mês, pontua o mês, avalia por evento."""

    def __init__(self, bundle: DataBundle, eval_start: str, eval_end: str,
                 max_fp_per_month: float = 1.0) -> None:
        self.bundle = bundle
        self.months = [str(p) for p in pd.period_range(eval_start, eval_end, freq="M")]
        self.max_fp_per_month = max_fp_per_month
        self.events = merge_events(bundle.trips(), EVENT_GAP)
        self.exclusion_events = bundle.exclusion_events()
        alarmes = bundle.alarms()
        self.alarm_events = (list(alarmes[alarmes["ativado"]].index)
                             if not alarmes.empty else [])
        self.alarm_eps = bundle.alarm_episodes()
        self._cache: dict[tuple, dict] = {}
        self._diag: dict[tuple, list[dict]] = {}
        self._diag_calc: dict[tuple, list[dict]] = {}
        # Cache do pós-processamento. Medido em 19/08/2026: a etapa "barata"
        # custava ~5,6 h da busca `full` contra ~1,2 h de TODOS os ajustes de
        # modelo, porque `smooth()` era refeito 144x idêntico por par
        # (chave, ewma) — os eixos threshold x sustain x confirm x min_alert só
        # mudam o que vem DEPOIS da suavização. Guardar o teste suavizado e a
        # cauda ordenada do baseline torna o eixo do limiar quase gratuito.
        self._suave: dict[tuple, dict] = {}
        self._suave_ordem: list[tuple] = []      # FIFO, para não estourar a RAM
        self._stops_cache: dict[str, list] = {}
        uteis = sum(1 for e in self.alarm_eps if e["utilidade"])
        print(f"[alarmes] {len(self.alarm_eps)} episódios de alarme em operação "
              f"({uteis} só de utilidade/gás) servirão de avaliação secundária", flush=True)
        print(f"[falhas] {len(self.events)} eventos físicos derivados de "
              f"{len(bundle.trips())} trips (parada >= {MIN_STOP_HOURS} h + alarme de nível)",
              flush=True)
        print(f"[exclusão] {len(self.exclusion_events)} eventos e {len(self.alarm_events)} "
              f"ativações de alarme servirão para limpar o baseline", flush=True)

    def _baseline_mask(self, index: pd.DatetimeIndex, exclude_days: int,
                       exclude_alarm_h: float = 0.0) -> pd.Series:
        """False nos instantes a até `exclude_days` antes de um evento conhecido
        e a até `exclude_alarm_h` de qualquer ativação de alarme."""
        keep = pd.Series(True, index=index)
        if exclude_alarm_h > 0 and len(self.alarm_events):
            # distância ao alarme mais próximo por busca binária: a matriz
            # completa (80 mil instantes × 3.757 alarmes) não cabe em memória
            marcas = np.sort(np.array([t.to_datetime64() for t in self.alarm_events]))
            alvo = index.values.astype("datetime64[ns]")
            pos = np.searchsorted(marcas, alvo)
            esquerda = marcas[np.clip(pos - 1, 0, len(marcas) - 1)]
            direita = marcas[np.clip(pos, 0, len(marcas) - 1)]
            dist = np.minimum(np.abs(alvo - esquerda), np.abs(alvo - direita))
            keep &= dist > np.timedelta64(int(exclude_alarm_h * 3600), "s")
        if exclude_days <= 0:
            return keep
        for evento in self.exclusion_events:
            inicio = evento - pd.Timedelta(days=exclude_days)
            keep.loc[(index >= inicio) & (index <= evento)] = False
        return keep

    # -------- etapa caro: score bruto por (modelo, grade, baseline) -----------
    def raw_scores(self, model: str, grid: str, baseline: BaselinePolicy,
                   exclude_days: int = 0, exclude_alarm_h: float = 0.0) -> dict:
        """Walk-forward: para cada mês, devolve o score do baseline e do teste.

        Guardar os dois permite que o limiar seja um quantil do baseline **já
        suavizado** com o mesmo EWMA do teste — sem essa correspondência, o
        limiar não controla o falso positivo de verdade.

        Qual trecho de histórico entra no treino é decidido por ``baseline``
        (ver :class:`BaselinePolicy`). Esta é a etapa CARA: 15 retreinos x 2
        famílias = 30 ajustes sklearn por chave de cache.
        """
        key = (model, grid, baseline, exclude_days, exclude_alarm_h)
        if key in self._cache:
            return self._cache[key]
        frame, oper = self.bundle.grid(grid)
        stable = oper["stable"]
        veto = self.bundle.frozen_veto(grid)      # sensor congelado não é anomalia
        veto_qualquer = veto.any(axis=1)          # para os sinais derivados
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        # elegibilidade do baseline calculada UMA vez para todo o período
        elegivel = stable & self._baseline_mask(frame.index, exclude_days, exclude_alarm_h)
        indices_elegiveis = elegivel[elegivel].index

        derivados = (("mancal_spread", self._spread), ("selagem_z", self._seal)) \
            if getattr(self.bundle, "derived_signals", True) else ()
        names = list(self.bundle.families) + [n for n, _ in derivados]
        parts: dict[str, list[tuple[pd.Series, pd.Series]]] = {n: [] for n in names}
        diagnosticos: list[dict] = []

        for month in self.months:
            start = pd.Timestamp(month + "-01")
            end = start + pd.offsets.MonthBegin(1)
            escolhidos, diag = baseline.select(indices_elegiveis, start, step)
            diag["mes"] = month
            if len(escolhidos) < 1000 or not diag["valido"]:
                diagnosticos.append(diag)
                continue
            base_stable = frame.loc[escolhidos]
            test = frame.loc[start:end - pd.Timedelta(seconds=1)]
            if test.empty:
                continue
            diagnosticos.append(diag)
            estavel_teste = stable.reindex(test.index, fill_value=False)

            for name, columns in self.bundle.families.items():
                cols = [c for c in columns if c in frame.columns]
                try:
                    scorer = FamilyScorer(model).fit(base_stable[cols])
                except ValueError:
                    continue
                # cada família é mascarada pelo veto DELA, não pelo das outras
                mask_familia = estavel_teste & ~veto[name].reindex(test.index,
                                                                   fill_value=False)
                parts[name].append((scorer.baseline_score,
                                    scorer.score(test[cols]).where(mask_familia)))

            mask = estavel_teste & ~veto_qualquer.reindex(test.index, fill_value=False)

            for name, derivado in derivados:
                base_serie = derivado(base_stable).dropna()
                if base_serie.empty:
                    continue
                centro = base_serie.median()
                escala = (base_serie - centro).abs().median() * 1.4826
                if not escala or not np.isfinite(escala):
                    continue
                parts[name].append((
                    ((base_serie - centro) / escala).abs(),
                    ((derivado(test) - centro) / escala).abs().where(mask)))

        self._cache[key] = parts
        self._diag[key] = diagnosticos
        return parts

    @staticmethod
    def _spread(frame: pd.DataFrame) -> pd.Series:
        peers = frame[[c for c in BEARING_PEERS if c in frame.columns]].median(axis=1)
        return frame[BEARING_TARGET] - peers

    @staticmethod
    def _seal(frame: pd.DataFrame) -> pd.Series:
        """Diferencial de selagem cru — o z robusto é aplicado por quem chama."""
        if SEAL_TARGET not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return frame[SEAL_TARGET]

    # -------- etapa barata: limiar/persistência/confirmação -----------------
    def diagnostico_baseline(self, grid: str, baseline: BaselinePolicy,
                            exclude_days: int, exclude_alarm_h: float) -> list[dict]:
        """Tamanho e alcance do baseline por retreino, SEM ajustar modelo.

        Existe separado de ``raw_scores`` porque é barato (só busca binária) e
        porque o replay injeta ``parts`` de cache em disco, caminho em que
        ``raw_scores`` não roda e o diagnóstico ficaria vazio.
        """
        chave = (grid, baseline, exclude_days, exclude_alarm_h)
        if chave in self._diag_calc:
            return self._diag_calc[chave]
        frame, oper = self.bundle.grid(grid)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        elegivel = oper["stable"] & self._baseline_mask(frame.index, exclude_days,
                                                        exclude_alarm_h)
        indices = elegivel[elegivel].index
        saida = []
        for month in self.months:
            _, diag = baseline.select(indices, pd.Timestamp(month + "-01"), step)
            diag["mes"] = month
            saida.append(diag)
        self._diag_calc[chave] = saida
        return saida

    def _suavizado(self, chave_fit: tuple, parts: dict, ewma: str) -> dict:
        """Baseline e teste já suavizados, por sinal e por retreino.

        Do baseline guarda-se só a CAUDA ordenada (decrescente) e a contagem de
        pontos válidos, o que basta para qualquer percentil >= 98,8 ou qualquer
        k-ésimo maior até 8.192 — e custa uma fração da memória do vetor todo.
        """
        chave = (*chave_fit, ewma)
        if chave in self._suave:
            return self._suave[chave]
        halflife = pd.Timedelta(ewma)

        def suaviza(serie: pd.Series) -> pd.Series:
            return serie.ewm(halflife=halflife, times=serie.index).mean()

        pronto: dict[str, list] = {}
        for name, meses in parts.items():
            if not meses:
                continue
            por_mes = []
            for baseline, test in meses:
                valores = suaviza(baseline).to_numpy(dtype=float)
                valores = valores[np.isfinite(valores)]
                n = int(valores.size)
                if n == 0:
                    continue
                # cauda suficiente para p>=98,8 e para k<=8.192
                m = min(n, max(8192, int(n * 0.012) + 16))
                cauda = np.sort(np.partition(valores, n - m)[n - m:])[::-1]
                por_mes.append((cauda, n, suaviza(test)))
            if por_mes:
                pronto[name] = por_mes

        self._suave[chave] = pronto
        self._suave_ordem.append(chave)
        while len(self._suave_ordem) > 4:            # 4 chaves cabem em RAM
            self._suave.pop(self._suave_ordem.pop(0), None)
        return pronto

    @staticmethod
    def _limite(cauda: np.ndarray, n: int, valor: float, kind: str) -> float:
        """Limiar a partir da cauda ordenada decrescente do baseline.

        ``percentil`` reproduz ``np.nanpercentile(..., interpolation="linear")``
        exatamente; ``k_maiores`` devolve o k-ésimo maior score, que mantém o
        orçamento de excedências fixo qualquer que seja o tamanho da janela.
        """
        if kind == "k_maiores":
            k = max(int(valor), 1)
            return float(cauda[min(k, cauda.size) - 1])
        # Posição virtual do percentil no vetor ASCENDENTE de n pontos. A ordem
        # das operações importa: o numpy faz (p/100)*(n-1), e (n-1)*p/100 dá
        # outro float na última casa — o que já basta para o limiar mudar e os
        # resultados anteriores não reproduzirem.
        h = (float(valor) / 100.0) * (n - 1)
        baixo = int(np.floor(h))
        frac = h - baixo
        # índice equivalente na cauda decrescente: asc[i] == cauda[n-1-i]
        j = n - 1 - baixo
        if j >= cauda.size:                     # fora da cauda guardada
            return float(cauda[-1])
        inferior = float(cauda[j])
        if frac == 0.0 or j == 0:
            return inferior
        superior = float(cauda[j - 1])
        # A forma da interpolação segue o `_lerp` do numpy, que troca de
        # expressão em frac=0,5. Não é preciosismo: sem isso o limiar difere na
        # 12a casa e os resultados das buscas anteriores deixam de reproduzir
        # bit a bit.
        if frac >= 0.5:
            return superior - (superior - inferior) * (1.0 - frac)
        return inferior + (superior - inferior) * frac

    def evaluate(self, trial: Trial) -> TrialResult:
        chave_fit = (trial.model, trial.grid, trial.baseline,
                     trial.exclude_days, trial.exclude_alarm_h)
        parts = self.raw_scores(trial.model, trial.grid, trial.baseline,
                                trial.exclude_days, trial.exclude_alarm_h)
        suave = self._suavizado(chave_fit, parts, trial.ewma)
        frame, oper = self.bundle.grid(trial.grid)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        n_sustain = max(int(pd.Timedelta(trial.sustain).total_seconds() / step), 1)

        order = list(suave)
        active = []
        for name in order:
            alvo = trial.threshold_for(name, order)
            flags = []
            for cauda, n_base, teste in suave[name]:
                limit = self._limite(cauda, n_base, alvo, trial.threshold_kind)
                hits = (teste > limit).astype(int)
                flags.append(hits.rolling(n_sustain, min_periods=n_sustain).sum() >= n_sustain)
            active.append(pd.concat(flags).sort_index())

        if not active:
            return TrialResult(asdict(trial), 0, len(self.events), None, {}, 0, 0.0, 0.0)
        combined = pd.concat(active, axis=1).fillna(False)
        alert = combined.sum(axis=1) >= trial.confirm
        # alerta só é EMITIDO se ficar de pé o tempo mínimo exigido
        n_min = max(int(pd.Timedelta(trial.min_alert).total_seconds() / step), 1)
        if n_min > 1:
            alert = (alert.astype(int).rolling(n_min, min_periods=n_min).sum() >= n_min)
            alert = alert.fillna(False)
        silenciado = post_start_mask(oper, alert.index)
        alert = alert & ~silenciado

        episodes = self._episodes(alert)
        if trial.grid not in self._stops_cache:   # só depende da grade
            self._stops_cache[trial.grid] = self._stops(oper)
        stops = self._stops_cache[trial.grid]
        detected, leads, matched = {}, {}, set()
        for event in self.events:
            window_start = event["inicio"] - pd.Timedelta(DETECTION_WINDOW)
            inside = [ep for ep in episodes if window_start <= ep <= event["inicio"]]
            if inside:
                lead = (event["inicio"] - inside[0]).total_seconds() / 3600
                detected[str(event["inicio"])] = True
                leads[str(event["inicio"])] = round(lead, 1)
                matched.update(inside)

        false_positives = []
        for ep in episodes:
            if ep in matched:
                continue
            horizon = ep + pd.Timedelta(DETECTION_WINDOW)
            near_trip = any(ep <= e["inicio"] <= horizon for e in self.events)
            near_stop = any(ep <= s <= horizon for s in stops)
            if not (near_trip or near_stop):
                false_positives.append(ep)

        # (c) episódio que coincide com alarme ativo é "explicado": não conta como
        # acerto nem como falso alarme. Medido em agosto/2025: dois episódios de
        # 145 h e 164 h coincidem com alarme de suprimento de gás — o modelo estava
        # reagindo a uma perturbação real que não virou parada.
        explica = pd.Timedelta(ALARM_EXPLAIN_WINDOW)
        marcas_alarme = [pd.Timestamp(t) for t in self.alarm_events]
        def explicado(ep):
            return any(abs((ep - t).total_seconds()) <= explica.total_seconds()
                       for t in marcas_alarme)
        fp_explicados = [ep for ep in false_positives if explicado(ep)]
        fp_liquidos = [ep for ep in false_positives if ep not in set(fp_explicados)]

        # (2) avaliação secundária: episódio de alarme antecipado por um alerta
        janela = pd.Timedelta(DETECTION_WINDOW)
        def antecipado(inicio):
            return any(inicio - janela <= ep <= inicio for ep in episodes)
        alarmes_ok = [e for e in self.alarm_eps if antecipado(e["inicio"])]
        util = [e for e in self.alarm_eps if e["utilidade"]]
        util_ok = [e for e in util if antecipado(e["inicio"])]

        # o tempo silenciado sai do denominador: não é oportunidade de errar
        vigiando = oper["stable"].reindex(alert.index, fill_value=False) & ~silenciado
        days = float(vigiando.sum()) * step / 86400
        fp_month = len(false_positives) / max(days, 1) * 30
        # horas de alerta falso: soma a duração dos episódios contados como FP
        falsos = set(false_positives)
        horas_fp = 0.0
        for inicio, fim in self._spans(alert):
            if inicio in falsos:
                horas_fp += (fim - inicio).total_seconds() / 3600
        fp_horas_mes = horas_fp / max(days, 1) * 30

        # --- liveness: o detector continua vivo ao longo da série? ---
        # Medido no relógio de OPERAÇÃO, porque silêncio com a máquina parada não
        # é cegueira. Inclui as bordas: ficar mudo desde o início ou até o fim
        # conta igual a um vazio no meio.
        operando = oper["in_operation"].reindex(alert.index, fill_value=False)
        relogio = operando.cumsum().to_numpy(dtype=float) * step / 3600
        if len(relogio):
            marcas = [float(relogio[alert.index.get_indexer([ep])[0]]) for ep in episodes]
            bordas = [0.0, *marcas, float(relogio[-1])]
            maior_silencio = float(np.max(np.diff(bordas))) if len(bordas) > 1 else 0.0
        else:
            maior_silencio = 0.0

        trimestres = pd.PeriodIndex(alert.index[operando.to_numpy()], freq="Q").unique()
        com_alerta = pd.PeriodIndex(pd.DatetimeIndex(episodes), freq="Q").unique() \
            if episodes else pd.PeriodIndex([], freq="Q")

        # --- detecção por metade da série ---
        corte = pd.Timestamp(EVAL_SPLIT)
        det_1a = sum(1 for k in leads if pd.Timestamp(k) < corte)
        det_2a = sum(1 for k in leads if pd.Timestamp(k) >= corte)

        # --- diagnóstico da política de baseline ---
        diags = self._diag.get(chave_fit) or self.diagnostico_baseline(
            trial.grid, trial.baseline, trial.exclude_days, trial.exclude_alarm_h)
        horas_medio = float(np.mean([d["horas"] for d in diags])) if diags else 0.0
        span_medio = float(np.mean([d["span_dias"] for d in diags])) if diags else 0.0

        # Colunas derivadas para o notebook 02 continuar lendo a grade nova com o
        # mesmo código que lê a antiga.
        trial_dict = asdict(trial)
        trial_dict["baseline_label"] = trial.baseline.label
        trial_dict["baseline_hours"] = trial.baseline.window_hours
        trial_dict["baseline_mode"] = trial.baseline.modo_legado

        result = TrialResult(
            trial=trial_dict, eventos_detectados=int(len(leads)),
            eventos_total=int(len(self.events)),
            lead_medio_h=round(float(np.mean(list(leads.values()))), 1) if leads else None,
            leads={k: float(v) for k, v in leads.items()},
            fp_episodios=int(len(false_positives)), fp_por_mes=float(round(fp_month, 2)),
            fp_horas_por_mes=float(round(fp_horas_mes, 1)),
            fp_explicados=int(len(fp_explicados)),
            fp_por_mes_liquido=float(round(len(fp_liquidos) / max(days, 1) * 30, 2)),
            alarmes_antecipados=int(len(alarmes_ok)), alarmes_total=int(len(self.alarm_eps)),
            alarmes_utilidade_antecipados=int(len(util_ok)),
            alarmes_utilidade_total=int(len(util)),
            dias_avaliados=float(round(days, 1)), episodios_totais=int(len(episodes)),
            baseline_amostras=int(np.mean([len(b) for b, _ in next(iter(parts.values()))]))
            if parts and next(iter(parts.values())) else 0,
            baseline_horas_medio=round(horas_medio, 1),
            baseline_span_dias_medio=round(span_medio, 1),
            retreinos_truncados=int(sum(1 for d in diags if d["truncado"])),
            retreinos_invalidos=int(sum(1 for d in diags if not d["valido"])),
            det_1a_metade=int(det_1a), det_2a_metade=int(det_2a),
            maior_silencio_h=round(maior_silencio, 1),
            trimestres_com_alerta=int(len(com_alerta)),
            trimestres_avaliados=int(len(trimestres)),
        )
        result.aprovado = bool(result.fp_por_mes <= self.max_fp_per_month)
        # "vivo" separa o detector calado do detector cego. Não substitui o teto
        # de FP, soma a ele: uma config pode ter FP baixíssimo justamente por
        # nunca alarmar, e era isso que a busca antiga premiava.
        result.vivo = bool(result.maior_silencio_h <= SILENCE_LIMIT_H
                           and result.trimestres_com_alerta >= max(
                               result.trimestres_avaliados - 1, 1))
        return result

    @staticmethod
    def _episodes(alert: pd.Series) -> list[pd.Timestamp]:
        active = alert.fillna(False)
        if not active.any():
            return []
        stamps = active[active].index.to_series()
        return [stamps.iloc[0]] + list(stamps[stamps.diff() > pd.Timedelta(EPISODE_GAP)])

    @staticmethod
    def _spans(alert: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(início, fim) de cada episódio — mesmo agrupamento de ``_episodes``."""
        ativo = alert.fillna(False)
        if not ativo.any():
            return []
        marcas = ativo[ativo].index.to_series()
        grupo = (marcas.diff() > pd.Timedelta(EPISODE_GAP)).cumsum()
        return [(t.iloc[0], t.iloc[-1]) for _, t in marcas.groupby(grupo)]

    @staticmethod
    def _stops(oper: pd.DataFrame) -> list[pd.Timestamp]:
        """Paradas REAIS (>= MIN_STOP_HOURS).

        Antes devolvia toda transição 1→0, inclusive as quedas de telemetria de
        90 s do RUNNING_A — e a regra "não conta como falso positivo se uma parada
        vem logo depois" perdoava alertas por paradas que nunca existiram.
        """
        running = oper["in_operation"].astype(bool)
        stopping = (~running) & running.shift(fill_value=False)
        reais = []
        for inicio in stopping[stopping].index:
            adiante = running.loc[inicio:]
            voltou = adiante[adiante]
            fim = voltou.index[0] if len(voltou) else adiante.index[-1]
            if (fim - inicio).total_seconds() / 3600 >= MIN_STOP_HOURS:
                reais.append(inicio)
        return reais


# --------------------------------------------------------------------- busca
# `thresholds` são PERCENTIS do baseline suavizado. p99 deixa ~1% do tempo de
# baseline acima do limiar; p99,97 deixa ~0,03% — é aí que mora o regime de
# baixo falso positivo que o projeto exige.
# Políticas de baseline da varredura. Escala aproximadamente logarítmica em
# horas (razão ~1,5), mais os equivalentes em dias de calendário para comparar
# de frente com a parametrização em meses, mais os híbridos.
# O acumulativo NÃO entra: medido em 19/08/2026, uma janela de 6.000 h reproduz
# o acumulativo dígito por dígito (5/8, lead 17,9 h, FP 0,77/mês, 21,2 h/mês) —
# ele é o nível superior da varredura, não um braço separado.
POLICIES_HORAS = [
    BaselinePolicy(window_hours=h) for h in (400, 600, 900, 1400, 2000, 3000, 4000)
]
POLICIES_DIAS = [BaselinePolicy(window_days=d) for d in (30, 60, 90)]
# Os pares NÃO são produto cartesiano: medido em 21/08/2026, 90 dias de
# calendário rendem ~1.500 h de operação elegível nesta série, então
# (3000 h, 90 d) deixa a restrição em horas inerte — o teto de idade decide
# sozinho e o braço vira um "90 dias" disfarçado (truncado em 15/15 retreinos).
# Cada par abaixo tem as duas restrições plausivelmente ativas.
POLICIES_HIBRIDAS = [
    BaselinePolicy(window_hours=900, max_age_days=45),
    BaselinePolicy(window_hours=1400, max_age_days=90),
    BaselinePolicy(window_hours=2000, max_age_days=120),
    BaselinePolicy(window_hours=3000, max_age_days=180),
]
POLICY_CONTROLE = BaselinePolicy()          # acumulativo, braço de controle

MODES = {
    "quick": dict(models=["pca"], grids=["2min"], policies=[BaselinePolicy(window_hours=300), POLICY_CONTROLE], ewma=["1h"],
                  thresholds=[99.0, 99.9], sustain=["30min"], confirm=[1, 2],
                  exclude_days=[0, 7], exclude_alarm_h=[0.0], min_alert=["0min"]),
    "balanced": dict(models=["pca", "iforest"], grids=["2min"], policies=[BaselinePolicy(window_hours=300), BaselinePolicy(window_hours=400),
                                POLICY_CONTROLE],
                     ewma=["30min", "1h"], thresholds=[99.0, 99.5, 99.9, 99.97],
                     sustain=["30min", "1h"], confirm=[2, 3], exclude_days=[0, 7],
                     exclude_alarm_h=[0.0, 1.0], min_alert=["0min", "1h"]),
    # Região de baixo falso positivo: limiar e persistência agressivos, exigindo
    # 2 ou 3 sinais simultâneos. É onde o teto de 1 episódio/mês é alcançável.
    "fp_first": dict(models=["pca", "iforest"], grids=["2min"], policies=[BaselinePolicy(window_hours=300), BaselinePolicy(window_hours=400),
                                POLICY_CONTROLE],
                     ewma=["1h", "2h"], thresholds=[99.9, 99.97, 99.99, 99.995],
                     sustain=["1h", "2h", "4h"], confirm=[2, 3], exclude_days=[0, 7],
                     exclude_alarm_h=[0.0, 1.0], min_alert=["30min", "1h"]),
    "full": dict(models=["pca", "iforest", "ae"], grids=["30s", "2min"],
                 policies=[BaselinePolicy(window_hours=300), BaselinePolicy(window_hours=400),
                           POLICY_CONTROLE], ewma=["30min", "1h", "2h"],
                 thresholds=[99.0, 99.5, 99.9, 99.97, 99.99, 99.995],
                 sustain=["30min", "1h", "2h", "4h"], confirm=[2, 3, 4],
                 exclude_days=[0, 7], exclude_alarm_h=[0.0, 1.0],
                 min_alert=["0min", "30min"]),
    # ESTÁGIO 1 — varredura de POLÍTICA de baseline. Fatores baratos fixos em
    # poucos centros: o objetivo é ordenar as políticas, não achar o ponto final.
    # Grade de 2 min só: o ajuste do autoencoder em 30 s com janela de 4.000 h
    # custa 61 s (contra 4,6 s em 2 min), o que sozinho inviabilizaria a
    # varredura. A grade de 30 s volta no estágio 2, em torno da vencedora.
    # IsolationForest fica fora: nunca passa de 2/8 e cai a 1/8 exigindo lead
    # >= 12 h (30 configs de 31.104).
    "policy_sweep": dict(
        models=["pca", "ae"], grids=["2min"],
        policies=POLICIES_HORAS + POLICIES_DIAS + POLICIES_HIBRIDAS + [POLICY_CONTROLE],
        ewma=["30min", "1h", "2h"],
        thresholds=[99.5, 99.9, 99.97], threshold_kinds=["percentil", "k_maiores"],
        thresholds_k=[10, 30, 100],
        sustain=["30min", "2h"], confirm=[2, 3],
        exclude_days=[0, 7], exclude_alarm_h=[1.0], min_alert=["0min"]),
    # ESTÁGIO 1b — varredura de ARQUITETURA. Cruza as 9 arquiteturas com três
    # políticas de baseline, e não com uma só: a arquitetura vencedora depende do
    # tamanho da janela (medido: com a janela do notebook 06 o PCA faz 5/8 e o
    # autoencoder 6/8; em 3.000 h a ordem se inverte). Cravar uma política aqui
    # escolheria a arquitetura sob uma condição arbitrária.
    # As 6 arquiteturas novas custam pouco — Mahalanobis ajusta em 0,06 s contra
    # 2,1 s do autoencoder —, então o custo continua dominado pelos 3 AE.
    "arch_sweep": dict(
        models=list(FamilyScorer.ARQUITETURAS), grids=["2min"],
        policies=[BaselinePolicy(window_hours=1400),
                  BaselinePolicy(window_hours=3000), POLICY_CONTROLE],
        ewma=["30min", "1h", "2h"],
        thresholds=[99.5, 99.9, 99.97], threshold_kinds=["percentil", "k_maiores"],
        thresholds_k=[10, 30, 100],
        sustain=["30min", "2h"], confirm=[2, 3],
        exclude_days=[0, 7], exclude_alarm_h=[1.0], min_alert=["0min"]),
    # ESTÁGIO 2 — refino em torno das políticas vencedoras do estágio 1. As
    # políticas são passadas por --policy-hours; o default é o preview medido.
    "policy_fine": dict(
        models=["pca", "mahal", "ae"], grids=["30s", "2min"],
        policies=[BaselinePolicy(window_hours=2000), BaselinePolicy(window_hours=3000),
                  BaselinePolicy(window_hours=3000, max_age_days=90)],
        ewma=["30min", "1h", "2h"],
        thresholds=[99.0, 99.5, 99.9, 99.97, 99.99, 99.995],
        sustain=["30min", "1h", "2h", "4h"], confirm=[2, 3, 4],
        exclude_days=[0, 7], exclude_alarm_h=[0.0, 1.0],
        min_alert=["0min", "30min"]),
}


def build_trials(mode: str, per_family: bool = False, n_signals: int = 4,
                 single_model: bool = False) -> list[Trial]:
    """Produto cartesiano do modo, ordenado para aproveitar cache.

    A ordem das chaves não é estética: os fatores CAROS (modelo, grade,
    política, exclusões) vêm primeiro e o ewma depois, para que os trials que
    compartilham o mesmo ajuste de modelo e a mesma suavização apareçam
    juntos — o cache de suavização guarda só 4 chaves e depende dessa
    localidade. Medido: cada ajuste é reaproveitado por centenas de trials.
    """
    space = dict(MODES[mode])
    if single_model:
        space["confirm"] = [1]          # com um sinal só, não existe confirmação

    # (tipo de limiar, valor) — os dois eixos são alternativos, não cruzados
    limiares: list[tuple[str, float | tuple[float, ...]]] = []
    for kind in space.get("threshold_kinds", ["percentil"]):
        niveis = space["thresholds"] if kind == "percentil" else space.get("thresholds_k", [])
        for nivel in niveis:
            if per_family:
                limiares += [(kind, tuple(c))
                             for c in itertools.product(niveis, repeat=n_signals)]
                break
            limiares.append((kind, float(nivel)))

    keys = ["models", "grids", "policies", "exclude_days", "exclude_alarm_h",
            "ewma", "sustain", "confirm", "min_alert"]
    combos = itertools.product(*(space[k] for k in keys), limiares)
    return [Trial(model=mo, grid=gr, baseline=po, exclude_days=ex,
                  exclude_alarm_h=ea, ewma=ew, sustain=su, confirm=cf,
                  min_alert=ma, threshold=thr, threshold_kind=kind)
            for mo, gr, po, ex, ea, ew, su, cf, ma, (kind, thr) in combos]


def select_best(results: list[TrialResult]) -> TrialResult | None:
    """Maximiza detecção sujeito ao teto de FP **e** ao teste de liveness.

    O ``vivo`` é o que mudou em 19/08/2026. Sem ele, a busca premiava o detector
    que emudece: o campeão anterior detectava os 5 eventos da 1a metade da série
    e ficava 3.186 h de operação sem emitir nada — trecho que contém as 3 falhas
    restantes. Com FP baixo, porque quem não alarma não erra.

    Desempate deliberado por ``det_2a_metade`` antes do lead: entre duas configs
    com a mesma detecção total, a que distribui no tempo é a defensável.
    """
    elegiveis = [r for r in results if r.aprovado and r.vivo]
    if not elegiveis:                       # sem ninguém vivo, relaxa o liveness
        elegiveis = [r for r in results if r.aprovado]
    pool = elegiveis or results
    return max(pool, key=lambda r: (r.eventos_detectados, r.det_2a_metade,
                                    r.lead_medio_h or 0.0, -r.fp_por_mes))


def main() -> None:
    global EXHAUST_GATE, EXHAUST_ON_C, EXHAUST_MIN_H
    global STARTUP_EXCLUDE, POST_START_SILENCE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(MODES), default="balanced")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--eval-start", default="2025-02")
    parser.add_argument("--eval-end", default="2026-04")
    parser.add_argument("--max-fp-per-month", type=float, default=1.0)
    parser.add_argument("--with-vibration", action="store_true",
                        help="tira a vibração da quarentena (não recomendado)")
    parser.add_argument("--policy-hours", type=float, nargs="*", default=None,
                        help="sobrescreve as políticas do modo por janelas em "
                             "horas de operação elegível (ex.: --policy-hours 2000 3000)")
    parser.add_argument("--policy-max-age-days", type=float, default=None,
                        help="teto de idade aplicado às janelas de --policy-hours")
    parser.add_argument("--single-model", action="store_true",
                        help="variante de controle: UM multivariado sobre todos os "
                             "sensores, um único score, sem os sinais derivados "
                             "(mantém o re-treino mensal e todo o resto do filtro)")
    parser.add_argument("--per-family-thresholds", action="store_true",
                        help="busca um limiar independente por família de sinal; "
                             "abre a fronteira detecção×FP ao custo de uma grade maior")
    parser.add_argument("--remote", action="store_true", help="enfileira no ClearML e sai")
    parser.add_argument("--queue", default="default")
    # A fila 'default' roda em Docker com a imagem CUDA (Python 3.8), onde
    # pandas>=2.2 não existe. Este projeto não usa GPU, então pedimos uma imagem
    # com Python moderno. Trocar aqui se a política de imagens da equipe mudar.
    parser.add_argument("--docker-image", default="python:3.12",
                        help="imagem Docker para a execução remota")
    parser.add_argument("--local", action="store_true", help="não usa ClearML para nada")
    parser.add_argument("--local-csv", default=None, help="CSV local em vez do dataset")
    parser.add_argument("--limit", type=int, default=None, help="corta a grade (debug)")
    parser.add_argument("--exhaust-gate", choices=("off", "stable", "operation"),
                        default=EXHAUST_GATE,
                        help="onde o marcador de exaustão (T5) age na operabilidade "
                             f"(padrão: {EXHAUST_GATE})")
    parser.add_argument("--exhaust-on-c", type=float, default=EXHAUST_ON_C,
                        help=f"limiar de 'em carga' em °C (padrão: {EXHAUST_ON_C:g})")
    parser.add_argument("--task-suffix", default="",
                        help="sufixo no nome da task do ClearML, para distinguir "
                             "runs que só diferem por flag (ex.: t5gate)")
    parser.add_argument("--startup-exclude", default=STARTUP_EXCLUDE,
                        help="descarte de partida no BASELINE e na pontuação "
                             f"(eixo b; padrão: {STARTUP_EXCLUDE})")
    parser.add_argument("--post-start-silence", default=POST_START_SILENCE,
                        help="silêncio do ALERTA após cada partida, sem tocar no "
                             f"baseline (eixo a; padrão: {POST_START_SILENCE})")
    parser.add_argument("--exhaust-min-h", type=float, default=EXHAUST_MIN_H,
                        help="duração mínima do trecho frio para desqualificar "
                             f"(0 desliga o piso; padrão: {EXHAUST_MIN_H:g} h)")
    args = parser.parse_args()
    if not args.local:
        print(f"[pedido] gate={args.exhaust_gate} | descarte={args.startup_exclude} | "
              f"silêncio={args.post_start_silence}", flush=True)

    task = None
    if not args.local:
        from clearml import Task
        sufixo = "-modelo-unico" if args.single_model else ""
        if args.task_suffix:
            sufixo += f"-{args.task_suffix}"
        task = Task.init(project_name="Cabiunas",
                         task_name=f"automl-fp-first-{args.mode}{sufixo}",
                         task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)
        task.connect(vars(args))
        # Pins deliberadamente frouxos: o script roda em pandas 2.0+ e não depende
        # de nada exclusivo das versões mais novas. Assim a task sobrevive a um
        # worker com Python/imagem diferente do esperado.
        task.set_packages(["pandas>=2.0", "numpy>=1.23", "pyarrow>=10",
                           "scikit-learn>=1.1", "clearml"])
        if args.remote:
            task.set_base_docker(args.docker_image)
            print(f"enfileirando na fila '{args.queue}' "
                  f"(imagem {args.docker_image})...")
            task.execute_remotely(queue_name=args.queue, exit_process=True)

    # DEPOIS do connect, nunca antes. No worker o argparse devolve os DEFAULTS — quem
    # injeta o que foi pedido é o `task.connect(vars(args))`, que reescreve o __dict__
    # do Namespace com os valores gravados na task. Ler antes disso fez as três
    # variantes de 26/08/2026 rodarem como baseline: a task guardava
    # `startup_exclude=24h` e o processo usava 2h, e as quatro deram resultado idêntico.
    EXHAUST_GATE, EXHAUST_ON_C = args.exhaust_gate, args.exhaust_on_c
    EXHAUST_MIN_H = args.exhaust_min_h
    STARTUP_EXCLUDE, POST_START_SILENCE = args.startup_exclude, args.post_start_silence
    print(f"[partida] descarte no baseline {STARTUP_EXCLUDE} | "
          f"silêncio do alerta {POST_START_SILENCE}", flush=True)
    print(f"[operabilidade] RUNNING_A >= {RUNNING_THRESHOLD}"
          + (f" + {EXHAUST_TAG} >= {EXHAUST_ON_C:g} °C por >= {EXHAUST_MIN_H:g} h "
             f"em '{EXHAUST_GATE}'"
             if EXHAUST_GATE != "off" else " (marcador de exaustão desligado)"), flush=True)

    n_signals = 1 if args.single_model else \
        len(FAMILIES) + (1 if args.with_vibration else 0) + 2   # + spread + selagem
    if args.policy_hours:
        MODES[args.mode]["policies"] = [
            BaselinePolicy(window_hours=h, max_age_days=args.policy_max_age_days)
            for h in args.policy_hours]
    trials = build_trials(args.mode, args.per_family_thresholds, n_signals,
                          args.single_model)
    if args.limit:
        trials = trials[:args.limit]
    print(f"[plano] modo={args.mode} | {len(trials)} configurações | "
          f"avaliação {args.eval_start}→{args.eval_end} | "
          f"teto FP={args.max_fp_per_month}/mês", flush=True)

    bundle = DataBundle(args.dataset_id, args.local_csv, args.with_vibration,
                        args.single_model).load()
    if args.single_model:
        print(f"[modo] MODELO ÚNICO: 1 multivariado sobre "
              f"{len(bundle.families['tudo'])} sensores, 1 score, sem sinais derivados",
              flush=True)
    evaluator = WalkForwardEvaluator(bundle, args.eval_start, args.eval_end,
                                     args.max_fp_per_month)
    print(f"[eventos] {len(evaluator.events)} eventos físicos (varredura): "
          + ", ".join(f"{e['inicio']:%d/%m/%Y}({e['mecanismo']})" for e in evaluator.events),
          flush=True)

    results: list[TrialResult] = []
    started = time.time()
    for i, trial in enumerate(trials, 1):
        try:
            result = evaluator.evaluate(trial)
        except Exception as exc:  # noqa: BLE001 — um trial ruim não derruba a busca
            print(f"  [{i}/{len(trials)}] {trial.label()} -> ERRO {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        results.append(result)
        flag = "OK " if result.aprovado else "FP+"
        print(f"  [{i}/{len(trials)}] {flag} {trial.label()} -> "
              f"detectou {result.eventos_detectados}/{result.eventos_total} | "
              f"lead {result.lead_medio_h} h | FP {result.fp_por_mes}/mês", flush=True)
        if task:
            logger = task.get_logger()
            logger.report_scalar("deteccao", "eventos", result.eventos_detectados, i)
            logger.report_scalar("falso_positivo", "por_mes", result.fp_por_mes, i)
            if result.lead_medio_h:
                logger.report_scalar("lead", "horas", result.lead_medio_h, i)

    table = pd.json_normalize([asdict(r) for r in results])
    best = select_best(results)
    elapsed = (time.time() - started) / 60
    print(f"\n[fim] {len(results)} trials em {elapsed:.1f} min | "
          f"aprovados (FP<=teto): {sum(r.aprovado for r in results)}")
    if best:
        print("\n=== MELHOR CONFIGURAÇÃO ===")
        print(json.dumps({"trial": best.trial, "eventos": f"{best.eventos_detectados}/"
                          f"{best.eventos_total}", "lead_medio_h": best.lead_medio_h,
                          "leads": best.leads, "fp_por_mes": best.fp_por_mes,
                          "aprovado": best.aprovado}, indent=2, ensure_ascii=False))

    out_dir = Path("automl_out")
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "automl_results.csv", index=False)
    if best:
        (out_dir / "best_trial.json").write_text(
            json.dumps(asdict(best), indent=2, ensure_ascii=False, default=str))
    if task:
        task.upload_artifact("automl_results", table)
        if best:
            task.upload_artifact("best_trial", asdict(best))
        # Só as melhores linhas vão para o gráfico: a grade completa já está no
        # artefato acima. O report_table manda a tabela como UM evento, e em
        # 51.840 linhas isso dá 16,8 MB contra o limite de 15 MB do servidor —
        # o batch não é divisível, então cada retry remonta o mesmo payload e a
        # task fica presa em retry infinito sem nunca fechar (visto em 22/08/2026).
        ordem = ["eventos_detectados", "det_2a_metade", "lead_medio_h"]
        resumo = table.sort_values(ordem, ascending=False).head(TABLE_PLOT_ROWS)
        task.get_logger().report_table(
            f"resultados (top {len(resumo)} de {len(table)})", "grade",
            table_plot=resumo)
    print(f"\nresultados em {out_dir.resolve()}")


if __name__ == "__main__":
    main()
