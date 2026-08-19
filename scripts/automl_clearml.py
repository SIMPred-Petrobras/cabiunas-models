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
        if self.kind == "pca":
            self.model.fit(scaled)
        elif self.kind == "ae":
            self.model.fit(scaled, scaled)
        else:
            self.model.fit(scaled)
        # Guarda o score do próprio baseline: o limiar é um QUANTIL desta
        # distribuição (após a mesma suavização aplicada ao teste), o que torna
        # a busca válida para qualquer modelo — inclusive os de score limitado,
        # como IsolationForest, em que múltiplos de p99 são inalcançáveis.
        self.baseline_score = pd.Series(self._raw(scaled), index=data.index)
        return self

    def _build(self, n_features: int):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.neural_network import MLPRegressor
        if self.kind == "pca":
            return PCA(n_components=0.95, svd_solver="full")
        if self.kind == "iforest":
            return IsolationForest(n_estimators=200, contamination="auto",
                                   random_state=self.seed, n_jobs=-1)
        if self.kind == "ae":
            hidden = (max(8, n_features // 2), max(4, n_features // 4),
                      max(8, n_features // 2))
            return MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                early_stopping=True, n_iter_no_change=5,
                                max_iter=60, random_state=self.seed)
        raise ValueError(f"modelo desconhecido: {self.kind}")

    def _raw(self, scaled: np.ndarray) -> np.ndarray:
        if self.kind == "pca":
            back = self.model.inverse_transform(self.model.transform(scaled))
            return np.mean((scaled - back) ** 2, axis=1)
        if self.kind == "ae":
            return np.mean((self.model.predict(scaled) - scaled) ** 2, axis=1)
        return -self.model.score_samples(scaled)   # maior = mais anômalo

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
        starts = in_operation & ~in_operation.shift(fill_value=False)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        window = max(int(pd.Timedelta(STARTUP_EXCLUDE).total_seconds() / step), 1)
        stable = in_operation & ~starts.rolling(window, min_periods=1).max().astype(bool)
        oper = pd.DataFrame({"in_operation": in_operation, "stable": stable}, index=frame.index)
        self._grids[freq] = (frame, oper)
        return self._grids[freq]


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
@dataclass(frozen=True)
class Trial:
    model: str
    grid: str
    # HORAS de operação estável no baseline, não dias de calendário. Medido: com
    # 28 dias fixos o baseline varia de 160 h a 672 h entre meses, porque a
    # máquina para muito — e o limiar é um percentil desse baseline, então sua
    # estabilidade variava junto. Com horas fixas, todo mês treina com o mesmo
    # tamanho e os limiares mensais ficam comparáveis.
    baseline_hours: int
    # "movel" = só as últimas `baseline_hours` (esquece o passado remoto)
    # "acumulativo" = tudo que já foi elegível desde o início da série
    # CUIDADO deliberado: o acumulativo carrega regimes antigos para sempre, e esta
    # máquina mudou depois da parada de 23 dias de abr/2025. Por isso o modo entra
    # como dimensão de busca em vez de virar o padrão.
    baseline_mode: str
    ewma: str
    # dias antes de cada evento conhecido removidos do baseline (0 = sem limpeza)
    exclude_days: int
    # horas em torno de CADA ativação de alarme removidas do baseline (0 = nenhuma).
    # Custo medido: ~5% do baseline com 1 h.
    exclude_alarm_h: float
    # PERCENTIL do baseline suavizado (99 = p99). Um único valor aplica-se a
    # todos os sinais; uma tupla define um limiar por sinal, na ordem de
    # SIGNAL_ORDER — isso abre a fronteira, porque as famílias têm estabilidade
    # muito diferente (a de temperatura desloca entre meses, a de pressão não).
    threshold: float | tuple[float, ...]
    sustain: str
    confirm: int          # nº de sinais simultâneos exigidos (1=atenção, 2=confirmado)
    # tempo mínimo que o alerta CONFIRMADO precisa ficar de pé para ser emitido.
    # Sem isso, a detecção de 17/03/2025 durava 9 minutos — antecedência de 29 h
    # no papel, invisível em um plantão.
    min_alert: str = "0min"

    def label(self) -> str:
        thr = (self.threshold if isinstance(self.threshold, float)
               else "/".join(str(t) for t in self.threshold))
        modo = "acum" if self.baseline_mode == "acumulativo" else f"{self.baseline_hours}h"
        return (f"{self.model}|{self.grid}|b{modo}|excl{self.exclude_days}d"
                f"|alm{self.exclude_alarm_h}h|ewma{self.ewma}|q{thr}"
                f"|sust{self.sustain}|conf{self.confirm}|min{self.min_alert}")

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
    aprovado: bool = False


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
    def raw_scores(self, model: str, grid: str, baseline_hours: int,
                   exclude_days: int = 0, exclude_alarm_h: float = 0.0,
                   baseline_mode: str = "movel") -> dict:
        """Walk-forward: para cada mês, devolve o score do baseline e do teste.

        Guardar os dois permite que o limiar seja um quantil do baseline **já
        suavizado** com o mesmo EWMA do teste — sem essa correspondência, o
        limiar não controla o falso positivo de verdade.

        O baseline é montado por **horas de operação estável elegível**, recuando
        no calendário o quanto for preciso, e não por um número fixo de dias.
        """
        key = (model, grid, baseline_hours, exclude_days, exclude_alarm_h, baseline_mode)
        if key in self._cache:
            return self._cache[key]
        frame, oper = self.bundle.grid(grid)
        stable = oper["stable"]
        veto = self.bundle.frozen_veto(grid)      # sensor congelado não é anomalia
        veto_qualquer = veto.any(axis=1)          # para os sinais derivados
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        n_baseline = max(int(baseline_hours * 3600 / step), 1000)
        # elegibilidade do baseline calculada UMA vez para todo o período
        elegivel = stable & self._baseline_mask(frame.index, exclude_days, exclude_alarm_h)
        indices_elegiveis = elegivel[elegivel].index

        derivados = (("mancal_spread", self._spread), ("selagem_z", self._seal)) \
            if getattr(self.bundle, "derived_signals", True) else ()
        names = list(self.bundle.families) + [n for n, _ in derivados]
        parts: dict[str, list[tuple[pd.Series, pd.Series]]] = {n: [] for n in names}

        for month in self.months:
            start = pd.Timestamp(month + "-01")
            end = start + pd.offsets.MonthBegin(1)
            anteriores = indices_elegiveis[indices_elegiveis <= start]
            if len(anteriores) < 1000:
                continue
            # móvel = só as últimas N horas; acumulativo = tudo desde o início
            escolhidos = (anteriores if baseline_mode == "acumulativo"
                          else anteriores[-n_baseline:])
            base_stable = frame.loc[escolhidos]
            test = frame.loc[start:end - pd.Timedelta(seconds=1)]
            if test.empty:
                continue
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
    def evaluate(self, trial: Trial) -> TrialResult:
        parts = self.raw_scores(trial.model, trial.grid, trial.baseline_hours,
                                trial.exclude_days, trial.exclude_alarm_h,
                                trial.baseline_mode)
        frame, oper = self.bundle.grid(trial.grid)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        n_sustain = max(int(pd.Timedelta(trial.sustain).total_seconds() / step), 1)
        halflife = pd.Timedelta(trial.ewma)

        def smooth(series: pd.Series) -> pd.Series:
            return series.ewm(halflife=halflife, times=series.index).mean()

        order = [n for n, months in parts.items() if months]
        active = []
        for name in order:
            pct = trial.threshold_for(name, order)
            flags = []
            for baseline, test in parts[name]:
                # limiar = quantil do baseline suavizado (mesmo EWMA do teste)
                limit = float(np.nanpercentile(smooth(baseline).to_numpy(), pct))
                hits = (smooth(test) > limit).astype(int)
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

        episodes = self._episodes(alert)
        stops = self._stops(oper)
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

        days = float(oper["stable"].reindex(alert.index, fill_value=False).sum()) * step / 86400
        fp_month = len(false_positives) / max(days, 1) * 30
        # horas de alerta falso: soma a duração dos episódios contados como FP
        falsos = set(false_positives)
        horas_fp = 0.0
        for inicio, fim in self._spans(alert):
            if inicio in falsos:
                horas_fp += (fim - inicio).total_seconds() / 3600
        fp_horas_mes = horas_fp / max(days, 1) * 30
        result = TrialResult(
            trial=asdict(trial), eventos_detectados=int(len(leads)),
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
        )
        result.aprovado = bool(result.fp_por_mes <= self.max_fp_per_month)
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
MODES = {
    "quick": dict(models=["pca"], grids=["2min"], baseline_hours=[300],
                  baseline_mode=["movel", "acumulativo"], ewma=["1h"],
                  thresholds=[99.0, 99.9], sustain=["30min"], confirm=[1, 2],
                  exclude_days=[0, 7], exclude_alarm_h=[0.0], min_alert=["0min"]),
    "balanced": dict(models=["pca", "iforest"], grids=["2min"], baseline_hours=[300, 400],
                     baseline_mode=["movel", "acumulativo"],
                     ewma=["30min", "1h"], thresholds=[99.0, 99.5, 99.9, 99.97],
                     sustain=["30min", "1h"], confirm=[2, 3], exclude_days=[0, 7],
                     exclude_alarm_h=[0.0, 1.0], min_alert=["0min", "1h"]),
    # Região de baixo falso positivo: limiar e persistência agressivos, exigindo
    # 2 ou 3 sinais simultâneos. É onde o teto de 1 episódio/mês é alcançável.
    "fp_first": dict(models=["pca", "iforest"], grids=["2min"], baseline_hours=[300, 400],
                     baseline_mode=["movel", "acumulativo"],
                     ewma=["1h", "2h"], thresholds=[99.9, 99.97, 99.99, 99.995],
                     sustain=["1h", "2h", "4h"], confirm=[2, 3], exclude_days=[0, 7],
                     exclude_alarm_h=[0.0, 1.0], min_alert=["30min", "1h"]),
    "full": dict(models=["pca", "iforest", "ae"], grids=["30s", "2min"],
                 baseline_hours=[300, 400], baseline_mode=["movel", "acumulativo"], ewma=["30min", "1h", "2h"],
                 thresholds=[99.0, 99.5, 99.9, 99.97, 99.99, 99.995],
                 sustain=["30min", "1h", "2h", "4h"], confirm=[2, 3, 4],
                 exclude_days=[0, 7], exclude_alarm_h=[0.0, 1.0],
                 min_alert=["0min", "30min"]),
}


def build_trials(mode: str, per_family: bool = False, n_signals: int = 4,
                 single_model: bool = False) -> list[Trial]:
    space = dict(MODES[mode])
    if single_model:
        space["confirm"] = [1]          # com um sinal só, não existe confirmação
    thresholds: list = list(space["thresholds"])
    if per_family:
        # combinações independentes por sinal (produto cartesiano dos percentis)
        thresholds = [tuple(c) for c in itertools.product(thresholds, repeat=n_signals)]
    keys = ["models", "grids", "baseline_hours", "baseline_mode", "ewma", "sustain",
            "confirm", "exclude_days", "exclude_alarm_h", "min_alert"]
    combos = itertools.product(*(space[k] for k in keys), thresholds)
    trials = [Trial(model=mo, grid=gr, baseline_hours=bh, baseline_mode=bm, ewma=ew,
                    sustain=su, confirm=cf, exclude_days=ex, exclude_alarm_h=ea,
                    min_alert=ma, threshold=th)
              for mo, gr, bh, bm, ew, su, cf, ex, ea, ma, th in combos]
    # no acumulativo o tamanho da janela não existe: evita duplicar a mesma busca
    return [t for t in trials
            if t.baseline_mode == "movel" or t.baseline_hours == min(space["baseline_hours"])]


def select_best(results: list[TrialResult]) -> TrialResult | None:
    """Maximiza detecção, depois lead, entre os que respeitam o teto de FP."""
    approved = [r for r in results if r.aprovado]
    pool = approved or results
    return max(pool, key=lambda r: (r.eventos_detectados, r.lead_medio_h or 0.0,
                                    -r.fp_por_mes))


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(MODES), default="balanced")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--eval-start", default="2025-02")
    parser.add_argument("--eval-end", default="2026-04")
    parser.add_argument("--max-fp-per-month", type=float, default=1.0)
    parser.add_argument("--with-vibration", action="store_true",
                        help="tira a vibração da quarentena (não recomendado)")
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
    args = parser.parse_args()

    task = None
    if not args.local:
        from clearml import Task
        sufixo = "-modelo-unico" if args.single_model else ""
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

    n_signals = 1 if args.single_model else \
        len(FAMILIES) + (1 if args.with_vibration else 0) + 2   # + spread + selagem
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
        task.get_logger().report_table("resultados", "grade", table_plot=table)
    print(f"\nresultados em {out_dir.resolve()}")


if __name__ == "__main__":
    main()
