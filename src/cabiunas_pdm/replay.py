"""Reexecução do detector campeão para inspeção visual.

O AutoML (``scripts/automl_clearml.py``) devolve **números**: quantos eventos
foram antecipados, com quanto de antecedência, a que custo de falso positivo.
Este módulo devolve as **séries** — score suavizado, limiar daquele mês, alerta
sustentado, episódios — para que a decisão do modelo possa ser *vista* sobre a
série temporal, ao lado das falhas reais.

As classes de cálculo não são reimplementadas aqui: o script de AutoML é
importado por caminho (ele é autocontido de propósito, para rodar nos workers do
ClearML) e reaproveitado, de modo que o que aparece no gráfico é o mesmo cálculo
que produziu a fronteira detecção×falso positivo.

Uso típico::

    from cabiunas_pdm.replay import DetectionReplay

    replay = DetectionReplay()
    resultado = replay.run(model="ae", grid="30s", baseline_hours=300,
                           exclude_days=7, ewma="30min", threshold=99.99,
                           sustain="2h", confirm=3)
    resultado.events_table()
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from . import config

AUTOML_SCRIPT = config.PROJECT / "scripts" / "automl_clearml.py"

# Sensores que contam a história de cada mecanismo de falha. Usados como padrão
# nas figuras por evento (o modelo vê a família inteira; o gráfico mostra os
# sensores em que o mecanismo aparece).
SENSORS_BY_MECHANISM: dict[str, list[str]] = {
    "mancal": ["954005_624_TI_0305", "954005_624_TI_0303",
               "954005_624_TI_0301", "954005_624_TI_0307"],
    "oleo": ["954005_624_PI_0315", "954005_624_PI_0319",
             "954005_624_PDI_0317", "954005_624_TI_0325"],
    "selagem": ["954005_624_PDIT_0305", "954005_624_PDI_0302",
                "954005_624_PI_0307", "954005_624_PI_0308"],
}

# família de sinais em que cada mecanismo aparece (para abrir o erro por sensor)
MECHANISM_FAMILY: dict[str, str] = {
    "mancal": "temperatura", "oleo": "pressao_oleo", "selagem": "pressao_oleo",
}

# Um sensor por assunto, para a visão panorâmica da série: o mancal que mais
# falha, a temperatura de exaustão da turbina, a pressão de óleo, a selagem e a
# vibração (esta em quarentena como alarme, mas útil de ver).
SENSORS_OVERVIEW: list[str] = [
    "954005_624_TI_0305", "T5_AVG_A", "954005_624_PI_0315",
    "954005_624_PDIT_0305", "TV_351X_A",
]


def load_automl(path: Path = AUTOML_SCRIPT) -> ModuleType:
    """Importa ``scripts/automl_clearml.py`` como módulo (fonte única do cálculo)."""
    if "cabiunas_automl" in sys.modules:
        return sys.modules["cabiunas_automl"]
    spec = importlib.util.spec_from_file_location("cabiunas_automl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cabiunas_automl"] = module
    spec.loader.exec_module(module)
    return module


def clearml_cached_csv(dataset_id: str) -> Path | None:
    """Caminho do CSV já baixado pelo ClearML, se existir (permite rodar offline)."""
    automl = load_automl()
    candidate = (Path.home() / ".clearml" / "cache" / "storage_manager" / "datasets"
                 / f"ds_{dataset_id}" / automl.SENSORS_FILE)
    return candidate if candidate.exists() else None


def duracao_da_parada(operabilidade: pd.DataFrame, quando, tolerancia: str = "2h") -> dict:
    """Mede a parada que se segue a um instante (o trip), se houver.

    Cuidado que motivou a função: na grade de 30 s a amostra do próprio instante do
    trip ainda costuma marcar "operando" — a máquina cai um ou dois passos depois.
    Medir ``primeiro instante operando após o trip`` devolve zero nesses casos.
    Aqui procura-se primeiro o **início da parada** (dentro de ``tolerancia``) e só
    então o retorno à operação.
    """
    operando = operabilidade["in_operation"].astype(bool)
    t = pd.Timestamp(quando)
    janela = operando.loc[t:t + pd.Timedelta(tolerancia)]
    parados = janela[~janela]
    if parados.empty:
        return {"parou": False, "inicio": None, "fim": None, "horas": 0.0}
    inicio = parados.index[0]
    depois = operando.loc[inicio:]
    voltou = depois[depois]
    fim = voltou.index[0] if len(voltou) else depois.index[-1]
    return {"parou": True, "inicio": inicio, "fim": fim,
            "horas": round((fim - inicio).total_seconds() / 3600, 1)}


def load_alarms(dataset_id: str | None = None) -> pd.DataFrame:
    """Alarmes do mesmo dataset usado pelo AutoML, lidos do cache local.

    É a planilha de alarmes da operação: serve para comparar, no gráfico, o que o
    modelo marcou com o que a instrumentação já registrava. ``nivel_trip`` marca
    os alarmes de nível (os que derrubam a máquina).
    """
    automl = load_automl()
    dataset_id = dataset_id or automl.DATASET_ID
    pasta = (Path.home() / ".clearml" / "cache" / "storage_manager" / "datasets"
             / f"ds_{dataset_id}")
    arquivos = list(pasta.rglob(automl.ALARMS_FILE)) or list(pasta.rglob("*alarmes*.csv"))
    if not arquivos:
        from clearml import Dataset
        raiz = Path(Dataset.get(dataset_id=dataset_id).get_local_copy())
        arquivos = list(raiz.rglob("*alarmes*.csv"))
    alarmes = pd.read_csv(arquivos[0])
    quando = alarmes.columns[0]
    alarmes[quando] = pd.to_datetime(alarmes[quando], errors="coerce")
    alarmes = alarmes.dropna(subset=[quando]).rename(columns={quando: "quando"})
    alarmes["ativado"] = alarmes["Status"].astype(str).str.startswith("ACT")
    # mesmo critério do AutoML: validado contra a tag real (TAHH_/PALL_/PDAHH)
    # nas 6.851 linhas da planilha, com 100% de concordância
    alarmes["nivel_trip"] = (alarmes["Descrição Alarme"].astype(str).str.upper()
                             .str.contains(automl.ALARM_LEVEL_PATTERN, regex=True))
    return alarmes.set_index("quando").sort_index()


class CachedBundle:
    """``DataBundle`` do AutoML com cache parquet da série limpa.

    Ler o CSV de 594 MB e converter 40 colunas de texto para número leva minutos
    e o resultado é determinístico — então guarda-se o resultado. O cache é
    invalidado pelo nome (dataset + com/sem vibração).
    """

    def __init__(self, dataset_id: str | None = None, with_vibration: bool = False,
                 cache_dir: Path | None = None, single_model: bool = False) -> None:
        automl = load_automl()
        self.automl = automl
        self.dataset_id = dataset_id or automl.DATASET_ID
        self.with_vibration = with_vibration
        self.single_model = single_model
        self.cache_dir = cache_dir or config.INTERIM
        self.csv_path = clearml_cached_csv(self.dataset_id)

    @property
    def cache_file(self) -> Path:
        sufixo = "_vib" if self.with_vibration else ""
        return self.cache_dir / f"replay_raw_{self.dataset_id[:8]}{sufixo}.parquet"

    def build(self):
        """Devolve um ``DataBundle`` carregado, usando cache quando possível."""
        bundle = self.automl.DataBundle(self.dataset_id, self.csv_path,
                                        self.with_vibration, self.single_model)
        cache = self.cache_file
        if cache.exists():
            bundle.raw = pd.read_parquet(cache)
            print(f"[dados] cache {cache.name}: {len(bundle.raw):,} amostras | "
                  f"{bundle.raw.index.min()} → {bundle.raw.index.max()}")
        else:
            bundle.load()
            cache.parent.mkdir(parents=True, exist_ok=True)
            bundle.raw.to_parquet(cache)
            print(f"[dados] cache gravado em {cache}")
        return bundle


# --------------------------------------------------------------------- saída
@dataclass
class ReplayResult:
    """Tudo que é preciso para desenhar a decisão do detector."""

    trial: dict
    scores: pd.DataFrame        # score suavizado por sinal
    limits: pd.DataFrame        # limiar daquele mês por sinal (série em escada)
    crossings: pd.DataFrame     # score > limiar (bruto, sem persistência)
    flags: pd.DataFrame         # cruzamento sustentado pelo tempo exigido
    alert: pd.Series            # nº de sinais sustentados >= confirm
    n_active: pd.Series
    episodes: list[pd.Timestamp]
    events: list[dict]          # eventos físicos com detecção e antecedência
    false_positives: list[pd.Timestamp]
    stops: list[pd.Timestamp]
    metrics: dict
    episode_spans: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    sensors: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    operability: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    # ---------------------------------------------------------------- leitura
    @property
    def ratio(self) -> pd.DataFrame:
        """Score em múltiplos do limiar: 1,0 = borda do normal, para todos os sinais."""
        return self.scores / self.limits

    @property
    def confirm(self) -> int:
        return int(self.trial["confirm"])

    def label(self) -> str:
        t = self.trial
        janela = t.get("baseline_label") or (
            "acumulativo" if t.get("baseline_mode") == "acumulativo"
            else f"{t['baseline_hours']} h")
        return (f"{t['model']} | {t['grid']} | baseline {janela} | "
                f"exclui {t['exclude_days']}d | EWMA {t['ewma']} | q{t['threshold']} | "
                f"sustentado {t['sustain']} | {t['confirm']} sinais")

    def events_table(self, window: str = "48h") -> pd.DataFrame:
        horas = self.operating_hours_before(window)
        rows = []
        for event in self.events:
            rows.append({
                "falha": event["inicio"],
                "mecanismo": event["mecanismo"],
                "detectada": event["detectada"],
                "antecedencia_h": event.get("lead_h"),
                "primeiro_alerta": event.get("primeiro_alerta"),
                # sem tempo de operação antes da falha não há janela precursora
                # possível: distingue "o modelo perdeu" de "não havia o que ver"
                f"operou_nas_{window}": horas.get(event["inicio"]),
                "trips": len(event["trips"]),
            })
        return pd.DataFrame(rows)

    def operating_hours_before(self, window: str = "48h") -> pd.Series:
        """Horas de operação na janela que antecede cada falha."""
        if self.operability.empty:
            return pd.Series(dtype=float)
        operando = self.operability["in_operation"]
        passo = operando.index.to_series().diff().dt.total_seconds().median() or 30.0
        return pd.Series({
            event["inicio"]: round(
                float(operando.loc[event["inicio"] - pd.Timedelta(window):
                                   event["inicio"]].sum()) * passo / 3600, 1)
            for event in self.events})

    def episodes_table(self) -> pd.DataFrame:
        """Um episódio por linha, com duração — um alerta de 10 min e um de 6 h
        pesam muito diferente para quem está de plantão."""
        matched = {t for e in self.events for t in e.get("episodios", [])}
        falsos = set(self.false_positives)
        duracao = {inicio: (fim - inicio) for inicio, fim in self.episode_spans}
        rows = []
        for ep in self.episodes:
            rows.append({
                "inicio": ep,
                "duracao_h": round(duracao.get(ep, pd.Timedelta(0)).total_seconds() / 3600, 2),
                "tipo": ("detecção" if ep in matched else
                         "falso positivo" if ep in falsos else
                         "antes de parada (não contado)"),
            })
        return pd.DataFrame(rows)

    def anomaly_index(self, start=None, end=None) -> pd.DatetimeIndex:
        """Instantes em alerta confirmado (o que vira ponto vermelho no gráfico)."""
        alert = self.alert
        if start is not None or end is not None:
            alert = alert.loc[start:end]
        return alert[alert.fillna(False)].index


# ------------------------------------------------------------------- replay
class DetectionReplay:
    """Roda uma configuração do AutoML guardando as séries, não só as métricas."""

    def __init__(self, bundle=None, eval_start: str = "2025-02",
                 eval_end: str = "2026-04", max_fp_per_month: float = 1.0,
                 with_vibration: bool = False, cache_dir: Path | None = None,
                 single_model: bool = False) -> None:
        self.automl = load_automl()
        self.cache_dir = cache_dir or config.INTERIM
        self.single_model = single_model
        self.bundle = bundle or CachedBundle(with_vibration=with_vibration,
                                            cache_dir=self.cache_dir,
                                            single_model=single_model).build()
        self.evaluator = self.automl.WalkForwardEvaluator(
            self.bundle, eval_start, eval_end, max_fp_per_month)

    # ------------------------------------------------------------ etapa cara
    def _parts(self, model: str, grid: str, baseline, exclude_days: int,
               exclude_alarm_h: float = 0.0) -> dict:
        """Scores brutos por mês, com cache em disco (o ajuste do AE é o gargalo).

        A chave do cache usa o rótulo da política, então os arquivos gravados
        antes da migração (``..._300h_...``, ``..._acum_...``) continuam válidos.
        """
        arq = "unico" if getattr(self, "single_model", False) else "4sinais"
        key = f"{arq}_{model}_{grid}_{baseline.label}_{exclude_days}_{exclude_alarm_h}"
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        cache = self.cache_dir / f"replay_parts_{key}_{digest}.pkl"
        if cache.exists():
            import joblib
            print(f"[scores] cache {cache.name}")
            return joblib.load(cache)
        print(f"[scores] ajustando {model} walk-forward (grade {grid}, "
              f"baseline {baseline.label}, exclui {exclude_days}d + "
              f"{exclude_alarm_h}h de alarme)...", flush=True)
        parts = self.evaluator.raw_scores(model, grid, baseline, exclude_days,
                                          exclude_alarm_h)
        import joblib
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(parts, cache, compress=3)
        return parts

    # ------------------------------------------------------------------ run
    def run(self, model: str = "ae", grid: str = "30s", baseline_hours: int = 300,
            baseline_mode: str = "movel", exclude_days: int = 7,
            exclude_alarm_h: float = 0.0, ewma: str = "30min",
            threshold: float = 99.99, sustain: str = "2h", confirm: int = 3,
            min_alert: str = "0min", keep_sensors: bool = True,
            mask_stops: bool = False, baseline=None,
            threshold_kind: str = "percentil") -> ReplayResult:
        """Reexecuta uma configuração e devolve as séries.

        ``mask_stops`` é um **diagnóstico**, não o padrão. O score do AutoML é
        suavizado por EWMA *depois* de mascarar a parada, e a média exponencial
        carrega o último valor válido para frente — então o score continua
        definido com a máquina parada e o alerta pode disparar (ou continuar
        disparado) sobre uma máquina desligada. Com ``mask_stops=True`` o alerta
        é anulado fora da operação, o que mede quanto do falso positivo vem
        desse arraste. Fora desse modo, o replay confere exatamente com o
        avaliador do AutoML — e a conferência é verificada.
        """
        automl = self.automl
        # `baseline` é a forma nova; os kwargs baseline_hours/baseline_mode
        # continuam aceitos para os notebooks já escritos não quebrarem.
        if baseline is None:
            baseline = (automl.BaselinePolicy() if baseline_mode == "acumulativo"
                        else automl.BaselinePolicy(window_hours=float(baseline_hours)))
        trial = automl.Trial(model=model, grid=grid, baseline=baseline,
                            exclude_days=exclude_days,
                            exclude_alarm_h=exclude_alarm_h, ewma=ewma,
                            threshold=threshold, sustain=sustain, confirm=confirm,
                            min_alert=min_alert, threshold_kind=threshold_kind)
        parts = self._parts(model, grid, baseline, exclude_days, exclude_alarm_h)
        # alimenta o cache do avaliador para que evaluate() não recalcule
        self.evaluator._cache[(model, grid, baseline, exclude_days,
                               exclude_alarm_h)] = parts

        frame, oper = self.bundle.grid(grid)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        n_sustain = max(int(pd.Timedelta(sustain).total_seconds() / step), 1)
        halflife = pd.Timedelta(ewma)

        def smooth(series: pd.Series) -> pd.Series:
            return series.ewm(halflife=halflife, times=series.index).mean()

        order = [name for name, months in parts.items() if months]
        scores, limits, crossings, flags = {}, {}, {}, {}
        for name in order:
            pct = trial.threshold_for(name, order)
            s_parts, l_parts, c_parts, f_parts = [], [], [], []
            for base_serie, test in parts[name]:
                valores = smooth(base_serie).to_numpy(dtype=float)
                valores = valores[np.isfinite(valores)]
                limit = self.evaluator._limite(
                    np.sort(valores)[::-1], int(valores.size), pct, threshold_kind)
                suave = smooth(test)
                cruzou = suave > limit
                s_parts.append(suave)
                l_parts.append(pd.Series(limit, index=test.index))
                c_parts.append(cruzou)
                f_parts.append(cruzou.astype(int)
                               .rolling(n_sustain, min_periods=n_sustain).sum() >= n_sustain)
            scores[name] = pd.concat(s_parts).sort_index()
            limits[name] = pd.concat(l_parts).sort_index()
            crossings[name] = pd.concat(c_parts).sort_index()
            flags[name] = pd.concat(f_parts).sort_index()

        scores = pd.DataFrame(scores)
        limits = pd.DataFrame(limits)
        crossings = pd.DataFrame(crossings).fillna(False)
        flags = pd.DataFrame(flags).fillna(False)
        if mask_stops:
            operando = oper["stable"].reindex(flags.index, fill_value=False)
            flags = flags.where(operando, other=False)
            crossings = crossings.where(operando, other=False)
            scores = scores.where(operando)
        n_active = flags.sum(axis=1)
        alert = n_active >= confirm
        n_min = max(int(pd.Timedelta(min_alert).total_seconds() / step), 1)
        if n_min > 1:
            alert = (alert.astype(int).rolling(n_min, min_periods=n_min).sum()
                     >= n_min).fillna(False)

        episodes = self.evaluator._episodes(alert)
        spans = self._spans(alert, automl.EPISODE_GAP)
        stops = self.evaluator._stops(oper)
        window = pd.Timedelta(automl.DETECTION_WINDOW)

        events, matched = [], set()
        for event in self.evaluator.events:
            inside = [ep for ep in episodes
                      if event["inicio"] - window <= ep <= event["inicio"]]
            registro = dict(event)
            registro["detectada"] = bool(inside)
            registro["episodios"] = inside
            if inside:
                registro["primeiro_alerta"] = inside[0]
                registro["lead_h"] = round((event["inicio"] - inside[0]).total_seconds() / 3600, 1)
                matched.update(inside)
            events.append(registro)

        false_positives = []
        for ep in episodes:
            if ep in matched:
                continue
            horizonte = ep + window
            perto_de_trip = any(ep <= e["inicio"] <= horizonte for e in self.evaluator.events)
            perto_de_parada = any(ep <= s <= horizonte for s in stops)
            if not (perto_de_trip or perto_de_parada):
                false_positives.append(ep)

        leads = [e["lead_h"] for e in events if e["detectada"]]
        dias = float(oper["stable"].reindex(alert.index, fill_value=False).sum()) * step / 86400
        fp_mes = round(len(false_positives) / max(dias, 1) * 30, 2)
        base = next(iter(parts.values()))
        metrics = {
            "eventos_detectados": len(leads),
            "eventos_total": len(events),
            "lead_medio_h": round(float(np.mean(leads)), 1) if leads else None,
            "fp_episodios": len(false_positives),
            "fp_por_mes": fp_mes,
            "dias_avaliados": round(dias, 1),
            "episodios_totais": len(episodes),
            "baseline_amostras": int(np.mean([len(b) for b, _ in base])) if base else 0,
            "fp_horas_por_mes": round(sum(
                (fim - ini).total_seconds() / 3600 for ini, fim in spans
                if ini in set(false_positives)) / max(dias, 1) * 30, 1),
            "aprovado": bool(fp_mes <= self.evaluator.max_fp_per_month),
            "paradas_mascaradas": mask_stops,
        }
        oficial = self.evaluator.evaluate(trial)
        metrics.update({
            "fp_explicados": oficial.fp_explicados,
            "fp_por_mes_liquido": oficial.fp_por_mes_liquido,
            "alarmes_antecipados": f"{oficial.alarmes_antecipados}/{oficial.alarmes_total}",
            "alarmes_gas_antecipados":
                f"{oficial.alarmes_utilidade_antecipados}/{oficial.alarmes_utilidade_total}",
        })
        if not mask_stops:
            # sem o diagnóstico ligado, o replay tem de bater com o AutoML
            if (metrics["eventos_detectados"] != oficial.eventos_detectados
                    or metrics["fp_episodios"] != oficial.fp_episodios):
                raise AssertionError(
                    "replay divergiu do avaliador do AutoML "
                    f"(detecção {metrics['eventos_detectados']} vs {oficial.eventos_detectados}, "
                    f"FP {metrics['fp_episodios']} vs {oficial.fp_episodios})")
        print(f"[replay] {trial.label()}"
              f"{' +mascara paradas' if mask_stops else ''} -> "
              f"{metrics['eventos_detectados']}/{metrics['eventos_total']} eventos | "
              f"lead {metrics['lead_medio_h']} h | FP {fp_mes}/mês")

        return ReplayResult(
            trial=dict(model=model, grid=grid, baseline=baseline,
                       baseline_label=baseline.label,
                       baseline_hours=baseline.window_hours,
                       baseline_mode=baseline.modo_legado,
                       exclude_days=exclude_days, exclude_alarm_h=exclude_alarm_h,
                       ewma=ewma, threshold=threshold, threshold_kind=threshold_kind,
                       sustain=sustain, confirm=confirm, min_alert=min_alert),
            scores=scores, limits=limits, crossings=crossings, flags=flags,
            alert=alert, n_active=n_active, episodes=episodes, events=events,
            false_positives=false_positives, stops=stops, metrics=metrics,
            episode_spans=spans,
            sensors=frame if keep_sensors else pd.DataFrame(),
            operability=oper if keep_sensors else pd.DataFrame(),
        )


    # ------------------------------------------------- erro por sensor
    def erro_por_sensor(self, evento: dict, familia: str | None = None,
                        model: str = "ae", grid: str = "30s", baseline_hours: int = 300,
                        exclude_days: int = 7, exclude_alarm_h: float = 0.0,
                        antes: str = "5d", depois: str = "12h") -> pd.DataFrame:
        """Erro de reconstrução **de cada sensor** em volta de uma falha.

        O score de uma família é a média do erro quadrático sobre os sensores dela;
        aqui a média é aberta, para responder *qual sensor* está puxando o erro.

        O modelo é ajustado no mesmo baseline que o walk-forward usaria para o mês
        da falha (mesmos dias, mesma máscara de operação, mesma limpeza), então o
        erro mostrado é o mesmo que entra no score — só que não agregado.
        Só faz sentido para modelos com reconstrução (``pca``, ``ae``).
        """
        if model not in {"pca", "ae"}:
            raise ValueError(f"{model} não reconstrói a entrada; use pca ou ae")
        familia = familia or MECHANISM_FAMILY.get(evento["mecanismo"], "temperatura")
        colunas = [c for c in self.bundle.families[familia]
                   if c in self.bundle.raw.columns]

        frame, oper = self.bundle.grid(grid)
        mes = pd.Timestamp(evento["inicio"]).to_period("M").to_timestamp()
        passo = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        elegivel = (oper["stable"]
                    & self.evaluator._baseline_mask(frame.index, exclude_days,
                                                    exclude_alarm_h)).loc[:mes]
        indices = elegivel[elegivel].index[-int(baseline_hours * 3600 / passo):]
        scorer = self.automl.FamilyScorer(model).fit(frame.loc[indices, colunas])

        janela = frame.loc[pd.Timestamp(evento["inicio"]) - pd.Timedelta(antes):
                           pd.Timestamp(evento["fim"]) + pd.Timedelta(depois), colunas]
        dados = janela.dropna()
        escalado = scorer.scaler.transform(dados)
        if model == "pca":
            reconstruido = scorer.model.inverse_transform(scorer.model.transform(escalado))
        else:
            reconstruido = scorer.model.predict(escalado)
        return pd.DataFrame((escalado - reconstruido) ** 2, index=dados.index,
                            columns=colunas)

    @staticmethod
    def _spans(alert: pd.Series, gap: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(início, fim) de cada episódio — mesmo agrupamento de ``_episodes``."""
        ativo = alert.fillna(False)
        if not ativo.any():
            return []
        marcas = ativo[ativo].index.to_series()
        grupo = (marcas.diff() > pd.Timedelta(gap)).cumsum()
        return [(trecho.iloc[0], trecho.iloc[-1])
                for _, trecho in marcas.groupby(grupo)]


def best_trial_from_task(task_id: str, max_fp: float | None = None) -> dict:
    """Melhor configuração de uma task de AutoML do ClearML (sem recalcular nada).

    Com ``max_fp``, escolhe a melhor configuração sob outro teto de falso
    positivo que não o usado na busca — útil para comparar o ponto conservador
    com o ponto operacional.
    """
    from clearml import Task

    task = Task.get_task(task_id=task_id)
    table = task.artifacts["automl_results"].get()
    teto = max_fp if max_fp is not None else float(
        json.loads(json.dumps(task.get_parameters().get("Args/max_fp_per_month", 1.0))))
    pool = table[table["fp_por_mes"] <= float(teto)]
    if pool.empty:
        pool = table
    melhor = pool.sort_values(
        ["eventos_detectados", "lead_medio_h", "fp_por_mes"],
        ascending=[False, False, True]).iloc[0]
    return {c.removeprefix("trial."): melhor[c] for c in table.columns
            if c.startswith("trial.")}
