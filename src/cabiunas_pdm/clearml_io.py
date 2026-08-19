"""Leitura reproduzível de datasets versionados no ClearML.

Além do leitor genérico, expõe ``CabiunasClearMLSource``: a fonte de dados do
projeto quando a análise deve usar **exclusivamente** o que está publicado no
ClearML, sem depender do canônico local.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import cleaning, config

TABULAR_SUFFIXES = (".parquet", ".feather", ".csv", ".xlsx", ".xls")

# Versões (tasks) publicadas no ClearML — projeto TesteMLCab.
# O ID que o SDK precisa é o da *task da versão* (segundo id da URL), não o da
# coleção exibida primeiro em /datasets/simple/<coleção>/tasks/<versão>.
DATASET_ALARMES_MAPEADOS = "68b25f9db0b8471a90b8100800d26e9a"   # "Cabiunas brutos 2025-2026 alarmes mapeados"
DATASET_SERIE_2025_2026 = "1180990c20984b9ab66bbb853832fd84"    # "Cabiunas 2025+2026"

# RUNNING_A vem interpolado e assume valores fracionários no instante da
# transição (ex.: 0,167 na parada de 26/02/2026 15:34). Comparar com ``== 1``
# perde essas paradas, então a operabilidade usa limiar.
RUNNING_THRESHOLD = 0.5


@dataclass(frozen=True)
class ClearMLDatasetRef:
    """Referência imutável a uma versão de dataset no ClearML."""

    dataset_id: str
    label: str


class ClearMLDatasetReader:
    """Baixa uma versão de dataset e localiza seus arquivos tabulares."""

    def download(self, reference: ClearMLDatasetRef) -> Path:
        try:
            from clearml import Dataset
        except ImportError as exc:
            raise RuntimeError(
                "ClearML não está instalado. Use `uv sync --extra clearml` ou "
                "`pip install -e '.[clearml]'`."
            ) from exc
        return Path(Dataset.get(dataset_id=reference.dataset_id).get_local_copy())

    @staticmethod
    def tabular_files(dataset_dir: Path) -> list[Path]:
        files = sorted(p for p in dataset_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo tabular em {dataset_dir}")
        return files

    @staticmethod
    def read(path: Path, nrows: int | None = None) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".feather":
            return pd.read_feather(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=nrows)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="calamine", nrows=nrows)
        raise ValueError(f"Formato não suportado: {path}")


class CabiunasClearMLSource:
    """Série de sensores e alarmes do TC-330.03A **apenas** do ClearML.

    O CSV publicado é bruto: colunas de sensor vêm como texto (misturando
    números e objetos de status do PI), há sentinelas fisicamente impossíveis e
    ``RUNNING_A`` é fracionário nas transições. Esta classe entrega os dados já
    tratados, e registra o que foi tratado em ``self.report``.

    Não usa o canônico local em nenhum ponto.
    """

    STATUS_HINT = "IsSystem"       # marca dos dicts de status do PI no texto

    def __init__(self, dataset_id: str = DATASET_ALARMES_MAPEADOS,
                 cache_dir: Path | None = None) -> None:
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir) if cache_dir else config.INTERIM
        self.reader = ClearMLDatasetReader()
        self.report: dict = {}

    # ------------------------------------------------------------- download
    def _dataset_dir(self) -> Path:
        return self.reader.download(ClearMLDatasetRef(self.dataset_id, "clearml"))

    def _file(self, needle: str) -> Path:
        files = self.reader.tabular_files(self._dataset_dir())
        matches = [p for p in files if needle in p.name]
        if not matches:
            raise FileNotFoundError(f"'{needle}' não está no dataset {self.dataset_id}. "
                                    f"Arquivos: {[p.name for p in files]}")
        return matches[0]

    # -------------------------------------------------------------- sensores
    def load_sensors(self, refresh: bool = False) -> pd.DataFrame:
        """Série 30 s limpa: sensores numéricos, sentinelas → NaN, RUNNING contínuo."""
        cache = self.cache_dir / f"clearml_{self.dataset_id[:8]}_sensores.parquet"
        if cache.exists() and not refresh:
            frame = pd.read_parquet(cache)
            self.report = {"origem": "cache", "arquivo": str(cache)}
            return frame

        path = self._file("sensores")
        raw = pd.read_csv(path, engine="pyarrow")
        time_col = raw.columns[0]
        raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
        raw = raw.dropna(subset=[time_col]).set_index(time_col).sort_index()

        # quanto do dado bruto é objeto de status do PI (antes de virar NaN)
        as_text = raw.astype("string")
        status_cells = int(as_text.apply(
            lambda s: s.str.contains(self.STATUS_HINT, na=False)).to_numpy().sum())

        numeric = raw.apply(pd.to_numeric, errors="coerce")
        before = int(numeric.notna().to_numpy().sum())
        clean = cleaning.apply_physical_ranges(numeric)
        removed_by_range = before - int(clean.notna().to_numpy().sum())

        self.report = {
            "origem": "clearml",
            "dataset_id": self.dataset_id,
            "arquivo": path.name,
            "linhas": int(len(clean)),
            "colunas": int(clean.shape[1]),
            "periodo": [str(clean.index.min()), str(clean.index.max())],
            "passo_s": float(clean.index.to_series().diff().dt.total_seconds().median()),
            "celulas_status_pi": status_cells,
            "celulas_fora_da_faixa_fisica": removed_by_range,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(cache)
        return clean

    # --------------------------------------------------------------- alarmes
    def load_alarms(self) -> pd.DataFrame:
        """Alarmes com timestamp normalizado e marcação de nível TRIP.

        No dataset "alarmes mapeados" a coluna ``Tag Alarme`` já traz o nome da
        **coluna de sensor** correspondente, o que permite ligar alarme→sensor
        sem tabela auxiliar.
        """
        path = self._file("alarmes")
        alarms = pd.read_csv(path)
        when = alarms.columns[0]
        alarms[when] = self._parse_datetimes(alarms[when])
        alarms = alarms.dropna(subset=[when]).rename(columns={when: "quando"})
        alarms["ativado"] = alarms["Status"].astype(str).str.startswith("ACT")
        texto = (alarms["Tag Alarme"].astype(str) + " "
                 + alarms["Descrição Alarme"].astype(str).str.upper())
        alarms["nivel_trip"] = (texto.str.contains(r"LL_|HH_|ALL|AHH", regex=True)
                                | texto.str.contains("TRIP"))
        alarms["mapeado_para_coluna"] = alarms["Tag Alarme"].astype(str).str.match(
            r"^(954005_|TC382_|TV_|T5_|PI_5|HSX_|RUNNING)")
        return alarms.set_index("quando").sort_index()

    def load_alarms_merged(self, other_dataset_id: str = DATASET_SERIE_2025_2026) -> pd.DataFrame:
        """Alarmes com tag original **e** coluna de sensor correspondente.

        As duas versões publicadas descrevem os mesmos 6.851 eventos, alinhados
        linha a linha, mas guardam informações diferentes em ``Tag Alarme``:
        a versão "alarmes mapeados" traz a coluna da série (``954005_624_PI_0315``)
        e a versão "2025+2026" traz a tag do alarme (``PAL_6240315``) — que é a
        única que revela o nível (``PALL_``/``TAHH_`` = TRIP). Unir as duas dá
        as duas coisas.
        """
        mapped = self.load_alarms()
        other = CabiunasClearMLSource(dataset_id=other_dataset_id).load_alarms()
        if not mapped.index.equals(other.index):
            raise ValueError("as duas versões de alarmes não estão alinhadas; "
                             "una por 'Identificador' antes de comparar")
        merged = mapped.copy()
        merged["coluna_sensor"] = mapped["Tag Alarme"].to_numpy()
        merged["tag_alarme"] = other["Tag Alarme"].to_numpy()
        texto = merged["tag_alarme"].astype(str) + " " + \
            merged["Descrição Alarme"].astype(str).str.upper()
        merged["nivel_trip"] = (texto.str.contains(r"LL_|HH_|ALL|AHH", regex=True)
                                | texto.str.contains("TRIP"))
        return merged

    @staticmethod
    def _parse_datetimes(values: pd.Series) -> pd.Series:
        """Parse tolerante ao formato da data.

        O CSV publicado no ClearML já vem em ISO (``2022-01-04 05:21:47``),
        enquanto o xlsx de origem usa ``DD/MM/AAAA``. Forçar ``dayfirst`` no
        arquivo ISO descarta linhas e inventa datas (fev virando dez), então o
        formato é detectado pelo que aproveita mais linhas sem extrapolar.
        """
        iso = pd.to_datetime(values, errors="coerce")
        dayfirst = pd.to_datetime(values, errors="coerce", dayfirst=True)
        return iso if iso.notna().sum() >= dayfirst.notna().sum() else dayfirst

    # ---------------------------------------------------------- operabilidade
    @staticmethod
    def operability(frame: pd.DataFrame, startup_exclude: str = "2h") -> pd.DataFrame:
        """Máscara de operação a partir de ``RUNNING_A`` com limiar (não ``== 1``).

        ``NGP_A`` seria a fonte preferida, mas **não existe nesta versão do
        dataset** — ver nota em reports/DIAGNOSTICO_OPERABILIDADE.md.
        """
        running = pd.to_numeric(frame[config.TAG_RUNNING], errors="coerce")
        in_operation = running.ge(RUNNING_THRESHOLD).fillna(False)
        starts = in_operation & ~in_operation.shift(fill_value=False)
        step = frame.index.to_series().diff().dt.total_seconds().median() or 30.0
        window = max(int(pd.Timedelta(startup_exclude).total_seconds() / step), 1)
        recent = starts.rolling(window, min_periods=1).max().astype(bool)
        return pd.DataFrame({"in_operation": in_operation,
                             "stable": in_operation & ~recent,
                             "running_raw": running}, index=frame.index)
