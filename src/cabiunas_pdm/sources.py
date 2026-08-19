"""Camada de acesso às fontes brutas de série temporal do TC-330.03A.

Duas famílias de arquivo mensal coexistem na base local, com esquemas diferentes:

``PortalIntegridade``
    86 tags, timestamp **UTC**, inclui os sinais de rotação (``NGP_A``,
    ``NPT_A``, ``NCPSR_A``, ``TM_TORQUE_A``). Espalhado por quatro pastas com
    duplicação; o catálogo resolve mês → melhor arquivo por ordem de prioridade.

``TagsSelecionadas``
    38 tags de modelagem, timestamp **UTC-3**, **sem** sinais de rotação.

Os catálogos apenas localizam e leem; nenhuma regra de limpeza ou de modelagem
vive aqui (isso é de ``cleaning``/``operability``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import cleaning, config

_MONTH_RE = re.compile(r"(?P<month>\d{2})_(?P<year>\d{4})")


@dataclass(frozen=True)
class MonthlyFile:
    """Um arquivo mensal já resolvido para um período e fuso."""

    month: str          # "YYYY-MM"
    path: Path
    utc: bool           # True => índice precisa deslocar para UTC-3
    origin: str         # pasta-raiz de onde veio (proveniência)

    @property
    def period(self) -> pd.Period:
        return pd.Period(self.month, freq="M")


class MonthlyExcelCatalog:
    """Base comum: descobre arquivos mensais e lê um subconjunto de tags."""

    utc: bool = False
    pattern: str = "*.xlsx"

    def __init__(self, roots: list[Path]) -> None:
        self.roots = [Path(r) for r in roots]

    # ------------------------------------------------------------ descoberta
    def catalog(self) -> dict[str, MonthlyFile]:
        """Mapa mês → arquivo, respeitando a ordem de prioridade das raízes."""
        found: dict[str, MonthlyFile] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob(self.pattern)):
                match = _MONTH_RE.search(path.name)
                if match is None:
                    continue
                month = f"{match.group('year')}-{match.group('month')}"
                if month in found:          # primeira raiz vence
                    continue
                found[month] = MonthlyFile(month=month, path=path, utc=self.utc,
                                           origin=self._origin(path))
        return dict(sorted(found.items()))

    def months(self) -> list[str]:
        return list(self.catalog())

    def _origin(self, path: Path) -> str:
        for root in self.roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return root.name if root.name else str(root)
        return str(path.parent)

    # ------------------------------------------------------------ leitura
    def read(self, item: MonthlyFile, tags: list[str] | None = None,
             grid: str | None = None) -> pd.DataFrame:
        """Lê um mensal, normaliza índice/fuso e reamostra para a grade.

        `tags` filtra colunas (as ausentes são ignoradas silenciosamente, pois
        o esquema varia entre as duas famílias de arquivo).
        """
        frame = pd.read_excel(item.path, engine="calamine")
        time_col = frame.columns[0]
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col]).set_index(time_col).sort_index()
        if item.utc:
            frame.index = frame.index - pd.Timedelta(hours=config.UTC_OFFSET_H)
        frame = cleaning.strip_prefix(frame)
        if tags is not None:
            frame = frame[[c for c in tags if c in frame.columns]]
        frame = cleaning.coerce_numeric(frame)
        return frame.resample(grid or config.GRID).median()

    def read_months(self, tags: list[str] | None = None, months: list[str] | None = None,
                    grid: str | None = None, verbose: bool = False) -> pd.DataFrame:
        """Concatena vários meses em um único frame ordenado e sem duplicatas."""
        catalog = self.catalog()
        wanted = months or list(catalog)
        frames: list[pd.DataFrame] = []
        for month in wanted:
            item = catalog.get(month)
            if item is None:
                if verbose:
                    print(f"[{self.__class__.__name__}] mês ausente: {month}")
                continue
            frame = self.read(item, tags=tags, grid=grid)
            frame["source_month"] = month
            frame["source_origin"] = item.origin
            frames.append(frame)
            if verbose:
                print(f"[{self.__class__.__name__}] {month} ({item.origin}): "
                      f"{frame.shape[0]} linhas", flush=True)
        if not frames:
            return pd.DataFrame()
        full = pd.concat(frames).sort_index()
        return full[~full.index.duplicated(keep="first")]


class PortalIntegridadeCatalog(MonthlyExcelCatalog):
    """Mensais de 86 tags em UTC — única fonte local de NGP_A/NCPSR_A."""

    utc = True
    pattern = "*interpolated_PortalIntegridade.xlsx"

    def __init__(self, roots: list[Path] | None = None) -> None:
        super().__init__(roots or config.PORTAL_ROOTS)


class TagsSelecionadasCatalog(MonthlyExcelCatalog):
    """Mensais de 38 tags de modelagem, já em UTC-3."""

    utc = False
    pattern = "*interpolated_TagsSelecionadas*.xlsx"

    def __init__(self, roots: list[Path] | None = None) -> None:
        super().__init__(roots or [config.TAGSSEL_30S, config.TAGSSEL_2MIN])


def load_operability_frame(cache: Path | None = None, refresh: bool = False,
                           verbose: bool = True) -> pd.DataFrame:
    """Série 2 min dos sinais de operabilidade (NGP/NPT/NCPSR/torque + RUNNING).

    Ler os 25 mensais do PortalIntegridade custa alguns minutos, então o
    resultado é cacheado em ``data/interim``. Use ``refresh=True`` para refazer.
    """
    cache = Path(cache or config.INTERIM / "operability_portal_2min.parquet")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    tags = config.OPERABILITY_TAGS + [config.TAG_RUNNING, config.TAG_MAINT]
    frame = PortalIntegridadeCatalog().read_months(tags=tags, verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame
