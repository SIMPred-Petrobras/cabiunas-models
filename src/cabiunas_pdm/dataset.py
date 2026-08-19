"""Construção do dataset canônico: grade 2 min, UTC-3, sensores + operabilidade.

Fontes de **sensores** (inalteradas — preservam a proveniência já validada):
  1. TagsSelecionadas 30s/2min (ABR_26) — 38 tags, já em UTC-3.
  2. Arquivos interpolated do PortalIntegridade (UTC) que cobrem meses ausentes
     na seleção (01/2024, 01/2025, 04/2025) — reduzidos às mesmas 38 tags.

Fonte de **operabilidade** (acréscimo de 04/08/2026): NGP_A/NPT_A/NCPSR_A/
TM_TORQUE_A vêm dos mensais PortalIntegridade (86 tags), a única fonte local
desses sinais, e são unidos por *left join* na grade canônica. O join não altera
nenhuma coluna de sensor — só acrescenta os sinais de rotação onde existem.

A máscara de operação sai do ``OperabilityResolver``: NGP_A >= limiar onde o
sinal existe, ``RUNNING_A == 1`` como fallback explícito onde não existe, sempre
excluindo a janela pós-partida. A coluna ``operability_source`` registra qual
regra governou cada instante.

Saída: data/processed/canonico_2min.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import cleaning, config, sources
from .operability import OperabilityResolver


def _read_interpolated_xlsx(path: Path, utc: bool) -> pd.DataFrame:
    """Compat: leitura de um mensal reduzido às tags canônicas."""
    catalog = sources.MonthlyExcelCatalog([path.parent])
    item = sources.MonthlyFile(month="", path=path, utc=utc, origin=path.parent.name)
    return catalog.read(item, tags=config.CANONICAL_TAGS)


def _sensor_frame(cache_tagssel: Path | None, verbose: bool) -> pd.DataFrame:
    """Concatena as fontes de sensores (comportamento original preservado)."""
    frames: list[pd.DataFrame] = []
    if cache_tagssel and Path(cache_tagssel).exists():
        base = pd.read_parquet(cache_tagssel)
        base = cleaning.strip_prefix(base)
        base = base[[c for c in config.CANONICAL_TAGS if c in base.columns]]
        frames.append(base)
        if verbose:
            print(f"[cache] TagsSelecionadas: {base.shape} "
                  f"{base.index.min()} → {base.index.max()}")
    else:
        files = sorted(config.TAGSSEL_30S.rglob("*.xlsx")) + \
                sorted(config.TAGSSEL_2MIN.rglob("*.xlsx"))
        for path in files:
            frame = _read_interpolated_xlsx(path, utc=False)
            frames.append(frame)
            if verbose:
                print(f"[30s/2min] {path.name}: {frame.shape}")

    for path in config.EXTRA_INTERPOLATED_UTC:
        if not path.exists():
            print(f"[aviso] ausente: {path}")
            continue
        frame = _read_interpolated_xlsx(path, utc=True)
        frames.append(frame)
        if verbose:
            print(f"[extra UTC] {path.name}: {frame.shape} "
                  f"{frame.index.min()} → {frame.index.max()}")

    full = pd.concat(frames).sort_index()
    return full[~full.index.duplicated(keep="first")]


def _join_operability(frame: pd.DataFrame, refresh: bool, verbose: bool) -> pd.DataFrame:
    """Anexa NGP/NPT/NCPSR/torque do PortalIntegridade, sem tocar nos sensores."""
    oper = sources.load_operability_frame(refresh=refresh, verbose=verbose)
    if oper.empty:
        print("[aviso] nenhuma fonte de operabilidade encontrada — só RUNNING_A")
        return frame
    cols = [c for c in config.OPERABILITY_TAGS if c in oper.columns]
    oper = oper[cols].reindex(frame.index)
    for col in cols:
        # não sobrescreve valor já presente (caso um mensal traga o sinal)
        if col in frame.columns:
            frame[col] = frame[col].combine_first(oper[col])
        else:
            frame[col] = oper[col]
    if verbose:
        got = frame[config.TAG_OPERABILITY].notna().sum() if config.TAG_OPERABILITY in frame else 0
        print(f"[operabilidade] {cols} anexados; "
              f"{int(got):,} linhas com {config.TAG_OPERABILITY} válido")
    return frame


def build(cache_tagssel: Path | None = None, verbose: bool = True,
          refresh_operability: bool = False,
          ngp_threshold: float | None = None) -> pd.DataFrame:
    """Monta o canônico completo (sensores + operabilidade + máscaras).

    `cache_tagssel` aceita um parquet 2 min pré-consolidado das TagsSelecionadas
    (economiza ~10 min de leitura de xlsx).
    """
    full = _sensor_frame(cache_tagssel, verbose)
    full = _join_operability(full, refresh_operability, verbose)

    # limpeza física dos sensores + flags de congelamento
    full = cleaning.apply_physical_ranges(full)
    frozen = cleaning.freeze_flags(full, full[config.TAG_RUNNING])
    full["n_frozen_sensors"] = frozen.sum(axis=1).astype("int16")

    resolver = OperabilityResolver(ngp_threshold=ngp_threshold)
    full = resolver.attach(full)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "canonico_2min.parquet"
    full.to_parquet(out)
    if verbose:
        counts = full["operability_source"].value_counts()
        print(f"\nsalvo: {out}  shape={full.shape}")
        print(f"período: {full.index.min()} → {full.index.max()}")
        print(f"limiar NGP: {resolver.ngp_threshold} | "
              f"transiente excluído: {resolver.startup_exclude}")
        print(f"em operação: {int(full['in_operation'].sum()):,} | "
              f"estável: {int(full['stable'].sum()):,}")
        print("fonte da máscara: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return full


def load() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "canonico_2min.parquet")
