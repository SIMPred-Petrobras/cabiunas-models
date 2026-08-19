"""Definição de "equipamento em operação" — o sinal que governa toda a máscara.

Por que este módulo existe
--------------------------
Toda a modelagem só é válida em operação: um baseline contaminado com máquina
parada aprende o "normal" errado e gera falso positivo na partida seguinte.
Até aqui a máscara vinha de ``RUNNING_A`` (discreto do supervisório). A decisão
de projeto é usar ``NGP_A`` (velocidade da turbina geradora de gás, %) como
fonte de verdade, porque é uma grandeza física contínua: permite exigir *carga
mínima*, não apenas "ligado", e não depende da lógica interna do supervisório.

Como o fallback funciona
------------------------
``NGP_A`` só existe nos mensais PortalIntegridade (86 tags). Os meses vindos das
TagsSelecionadas (38 tags) não o têm. A resolução é portanto **por linha**, não
global: onde há NGP usa-se NGP; onde não há, cai-se para ``RUNNING_A`` — e a
coluna de proveniência registra qual regra governou cada instante, para que
nenhuma análise posterior confunda as duas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

SOURCE_NGP = "NGP_A"
SOURCE_FALLBACK = "RUNNING_A"
SOURCE_NONE = "indisponivel"


@dataclass(frozen=True)
class OperabilityResolution:
    """Resultado da resolução de operabilidade para um recorte temporal."""

    in_operation: pd.Series          # bool — máquina girando (sem excluir transiente)
    stable: pd.Series                # bool — operação estável (exclui pós-partida)
    source: pd.Series                # str  — NGP_A | RUNNING_A | indisponivel
    threshold: float | None
    startup_exclude: str
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        counts = self.source.value_counts()
        total = max(int(len(self.source)), 1)
        return {
            "threshold_ngp": self.threshold,
            "startup_exclude": self.startup_exclude,
            "linhas": total,
            "em_operacao": int(self.in_operation.sum()),
            "estavel": int(self.stable.sum()),
            "fonte_pct": {str(k): round(float(v) / total * 100, 2)
                          for k, v in counts.items()},
            **self.stats,
        }


class OperabilityResolver:
    """Constrói a máscara de operação a partir de NGP com fallback para RUNNING.

    Parameters
    ----------
    ngp_threshold
        Limiar de ``NGP_A`` (em %) a partir do qual a máquina é considerada em
        operação. ``None`` desabilita o NGP e força o fallback — usado enquanto
        o limiar não estiver validado com a operação.
    require_ngp
        Se ``True``, levanta erro quando o NGP não cobre o recorte, em vez de
        cair para ``RUNNING_A``. Use nos treinos definitivos, quando a decisão
        for "não modelar sem sinal físico de operabilidade".
    """

    def __init__(self, ngp_threshold: float | None = None,
                 startup_exclude: str | None = None,
                 ngp_tag: str = config.TAG_OPERABILITY,
                 fallback_tag: str = config.TAG_RUNNING,
                 require_ngp: bool = False) -> None:
        self.ngp_threshold = (config.NGP_OPERATIONAL_THRESHOLD
                              if ngp_threshold is None else ngp_threshold)
        self.startup_exclude = startup_exclude or config.STARTUP_EXCLUDE
        self.ngp_tag = ngp_tag
        self.fallback_tag = fallback_tag
        self.require_ngp = require_ngp

    # ------------------------------------------------------------------ API
    def resolve(self, frame: pd.DataFrame) -> OperabilityResolution:
        index = frame.index
        ngp = self._numeric(frame, self.ngp_tag)
        run = self._numeric(frame, self.fallback_tag)

        use_ngp = ngp.notna() if self.ngp_threshold is not None else pd.Series(False, index=index)
        if self.require_ngp and not bool(use_ngp.any()):
            raise ValueError(
                f"{self.ngp_tag} indisponível no recorte "
                f"{index.min()} → {index.max()} e require_ngp=True. "
                "Extraia o sinal do PI ou passe require_ngp=False para usar "
                f"{self.fallback_tag} explicitamente."
            )

        by_ngp = use_ngp & ngp.ge(self.ngp_threshold if self.ngp_threshold is not None else np.inf)
        use_run = ~use_ngp & run.notna()
        by_run = use_run & run.eq(1)

        in_operation = (by_ngp | by_run).fillna(False).astype(bool)
        source = pd.Series(SOURCE_NONE, index=index, dtype="object")
        source[use_run] = SOURCE_FALLBACK
        source[use_ngp] = SOURCE_NGP

        stable = self._exclude_startup(in_operation)
        stats = self._agreement_stats(ngp, run)
        return OperabilityResolution(in_operation=in_operation, stable=stable,
                                     source=source, threshold=self.ngp_threshold,
                                     startup_exclude=self.startup_exclude, stats=stats)

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Devolve `frame` com as colunas de operabilidade anexadas."""
        resolution = self.resolve(frame)
        out = frame.copy()
        out["in_operation"] = resolution.in_operation
        out["stable"] = resolution.stable
        out["operability_source"] = pd.Categorical(
            resolution.source, categories=[SOURCE_NGP, SOURCE_FALLBACK, SOURCE_NONE])
        return out

    # -------------------------------------------------------------- internos
    @staticmethod
    def _numeric(frame: pd.DataFrame, tag: str) -> pd.Series:
        if tag not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[tag], errors="coerce")

    def _exclude_startup(self, in_operation: pd.Series) -> pd.Series:
        """Remove a janela pós-partida (transiente térmico/mecânico)."""
        starts = in_operation & ~in_operation.shift(fill_value=False)
        window = int(pd.Timedelta(self.startup_exclude) / pd.Timedelta(config.GRID))
        window = max(window, 1)
        recent = starts.rolling(window, min_periods=1).max().astype(bool)
        return (in_operation & ~recent).astype(bool)

    def _agreement_stats(self, ngp: pd.Series, run: pd.Series) -> dict:
        """Concordância NGP×RUNNING onde ambos existem — auditoria do fallback."""
        both = ngp.notna() & run.notna()
        if not bool(both.any()) or self.ngp_threshold is None:
            return {"linhas_com_ngp_e_running": int(both.sum())}
        ngp_on = ngp[both].ge(self.ngp_threshold)
        run_on = run[both].eq(1)
        return {
            "linhas_com_ngp_e_running": int(both.sum()),
            "concordancia_pct": round(float((ngp_on == run_on).mean() * 100), 3),
            "ngp_on_run_off": int((ngp_on & ~run_on).sum()),
            "ngp_off_run_on": int((~ngp_on & run_on).sum()),
        }


class OperabilityDiagnostics:
    """Estudo do NGP para propor e auditar o limiar de operação.

    Não decide sozinho: produz a evidência (cobertura, bimodalidade, vale entre
    os modos, concordância com RUNNING) para a validação com a operação.
    """

    def __init__(self, frame: pd.DataFrame, ngp_tag: str = config.TAG_OPERABILITY,
                 fallback_tag: str = config.TAG_RUNNING) -> None:
        self.frame = frame
        self.ngp = OperabilityResolver._numeric(frame, ngp_tag)
        self.run = OperabilityResolver._numeric(frame, fallback_tag)

    # ---------------------------------------------------------- cobertura
    def coverage_by_month(self) -> pd.DataFrame:
        month = self.frame.index.to_period("M")
        out = pd.DataFrame({
            "linhas": self.ngp.groupby(month).size(),
            "ngp_validos": self.ngp.notna().groupby(month).sum(),
            "running_validos": self.run.notna().groupby(month).sum(),
        })
        out["ngp_pct"] = (out["ngp_validos"] / out["linhas"] * 100).round(1)
        return out

    # ------------------------------------------------------- distribuição
    def distribution(self) -> dict:
        valid = self.ngp.dropna()
        if valid.empty:
            return {"n": 0}
        quantiles = {f"p{int(q*100):02d}": round(float(valid.quantile(q)), 2)
                     for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}
        return {"n": int(valid.size), "quantiles": quantiles,
                "pct_abaixo_1": round(float((valid < 1).mean() * 100), 2),
                "pct_acima_50": round(float((valid > 50).mean() * 100), 2)}

    def valley(self, low: float = 1.0, high: float = 100.0,
               bins: int = 200) -> dict:
        """Maior faixa vazia (ou rarefeita) entre o modo "parado" e o de operação.

        Um limiar colocado no meio desse vale separa os dois regimes sem cortar
        nenhum ponto de operação real.
        """
        valid = self.ngp.dropna()
        window = valid[(valid >= low) & (valid <= high)]
        if window.empty:
            return {"encontrado": False}
        counts, edges = np.histogram(window, bins=bins, range=(low, high))
        empty = counts == 0
        best_len, best_start = 0, None
        run_len, run_start = 0, None
        for i, is_empty in enumerate(empty):
            if is_empty:
                run_start = i if run_len == 0 else run_start
                run_len += 1
                if run_len > best_len:
                    best_len, best_start = run_len, run_start
            else:
                run_len = 0
        if best_start is None:
            return {"encontrado": False}
        lo, hi = float(edges[best_start]), float(edges[best_start + best_len])
        return {"encontrado": True, "vale_inicio": round(lo, 2), "vale_fim": round(hi, 2),
                "vale_largura": round(hi - lo, 2),
                "limiar_no_meio": round((lo + hi) / 2, 2)}

    # -------------------------------------------------------- concordância
    def agreement_curve(self, candidates: list[float]) -> pd.DataFrame:
        """Concordância com RUNNING para cada limiar candidato."""
        both = self.ngp.notna() & self.run.notna()
        ngp, run = self.ngp[both], self.run[both].eq(1)
        rows = []
        for thr in candidates:
            on = ngp.ge(thr)
            rows.append({
                "limiar": thr,
                "concordancia_pct": round(float((on == run).mean() * 100), 3),
                "ngp_on_run_off": int((on & ~run).sum()),
                "ngp_off_run_on": int((~on & run).sum()),
                "horas_operacao": round(float(on.sum()) * 2 / 60, 1),
            })
        return pd.DataFrame(rows)

    def operating_load_profile(self, threshold: float) -> dict:
        """Distribuição do NGP *dentro* da operação — base para exigir carga mínima."""
        operating = self.ngp[self.ngp.ge(threshold)]
        if operating.empty:
            return {"n": 0}
        return {"n": int(operating.size),
                "quantiles": {f"p{int(q*100):02d}": round(float(operating.quantile(q)), 2)
                              for q in (0, .01, .05, .25, .5, .95, 1)}}
