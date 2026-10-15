"""Métricas de evento e de custo, com incerteza.

Convenções (iguais às da pesquisa, para que o controle seja comparável):
  janela de detecção  [T - 48 h, T]  (um episódio a detecta se a intersecta)
  início              o episódio NASCE dentro da janela
  banda               nasce dentro da janela com >= 4 h de antecedência (acionável)
  FP (bruto)          episódio que não intersecta nenhuma janela de trip
  FP/mês              FP por 730 h de máquina vigiada (por mês de MÁQUINA, não de calendário)
  horas de alarme     tempo em alarme; SEMPRE reportado junto (a contagem de FP sozinha é manipulável por fusão de episódios)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

PASSO_H = 2 / 60
JANELA = pd.Timedelta(hours=48)
TMIN = pd.Timedelta(hours=4)


@dataclass
class Resultado:
    n_eventos: int
    det: int
    inicio: int
    banda: int
    leads_h: list
    fp: int
    horas_vigiadas: float
    horas_alarme: float
    horas_alarme_fora: float
    por_evento: dict = field(default_factory=dict)

    @property
    def fp_mes(self) -> float:
        return self.fp / (self.horas_vigiadas / 730.0) if self.horas_vigiadas > 0 else float("nan")

    @property
    def lead_medio_h(self) -> float:
        return float(np.mean(self.leads_h)) if self.leads_h else float("nan")

    def linha(self) -> dict:
        return dict(det=f"{self.det}/{self.n_eventos}", inicio=self.inicio, banda=self.banda, lead_h=round(self.lead_medio_h, 1), fp=self.fp,
                    fp_mes=round(self.fp_mes, 3), horas_alarme=round(self.horas_alarme), alarme_fora_h=round(self.horas_alarme_fora), horas_vig=round(self.horas_vigiadas))


def episodios_de_serie(alarme: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Corridas contíguas de alarme numa série booleana de grade completa (um passo = uma linha)."""
    a = alarme.fillna(False).to_numpy(bool)
    if not a.any():
        return []
    d = np.diff(np.concatenate(([0], a.astype(np.int8), [0])))
    ini = np.flatnonzero(d == 1); fim = np.flatnonzero(d == -1) - 1
    idx = alarme.index
    return [(idx[i], idx[j]) for i, j in zip(ini, fim)]


def avalia(alarme: pd.Series, vigiado: pd.Series, trips: list[pd.Timestamp], inicio_periodo=None, fim_periodo=None) -> Resultado:
    """Métricas de um alarme (booleano na grade) contra os trips, só dentro do período (padrão: a série toda)."""
    ini = inicio_periodo or alarme.index[0]; fim = fim_periodo or alarme.index[-1]
    al = alarme.loc[ini:fim]; vg = vigiado.reindex(al.index).fillna(False)
    eps = episodios_de_serie(al)
    T = [t for t in trips if ini <= t <= fim]
    det = inic = ban = 0; leads = []; por = {}
    for t in T:
        inter = [(a, b) for a, b in eps if a <= t and b >= t - JANELA]
        nasc = [(a, b) for a, b in inter if a >= t - JANELA]
        d = bool(inter); i = bool(nasc); b_ = bool([x for x in nasc if x[0] <= t - TMIN])
        det += d; inic += i; ban += b_
        if inter:
            lead = (t - max(inter[0][0], t - JANELA)).total_seconds() / 3600
            leads.append(lead)
        por[str(t)] = dict(det=d, inicio=i, banda=b_, lead_h=(leads[-1] if inter else None))
    fp = sum(1 for a, b in eps if not any(a <= t and b >= t - JANELA for t in T))
    h_al = float(sum((b - a).total_seconds() / 3600 + PASSO_H for a, b in eps))
    h_fora = float(sum((b - a).total_seconds() / 3600 + PASSO_H for a, b in eps if not any(a <= t and b >= t - JANELA for t in T)))
    return Resultado(n_eventos=len(T), det=det, inicio=inic, banda=ban, leads_h=leads, fp=fp, horas_vigiadas=float(vg.sum()) * PASSO_H, horas_alarme=h_al, horas_alarme_fora=h_fora, por_evento=por)


def ic_blocos_mensais(alarme: pd.Series, vigiado: pd.Series, trips: list[pd.Timestamp], n=2000, semente=20261015, nivel=0.95, inicio_periodo=None, fim_periodo=None) -> dict:
    """IC por reamostragem de MESES (blocos) para FP/mês, horas de alarme por 730 h vigiadas e fração de eventos detectados."""
    ini = inicio_periodo or alarme.index[0]; fim = fim_periodo or alarme.index[-1]
    al = alarme.loc[ini:fim]; vg = vigiado.reindex(al.index).fillna(False)
    eps = episodios_de_serie(al); T = [t for t in trips if ini <= t <= fim]
    mes = lambda i: i.tz_localize(None).to_period("M") if getattr(i, "tzinfo", None) is not None or getattr(getattr(i, "tz", None), "zone", None) else i.to_period("M")
    meses = sorted(set(mes(al.index)))
    por_mes = {m: dict(h=0.0, fp=0, h_al=0.0, ev=[]) for m in meses}
    for m in meses:
        sel = mes(vg.index) == m
        por_mes[m]["h"] = float(vg[sel].sum()) * PASSO_H
    for a, b in eps:
        m = pd.Timestamp(a).tz_localize(None).to_period("M"); h = (b - a).total_seconds() / 3600 + PASSO_H
        por_mes[m]["h_al"] += h
        if not any(a <= t and b >= t - JANELA for t in T):
            por_mes[m]["fp"] += 1
    for t in T:
        m = pd.Timestamp(t).tz_localize(None).to_period("M")
        if m in por_mes:
            por_mes[m]["ev"].append(any(a <= t and b >= t - JANELA for a, b in eps))
    rng = np.random.default_rng(semente); fpm, hal, dfr = [], [], []
    for _ in range(n):
        s = [meses[i] for i in rng.integers(0, len(meses), len(meses))]
        h = sum(por_mes[m]["h"] for m in s)
        if h <= 0:
            continue
        fpm.append(sum(por_mes[m]["fp"] for m in s) / (h / 730.0)); hal.append(sum(por_mes[m]["h_al"] for m in s) / (h / 730.0))
        ev = [x for m in s for x in por_mes[m]["ev"]]
        if ev:
            dfr.append(float(np.mean(ev)))
    q = lambda v: tuple(float(x) for x in np.quantile(v, [(1 - nivel) / 2, 1 - (1 - nivel) / 2]))
    return dict(fp_mes=q(fpm), horas_alarme_730h=q(hal), frac_detectada=q(dfr) if dfr else (float("nan"), float("nan")), meses=len(meses))


def cobertura_ao_acaso(alarme: pd.Series, vigiado: pd.Series, trips: list[pd.Timestamp], n=4000, semente=20261015, inicio_periodo=None, fim_periodo=None) -> float:
    """Probabilidade de uma janela de 48 h tomada ao acaso (em máquina vigiada, longe de trips reais) conter alarme: o 'acaso' da detecção."""
    ini = inicio_periodo or alarme.index[0]; fim = fim_periodo or alarme.index[-1]
    al = alarme.loc[ini:fim]; vg = vigiado.reindex(al.index).fillna(False)
    eps = episodios_de_serie(al)
    cand = vg[vg].index
    cand = [t for t in cand if not any(abs((t - f).total_seconds()) < 7 * 86400 for f in trips) and t - JANELA >= al.index[0]]
    if not cand:
        return float("nan")
    rng = np.random.default_rng(semente); amostra = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return float(np.mean([any(a <= cand[i] and b >= cand[i] - JANELA for a, b in eps) for i in amostra]))


def p_binomial_de_deteccao(det: int, n_eventos: int, p_acaso: float) -> float:
    """p unilateral: detectar >= det de n_eventos se cada um fosse detectado com a probabilidade do acaso."""
    if n_eventos == 0 or not np.isfinite(p_acaso):
        return float("nan")
    return float(stats.binomtest(det, n_eventos, min(max(p_acaso, 1e-9), 1 - 1e-9), alternative="greater").pvalue)


def resumo_tabela(resultados: dict) -> pd.DataFrame:
    return pd.DataFrame({k: v.linha() for k, v in resultados.items()}).T
