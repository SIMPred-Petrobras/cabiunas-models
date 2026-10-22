"""Prioridade do alarme pela carga — a camada de NOTIFICAÇÃO dos EXP 61, 62 e 65, sem tocar no detector.

L = percentil da mediana do T5 estável numa janela contra os 30 d de operação estável anteriores ao nascimento do episódio.
ESTÁTICA (EXP 62): L na notificação ([nascimento, +2 h]) < θ -> o episódio inteiro vira atenção.
DINÂMICA (EXP 65): o episódio de carga baixa é promovido a alarme no primeiro t >= nascimento + 2 h com L em (t − 2 h, t] >= θ, mesma referência.
O episódio rebaixado continua armando o refratário do `prever` (isto é pós-processamento da série de alarme já produzida).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

THETA, REF_D, JANELA_H = 0.5, 30, 2
DOIS_MIN = pd.Timedelta(minutes=2)


def _ref(t5: pd.Series, a: pd.Timestamp, ref_d: float) -> np.ndarray:
    return np.sort(t5.loc[a - pd.Timedelta(days=ref_d):a - DOIS_MIN].dropna().to_numpy())


def carga_nascimento(t5: pd.Series, a: pd.Timestamp, ref_d: float = REF_D, janela_h: float = JANELA_H) -> float:
    """L na notificação; NaN se a janela não tem dado estável ou a referência tem < 30 amostras."""
    w = t5.loc[a:a + pd.Timedelta(hours=janela_h)].dropna(); r = _ref(t5, a, ref_d)
    if len(w) == 0 or len(r) < 30:
        return np.nan
    return float((r < float(w.median())).mean())


def promocao(t5: pd.Series, a: pd.Timestamp, b: pd.Timestamp, theta: float = THETA, ref_d: float = REF_D, janela_h: float = JANELA_H):
    """Primeiro t >= a + janela com L(t) >= θ dentro de [a, b], ou None. Usa só (t − janela, t] e a referência antes de a."""
    ref = _ref(t5, a, ref_d)
    if len(ref) < 30:
        return None
    jan = pd.Timedelta(hours=janela_h)
    med = t5.loc[a:b].rolling(jan, min_periods=1).median().loc[a + jan:]
    v = med.to_numpy(); fin = np.isfinite(v)
    L = np.searchsorted(ref, np.where(fin, v, 0.0), side="left") / len(ref)
    k = np.flatnonzero(fin & (L >= theta))
    return med.index[k[0]] if len(k) else None


def prioriza(episodios: list[tuple], t5: pd.Series, theta: float = THETA, ref_d: float = REF_D, janela_h: float = JANELA_H) -> pd.DataFrame:
    """Uma linha por episódio: L na notificação, prioridade estática, instante de promoção e prioridade dinâmica.

    `decidivel` é falso enquanto o dado não cobre a notificação (nascimento + janela): a prioridade ainda não existe."""
    fim_dado = t5.index[-1] if len(t5) else None; linhas = []
    for a, b in episodios:
        dec = fim_dado is not None and fim_dado >= a + pd.Timedelta(hours=janela_h)
        L = carga_nascimento(t5, a, ref_d, janela_h) if dec else np.nan
        baixa = bool(np.isfinite(L) and L < theta)
        t = promocao(t5, a, b, theta, ref_d, janela_h) if baixa else None
        linhas.append(dict(inicio=a, fim=b, decidivel=dec, L_notif=L,
                           estatica=("pendente" if not dec else "baixa" if baixa else "alta" if np.isfinite(L) else "indefinida"),
                           promovido_em=t, dinamica=("pendente" if not dec else "promovida" if t is not None else "baixa" if baixa else "alta" if np.isfinite(L) else "indefinida")))
    return pd.DataFrame(linhas, columns=["inicio", "fim", "decidivel", "L_notif", "estatica", "promovido_em", "dinamica"])


def aplica(alarme: pd.Series, tab: pd.DataFrame, modo: str) -> pd.Series:
    """A série de ALARME de alta prioridade: `estatica` tira o episódio de carga baixa inteiro; `dinamica` tira só até a promoção."""
    a = alarme.copy()
    for r in tab.itertuples():
        if r.estatica != "baixa":
            continue
        if modo == "dinamica" and r.promovido_em is not None and pd.notna(r.promovido_em):
            a.loc[r.inicio:r.promovido_em - DOIS_MIN] = False
        else:
            a.loc[r.inicio:r.fim] = False
    return a
