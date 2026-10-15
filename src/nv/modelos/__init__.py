"""Modelos de pontuação de anomalia (escada de baselines). Todos: fit(X de treino sem NaN) e score(X) -> um escore por linha (NaN onde a linha tem NaN)."""
from .baselines import FABRICAS, Escala  # noqa: F401
