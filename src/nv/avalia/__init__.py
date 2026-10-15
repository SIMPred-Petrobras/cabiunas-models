"""Avaliação: o que decide se um candidato vale. Ver `metricas` (eventos, FP, blocos mensais, acaso) e `controle` (o v2 causal)."""
from .metricas import (Resultado, avalia, episodios_de_serie, ic_blocos_mensais, cobertura_ao_acaso, p_binomial_de_deteccao, resumo_tabela)  # noqa: F401
from .controle import carrega_controle  # noqa: F401
