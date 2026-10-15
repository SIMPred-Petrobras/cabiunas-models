"""Nova visão (exp/nova-visao): análise e modelos do TC-33003A, com avaliação em primeiro lugar.

Camadas:
  nv.io        contrato de dados (grade de 2 min, máscara de operação, grupos de tags) e checagem de linhagem
  nv.causal    registro de trips causal e teste de vazamento do futuro
  nv.avalia    episódios, métricas de evento, IC por blocos mensais, placebo e piso de ruído
"""
__version__ = "0.1.0"
