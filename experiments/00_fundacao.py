#!/usr/bin/env python3
"""EXP 00 — FUNDAÇÃO (Gate G0 da nova frente).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. O harness novo (`nv.avalia`) e o contrato de dados (`nv.io`) são confiáveis o bastante para julgar qualquer modelo? Isto NÃO é um teste de modelo.

O QUE SE FAZ.
  1. LINHAGEM: a grade de 2 min derivada pela pesquisa é comparada com o CSV bruto de 30 s reamostrado por ESTE repositório (2025-01-01 a 2026-04-21), por tag, e se procura desalinhamento de 1 passo.
  2. CONTROLE: o detector v2 da pesquisa, rodado de forma causal (replay semanal do protocolo da sombra, registro de trips crescendo), entra no harness como caixa-preta (série de alarme + vigilância) nas três variantes (atual, G168, B800).
     Os números do harness têm de reproduzir os conhecidos da pesquisa (7 eventos a partir de 17/03/2025): atual detecção 6/7 e 9 episódios fora das janelas; G168 7/7 e 13; B800 7/7 e 12.
  3. ACASO E INCERTEZA: cobertura ao acaso (probabilidade de uma janela de 48 h tomada ao acaso conter alarme), p binomial unilateral da detecção contra ela, e IC por blocos mensais de FP/mês, horas de alarme e fração detectada.
  4. AUDITORIA DE VAZAMENTO: as funções de atributos do carregador (`mascaras`) passam `auditoria_do_futuro`.
  5. REFERÊNCIA DE RUÍDO (externa, da pesquisa, M37): perturbações triviais mudam a detecção mediana em +-0,5, o FP/mês em +-0,1 e a carga em +-20%. Um candidato só conta se passar disso E do acaso.

CRITÉRIOS DO GATE G0 (escritos antes):
  G0-a  linhagem: a mediana (entre tags) da diferença relativa mediana é <= 1e-3 no deslocamento 0, o melhor deslocamento é 0, e >= 90% das tags ficam <= 1e-3;
  G0-b  o harness reproduz o controle: detecção igual e episódios fora das janelas a menos de 1 de distância, nas três variantes;
  G0-c  a auditoria do futuro passa em `mascaras`;
  G0-d  os testes unitários do harness passam (feito fora deste script, `pytest`).
EXPECTATIVAS REGISTRADAS: G0-a passa com ~70% (a grade da pesquisa pode ter preenchimento/interpolação que o bruto não tem; se falhar, a causa se investiga antes de usar a grade); G0-b passa com ~75% (o harness define episódio por corrida de alarme na série, a pesquisa por episódio do detector;
  1 de diferença é o esperado); G0-c passa (por construção). A cobertura ao acaso do controle fica entre 0,15 e 0,45 e o p binomial do `atual` (6/7) é < 0,05 com ~55%.

RESULTADO (15/10/2026) -- GATE G0 ABERTO (G0-a, b, c e d passam).
  G0-a LINHAGEM: a grade da pesquisa é IDÊNTICA ao CSV bruto reamostrado aqui (38 tags comuns, 2025-01-01 a 2026-04-30): diferença relativa mediana 0,0 no deslocamento 0 (0,00072 nos deslocamentos de +-1 passo), melhor deslocamento 0, 100% das tags <= 1e-3. Não há transformação oculta no artefato derivado; a grade é aceita como fonte (jan/2024 a dez/2024 não tem bruto de 2 min aqui; seguem os .xlsx de 30 s em `tagsselecionadas_30s`).
  G0-b O HARNESS REPRODUZ O CONTROLE EXATAMENTE (7 eventos a partir de 17/03/2025; 7.356 h vigiadas = 10,1 meses de máquina): atual detecção 6/7 e 9 FP; G168 7/7 e 13; B800 7/7 e 12.
        atual: início 5, banda 5, lead médio 32,7 h, FP/mês 0,893, 1.304 h de alarme (17,7% do tempo vigiado), 407 h fora das janelas; G168: 6, 6, 29,3 h, 1,29, 1.772 h, 869 h; B800: 6, 6, 25,8 h, 1,191, 1.552 h, 660 h.
  G0-c AUDITORIA DO FUTURO: `mascaras` passa (diferença máxima 0,0 nos dois cortes). A auditoria foi validada nos testes: pega janela centrada e normalização global e deixa passar EWMA causal.
  G0-d: 11 testes unitários passam (episódios, detecção/lead/banda/FP, alarme nascido antes da janela, IC por blocos, acaso, binomial, auditoria do futuro, trips conhecidos).
  ACASO E INCERTEZA (primeiras medidas da nova frente): cobertura ao acaso 0,169 (atual), 0,259 (G168), 0,212 (B800): o p binomial de detecção é < 1e-3 nas três; IC95% por blocos mensais (14 meses) do FP/mês: atual [0,32; 1,54], G168 [0,63; 2,03], B800 [0,54; 1,91]; horas de alarme por 730 h: atual [26; 273];
        fração detectada: atual [0,50; 1,00]. Leitura: com 14 meses, o FP/mês do controle é conhecido só a +-0,6; "0,893 contra 1,291" NÃO se distingue por este IC (os intervalos se sobrepõem muito).
  AMBIENTE: venv isolado com PyTorch 2.14 (CPU), LightGBM 4.7, statsmodels 0,15, Optuna 5, ruptures, scikit-learn 1.9 e pandas 3.0.2 (as versões da pesquisa para os dados). O teste do PyTorch na fila remota (Cica) fica para depois.
PLACAR: ACERTEI: G0-b, G0-c, cobertura ao acaso entre 0,15 e 0,45 (0,17 a 0,26) e p binomial do atual < 0,05. ERREI: G0-a, que previ com ~70% de chance de passar com folga, passou com diferença ZERO (a grade derivada é exatamente o bruto reamostrado); não previ o tamanho do IC mensal (+-0,6 FP/mês).

Uso:  PYTHONPATH=src .venv/bin/python experiments/00_fundacao.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro, auditoria_do_futuro
from nv.avalia import avalia, carrega_controle, cobertura_ao_acaso, ic_blocos_mensais, p_binomial_de_deteccao, resumo_tabela

INICIO = pd.Timestamp("2025-03-10", tz="UTC")           # eventos a partir de 17/03/2025 (o de 27/02 fica sem aquecimento no controle)
REF = {"atual": (6, 9), "G168": (7, 13), "B800": (7, 12)}


def main():
    print("1. LINHAGEM (grade da pesquisa contra o CSV bruto reamostrado aqui)", flush=True)
    g = nvio.carrega_grade(); b = nvio.carrega_csv_bruto_2min()
    L = nvio.verifica_linhagem(g, b)
    desloc = {k: round(v, 6) for k, v in L["mediana_por_deslocamento"].items()}
    print(f"   tags comuns {L['tags_comuns']} | período {L['periodo'][0]:%Y-%m-%d} a {L['periodo'][1]:%Y-%m-%d} | mediana da diferença relativa por deslocamento: {desloc}")
    print(f"   melhor deslocamento {L['melhor_deslocamento']} | fração de tags com diferença <= 1e-3: {L['frac_tags_ate_1e3']:.2f} | piores: { {k: round(v, 4) for k, v in L['piores_tags'].items()} }")
    a_ok = L["mediana_por_deslocamento"][0] <= 1e-3 and L["melhor_deslocamento"] == 0 and L["frac_tags_ate_1e3"] >= 0.90
    print(f"   G0-a: {'PASSA' if a_ok else 'FALHA'}", flush=True)

    print("\n2. CONTROLE PELO HARNESS (7 eventos a partir de 17/03/2025)")
    reg = [t for t in carrega_registro() if t >= INICIO]
    res = {}; b_ok = True
    for v, (det_ref, fp_ref) in REF.items():
        al, vig = carrega_controle(v); r = avalia(al, vig, reg, inicio_periodo=INICIO); res[v] = r
        ok = r.det == det_ref and abs(r.fp - fp_ref) <= 1; b_ok &= ok
        print(f"   {v:6s}: detecção {r.det}/{r.n_eventos} (pesquisa {det_ref}), FP {r.fp} (pesquisa {fp_ref}) -> {'ok' if ok else 'DIFERE'}")
    print("\n" + resumo_tabela(res).to_string())
    print(f"   G0-b: {'PASSA' if b_ok else 'FALHA'}", flush=True)

    print("\n3. ACASO E INCERTEZA (por blocos mensais)")
    for v, r in res.items():
        al, vig = carrega_controle(v)
        p = cobertura_ao_acaso(al, vig, reg, inicio_periodo=INICIO); pv = p_binomial_de_deteccao(r.det, r.n_eventos, p)
        ic = ic_blocos_mensais(al, vig, reg, inicio_periodo=INICIO)
        print(f"   {v:6s}: cobertura ao acaso {p:.3f} | p binomial {pv:.3f} | IC95% FP/mês [{ic['fp_mes'][0]:.2f}; {ic['fp_mes'][1]:.2f}] | horas de alarme por 730 h [{ic['horas_alarme_730h'][0]:.0f}; {ic['horas_alarme_730h'][1]:.0f}] | fração detectada [{ic['frac_detectada'][0]:.2f}; {ic['frac_detectada'][1]:.2f}] ({ic['meses']} meses)")

    print("\n4. AUDITORIA DO FUTURO (carregador)")
    amostra = g.loc["2025-03-01":"2025-04-30", [c for c in ("RUNNING_A", "T5_AVG_A") if c in g.columns]]
    cortes = [amostra.index[2000], amostra.index[5000]]
    A = auditoria_do_futuro(lambda d: nvio.mascaras(d).astype(float), amostra, cortes)
    print(f"   `mascaras` passa a auditoria do futuro: {A['passa']} (diferença máxima por corte: { {k: v['dif_max'] for k, v in A['por_corte'].items()} })")
    print(f"\nGATE G0: {'ABERTO (a, b e c passam; d: rodar pytest)' if a_ok and b_ok and A['passa'] else 'FECHADO: investigar antes de qualquer modelo'}")


if __name__ == "__main__":
    main()
