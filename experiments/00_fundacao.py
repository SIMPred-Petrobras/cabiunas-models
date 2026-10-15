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
