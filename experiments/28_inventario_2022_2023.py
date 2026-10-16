#!/usr/bin/env python3
"""EXP 28 — INVENTÁRIO DOS DADOS DE 2022-2023 (qualidade, paradas e candidatos a trip), SEM rotular nada.

PRÉ-REGISTRO (15/10/2026 -> 16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. A pasta `tagsselecionadas_30s` tem xlsx de 30 s de fev/2022 a dez/2023 com as mesmas 38 tags da grade (a grade da pesquisa, e todos os experimentos até aqui, só vão de jan/2024 a abr/2026). O histórico de ordens (`Historico Ordens Turbina A.xlsx`) tem em 2022 registros como 'TC-33003A CAINDO POR ATUAÇÃO DE TAHH' (07/03/2022), 'SANAR FALHA NA PARTIDA' (17/02/2022), 'SUBSTITUIR TURBINA' (05/04/2022), 'TV-354A-A SANAR INDICAÇÃO' (29/06/2022), 'Válvula inoperante' (12/09/2022) e 'SANAR FALHA NA EGF-344' (26/12/2023). Mais trips-alvo seriam o que mais falta (7 eventos); mas o alvo é definido pelo usuário (trip = 2º estágio) e eu NÃO rotulo eventos: este experimento só descreve os dados e LISTA candidatos, para o usuário aprovar a lista exata.
O QUE SE FAZ (descritivo): a grade de 2 min de 2022-2023 (fuso validado: xlsx + 3 h = UTC; `scripts/valida_fuso_xlsx_2024.py`) para (i) cobertura por mês: linhas, fração média de NaN nas 38 tags e horas estáveis (RUNNING_A > 0,5 e T5_AVG_A > 300); (ii) as PARADAS de >= 2 h (RUNNING_A virando 0): data, duração, e a ABRUPTIDADE da chegada à parada = queda do T5 nos 10 minutos antes do instante da parada, e o tempo em marcha contínua antes; (iii) as mesmas medidas para os 9 trips do registro e as demais paradas de 2024-2026 (a referência: as paradas de trip têm abruptidade distinta das demais?); (iv) as ordens do histórico a <= 5 dias de cada parada de 2022-2023 (texto breve).
NÃO SE FAZ: rotular trip; escolher limiar de abruptidade pelo resultado; alterar o registro de trips. A saída é uma TABELA de candidatos para o usuário.
EXPECTATIVAS: a qualidade de 2022-2023 é parecida com a de 2024 (fração média de NaN <= 2% em >= 80% dos meses) ~70%; há entre 25 e 60 paradas de >= 2 h nos 23 meses ~65%; a abruptidade (queda do T5 em 10 min) NÃO separa os 9 trips das demais paradas de 2024-2026 (as paradas por utilidade/gás também são abruptas; memória 'o-alvo-esta-certo') ~70%; a parada de 07/03/2022 (TAHH) aparece com abruptidade alta e T5 alto antes ~65%; as ordens de 2022 e 2023 casam com <= 5 paradas ~60%.
INTERPRETAÇÃO PREVISTA. Se a abruptidade não separa, a lista de candidatos não pode ser feita pelos dados e depende do usuário (registro de trips de 2022-2023, ou o motivo de cada parada). Se separa, é um critério a propor, ainda sujeito à aprovação. Em qualquer caso, a avaliação do detector publicado em 2022-2023 (um teste fora do tempo e fora da amostra, com o ponto de operação intocado) só é feita com a lista aprovada.
LIMITES. A reamostragem do xlsx reproduz a grade de 2024 a 2e-4 de erro relativo mediano (não exato); a substituição da turbina em 05/04/2022 pode ser uma mudança de regime; nenhuma conclusão sobre eventos sem a lista do usuário.

Uso:  PYTHONPATH=src .venv/bin/python experiments/28_inventario_2022_2023.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro

ROOT = Path(__file__).resolve().parents[1] / "data_local"
ORDENS = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/Historico Ordens Turbina A.xlsx")


def paradas(g, min_h=2.0):
    r = (g["RUNNING_A"] > 0.5).fillna(False); ch = r.ne(r.shift()).cumsum(); linhas = []
    segs = pd.DataFrame({"lig": r.to_numpy(), "ch": ch.to_numpy()}, index=g.index).groupby("ch")
    info = [(s.lig.iloc[0], s.index[0], s.index[-1], len(s)) for _, s in segs]
    for k, (lig, ini, fim, n) in enumerate(info):
        if lig or n * 2 / 60 < min_h or k == 0:
            continue
        # o instante da parada é o 1.º instante com RUNNING_A = 0; a marcha anterior é info[k-1]
        marcha_h = info[k - 1][3] * 2 / 60; t5 = g["T5_AVG_A"]; a = t5.loc[ini - pd.Timedelta(minutes=10):ini]
        queda = float(a.iloc[0] - a.iloc[-1]) if len(a) > 1 and a.notna().all() else np.nan
        linhas.append(dict(parada=ini, duracao_h=round(n * 2 / 60, 1), marcha_anterior_h=round(marcha_h, 1), T5_antes=round(float(t5.loc[ini - pd.Timedelta(minutes=10)]), 0) if (ini - pd.Timedelta(minutes=10)) in t5.index else np.nan, queda_T5_10min=round(queda, 0)))
    return pd.DataFrame(linhas)


def main():
    g22 = pd.read_parquet(ROOT / "grade_2022_2023.parquet"); g24 = nvio.carrega_grade(); est = lambda g: ((g["RUNNING_A"] > 0.5) & (g["T5_AVG_A"] > 300))
    print(f"grade 2022-2023: {g22.shape}, {g22.index[0]} a {g22.index[-1]}", flush=True)
    print("\n(i) COBERTURA POR MÊS (2022-2023 e, para comparar, 2024)", flush=True)
    for nome, g in (("2022-2023", g22), ("2024", g24.loc["2024"])):
        c = pd.DataFrame({"linhas": g.iloc[:, 0].groupby(g.index.to_period("M")).size(), "nan_medio": g.isna().mean(axis=1).groupby(g.index.to_period("M")).mean().round(3), "horas_estaveis": est(g).groupby(g.index.to_period("M")).sum().mul(2 / 60).round(0)})
        print(f"   {nome}:", flush=True); print(c.to_string().replace("\n", "\n      "), flush=True)
    P22 = paradas(g22); P24 = paradas(g24)
    print(f"\n(ii) PARADAS >= 2 h em 2022-2023: {len(P22)} | em 2024-2026: {len(P24)}", flush=True)
    reg = carrega_registro(); P24["trip"] = [any(abs((p - t).total_seconds()) <= 4 * 3600 for t in reg) for p in P24.parada]
    print("\n(iii) ABRUPTIDADE (queda do T5 nos 10 min antes da parada) por tipo, 2024-2026:", flush=True)
    print(P24.groupby("trip").queda_T5_10min.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(0).to_string(), flush=True)
    print("   trips do registro:", P24[P24.trip][["parada", "duracao_h", "marcha_anterior_h", "T5_antes", "queda_T5_10min"]].to_string(index=False), flush=True)
    from sklearn.metrics import roc_auc_score
    v = P24.dropna(subset=["queda_T5_10min"]); print(f"   AUC da queda do T5 (10 min) para separar trip de não-trip nas {len(v)} paradas de 2024-2026: {roc_auc_score(v.trip, v.queda_T5_10min):.2f}; do tempo em marcha anterior: {roc_auc_score(v.trip, v.marcha_anterior_h):.2f}", flush=True)
    d = pd.read_excel(ORDENS); d["ini"] = pd.to_datetime(d["Data-base iníc."], errors="coerce")
    P22["ordens_a_5_dias"] = [" | ".join(f"{r.ini:%Y-%m-%d} {str(r['Texto breve'])[:40]}" for _, r in d[(d.ini >= p - pd.Timedelta(days=5)) & (d.ini <= p + pd.Timedelta(days=5))].iterrows()) for p in P22.parada]
    print("\n(iv) PARADAS DE 2022-2023 (candidatos NÃO rotulados; a lista exata é do usuário):", flush=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); print(P22.to_string(index=False), flush=True)
    P22.to_csv(ROOT / "exp28_paradas_2022_2023.csv", index=False)


if __name__ == "__main__":
    main()
