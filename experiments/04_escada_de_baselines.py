#!/usr/bin/env python3
"""EXP 04 — FASE 2: A ESCADA DE BASELINES, A CUSTO IGUAL, COM PISO DE RUÍDO PRÓPRIO.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. Dos baselines clássicos (do univariado ao não paramétrico), algum se aproxima ou supera o controle v2 quando todos pagam o MESMO orçamento de falsos alarmes, com a mesma máscara de vigilância, a mesma política de alarme e o mesmo retreino mensal causal? E qual é o piso de ruído de cada um?

O QUE SE FAZ.
  · ESCORES (`nv.modelos`): rz_max (maior |z| robusto), pca_spe (erro de reconstrução), pca_t2 (Hotelling no subespaço), maha_mcd (Mahalanobis com covariância robusta), iforest (Isolation Forest), knn (distância média aos 10 vizinhos). 34 tags (todas menos RUNNING_A, T5_AVG_A, HSX_6240001A e PI_5134001, de outra unidade); escala robusta com piso (5% da mediana de |x|).
  · PROTOCOLO CAUSAL (`nv.walk`): cada mês é pontuado por um modelo ajustado nas últimas 20.000 amostras ESTÁVEIS anteriores ao dia 1; o escore é dividido pelo p99 dos escores do próprio treino. Meses: 03/2025 a 04/2026 (o período do controle).
  · MÁSCARA: a vigilância do controle `atual` (a mesma exposição, 7.356 h vigiadas); ELE também fica de fora: nenhum baseline vê o que o controle não vê.
  · POLÍTICA FIXA (`nv.politica`): EWMA de 1 h só sobre instantes vigiados; episódio = EWMA > limiar por >= 2 h seguidas; refratário de 72 h. Sem escalada, sem portão, sem canais.
  · CUSTO IGUAL: o limiar de cada escore é achado por bisseção para que as HORAS DE ALARME FORA DAS JANELAS DE TRIP igualem as do controle (407 h) [e 0,5x e 2x como curva]. Isto usa as janelas dos trips só para medir o custo, igual para todos; a comparação é "detecção a custo igual" e é otimista para TODOS (incluído o controle, que tem um ponto só).
  · MÉTRICAS (`nv.avalia`): detecção, início, banda, lead médio, FP, horas de alarme, cobertura ao acaso e p binomial; 7 eventos (a partir de 17/03/2025).
  · PISO DE RUÍDO PRÓPRIO: cada escore é repetido com janelas de treino de 19.000 e 21.000 amostras e, nos estocásticos (maha_mcd, iforest, knn), sementes 1 e 2; o piso é a faixa de detecção e de lead entre as variantes a custo igual. Uma vantagem sobre o controle só conta se passar do piso.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  E1  a 1x o orçamento, pelo menos 1 dos 6 baselines detecta >= 6 dos 7 eventos (o que o controle faz): ~65%.
  E2  nenhum baseline supera o controle de forma CLARA (mais detecção E mais lead, além do piso de ruído do próprio baseline): ~85%.
  E3  pca_spe, pca_t2 e maha_mcd ficam entre os 3 melhores em detecção a 1x (~55%); o rz_max é o pior ou empata com o pior (~65%).
  E4  o piso de ruído de detecção é de pelo menos 1 evento em >= 3 dos 6 escores (~65%).
  E5  o lead médio do melhor baseline fica a +-10 h do 32,7 h do controle (~60%).
  E6  a cobertura ao acaso dos baselines a 1x fica entre 0,10 e 0,30 (~70%).

LIMITES. 7 eventos; um período de 13 meses e uma política fixa e simples (o controle tem canais, voto, portões e escalada: mais sofisticado); o orçamento é calibrado com as janelas (otimista e igual para todos); baselines sem ajuste de hiperparâmetros (k, nº de árvores, nº de componentes fixados a priori); a normalização por p99 do treino repete a fragilidade do v2 (M26 da pesquisa).

RESULTADO (15/10/2026). CONTROLE (v2 atual) NO HARNESS: detecção 6/7, lead 32,7 h, 9 FP, 1.304 h de alarme (407 h fora das janelas = o ORÇAMENTO), 7.356 h vigiadas. 34 tags nos baselines. Tempo total ~3 min (modelos pequenos).
  A CUSTO IGUAL (1x = 407 h de alarme fora das janelas; detecção de 7 | início | lead h | FP episódios | cobertura ao acaso | p binomial):
    rz_max 4/7 | 3 | 27,1 | 45 | 0,341 | 0,186;  pca_spe 1/7 | 1 | 20,3 | 14 | 0,168 | 0,724;  pca_t2 3/7 | 2 | 31,2 | 47 | 0,355 | 0,480;  maha_mcd 1/7 | 1 | 9,3 | 10 | 0,116 | 0,579;  iforest 4/7 | 3 | 21,8 | 18 | 0,147 | 0,011;  knn 1/7 | 0 | 48,0 | 22 | 0,223 | 0,829.
  A 0,5x: 1 a 2 de 7 (rz_max 2, pca_t2 2, iforest 2; os demais 1). A 2x: rz_max 5/7, pca_spe 5/7, pca_t2 5/7, knn 4/7, maha_mcd 3/7, iforest 3/7.
  PISO DE RUÍDO PRÓPRIO (detecção a 1x entre as variantes: treino de 19.000/21.000 e sementes): rz_max 3 a 4; pca_spe 0 a 1; pca_t2 2 a 4; maha_mcd 1 a 1; iforest 2 a 5; knn 1 a 4; lead: rz_max 27 a 29 h, pca_t2 26 a 35 h, iforest 22 a 38 h, knn 13 a 48 h.
  LEITURA CONTRA O CONTROLE: TODOS ficam ABAIXO do controle (6/7) além do piso de ruído: o máximo de qualquer variante a 1x é 5/7 (um iforest) e a mediana dos baselines é ~2,5/7. Só o iforest tem detecção acima do acaso (p = 0,011).
  O ORÇAMENTO IGUALA HORAS, NÃO EPISÓDIOS, e isso FAVORECE OS BASELINES: o controle gasta as 407 h em 9 FP (episódios longos, a memória do CUSUM), os baselines em 10 a 47 episódios curtos (a política simples não tem memória). Igualar o NÚMERO de episódios os penalizaria mais.
  A PCA (SPE) e a Mahalanobis robusta, a base clássica do monitoramento multivariado, são as PIORES (1/7), e o máximo de |z| univariado e o Isolation Forest, as melhores (4/7): o sinal está mais na MAGNITUDE de poucas tags do que na QUEBRA DA ESTRUTURA DE CORRELAÇÃO (consistente com o M31 e a anatomia: vibração e termopares isolados).
  A VANTAGEM DO v2 (de +2 a +5 detecções a custo igual) NÃO vem de um escore mais sofisticado que estes: vem da ENGENHARIA em torno dele (quatro canais por tipo de sinal, o `vb` com referência rolante, o `sp` do mancal, o CUSUM com memória, o voto entre canais, o portão e a escalada). O próximo experimento decompõe isso (sinais do v2 sob a política simples; baselines sob a política do v2).
PLACAR: ACERTEI: nenhum baseline supera o controle (E2); o piso de ruído de detecção é de pelo menos 1 evento em >= 3 escores (E4: 4 de 6); o lead do melhor baseline a +-10 h do controle (pca_t2 31,2 h; E5). ERREI: pelo menos 1 baseline detecta >= 6/7 a custo igual (E1: o máximo é 5); PCA-SPE, T2 e Mahalanobis entre os 3 melhores (E3: SPE e Mahalanobis são os piores); o rz_max como o pior (é o melhor, junto com o iforest);
  cobertura ao acaso entre 0,10 e 0,30 (rz_max 0,34 e pca_t2 0,36 passam do limite).

Uso:  PYTHONPATH=src .venv/bin/python experiments/04_escada_de_baselines.py
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, cobertura_ao_acaso, p_binomial_de_deteccao
from nv.modelos import FABRICAS
from nv.walk import walk_forward
from nv.politica import ewma_vigiado, episodios, alarme_de, calibra_limiar

CACHE = Path(__file__).resolve().parents[1] / "data_local" / "escores"
INICIO = pd.Timestamp("2025-03-10", tz="UTC")
ESTOCASTICOS = {"maha_mcd", "iforest", "knn"}
SEM_TAGS = ("RUNNING_A", "T5_AVG_A", "HSX_6240001A", "PI_5134001")


def escore(nome, g, tags, est, fim, n_treino=20000, semente=0):
    CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / f"{nome}_{n_treino}_{semente}.parquet"
    if f.exists():
        return pd.read_parquet(f)["s"]
    t0 = time.time()
    s = walk_forward(FABRICAS[nome], g, tags, est, pd.Timestamp("2025-03-01", tz="UTC"), fim, n_treino=n_treino, semente=semente)
    s.to_frame("s").to_parquet(f); print(f"      {nome} n={n_treino} semente={semente}: {time.time() - t0:.0f} s", flush=True)
    return s


def avalia_a_custo(s, vig, reg, alvo_h):
    ini = INICIO - pd.Timedelta(days=10); s = s.loc[ini:]; vig = vig.loc[ini:]                # só o período avaliado (acelera a bisseção)
    e = ewma_vigiado(s, vig); thr = calibra_limiar(e, vig, reg, alvo_h, inicio=INICIO)
    al = alarme_de(episodios(e, thr), e.index)
    return avalia(al, vig, reg, inicio_periodo=INICIO), al, thr


def main():
    g = nvio.carrega_grade(); m = nvio.mascaras(g); est = m["estavel"].to_numpy(bool)
    tags = [c for c in g.columns if c not in SEM_TAGS]
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    vig = vig_c.reindex(g.index).fillna(False); reg = [t for t in carrega_registro() if t >= INICIO]
    ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO); alvo = ctrl.horas_alarme_fora
    print(f"CONTROLE (v2 atual): detecção {ctrl.det}/{ctrl.n_eventos}, lead {ctrl.lead_medio_h:.1f} h, FP {ctrl.fp}, horas de alarme {ctrl.horas_alarme:.0f}, FORA das janelas {alvo:.0f} h (o ORÇAMENTO), {ctrl.horas_vigiadas:.0f} h vigiadas, {len(tags)} tags nos baselines", flush=True)

    print("\nescores (retreino mensal causal)", flush=True)
    S = {n: escore(n, g, tags, est, fim) for n in FABRICAS}
    linhas = []; curva = []
    for n, s in S.items():
        for mult in (0.5, 1.0, 2.0):
            r, al, thr = avalia_a_custo(s, vig, reg, mult * alvo)
            cov = cobertura_ao_acaso(al, vig_c.reindex(al.index).fillna(False), reg, inicio_periodo=INICIO); p = p_binomial_de_deteccao(r.det, r.n_eventos, cov)
            curva.append(dict(escore=n, orcamento=f"{mult:g}x", det=f"{r.det}/{r.n_eventos}", inicio=r.inicio, banda=r.banda, lead_h=round(r.lead_medio_h, 1), fp=r.fp, horas_alarme=round(r.horas_alarme), fora_h=round(r.horas_alarme_fora), acaso=round(cov, 3), p_binom=round(p, 4)))
    C = pd.DataFrame(curva); print("\nA CUSTO IGUAL (orçamento = horas de alarme fora das janelas; 1x = o do controle)")
    print(C[C.orcamento == "1x"].to_string(index=False))
    print("\nCURVA (0,5x e 2x)"); print(C[C.orcamento != "1x"].to_string(index=False))

    print("\nPISO DE RUÍDO PRÓPRIO (a 1x o orçamento): variantes = treino 19.000/21.000 e sementes 1/2 nos estocásticos")
    piso = {}
    for n in FABRICAS:
        v = [("20000/s0", S[n])]
        for nt in (19000, 21000):
            v.append((f"{nt}/s0", escore(n, g, tags, est, fim, n_treino=nt)))
        if n in ESTOCASTICOS:
            for sem in (1, 2):
                v.append((f"20000/s{sem}", escore(n, g, tags, est, fim, semente=sem)))
        res = [avalia_a_custo(s, vig, reg, alvo)[0] for _, s in v]
        dets = [r.det for r in res]; leads = [r.lead_medio_h for r in res]; fps = [r.fp for r in res]
        piso[n] = (min(dets), max(dets), np.nanmin(leads), np.nanmax(leads))
        print(f"   {n:9s}: detecção {min(dets)} a {max(dets)} (variantes {dets}) | lead {np.nanmin(leads):.1f} a {np.nanmax(leads):.1f} h | FP {min(fps)} a {max(fps)}")

    print("\nLEITURA CONTRA O CONTROLE (detecção e lead a custo igual; 'além do piso' = o MÍNIMO do baseline supera o controle)")
    for n in FABRICAS:
        r = C[(C.escore == n) & (C.orcamento == "1x")].iloc[0]; lo, hi, ll, lh = piso[n]; d = int(r.det.split("/")[0])
        print(f"   {n:9s}: detecção {r.det} (faixa {lo} a {hi}) contra o controle {ctrl.det}/{ctrl.n_eventos} | lead {r.lead_h} h (faixa {ll:.1f} a {lh:.1f}) contra {ctrl.lead_medio_h:.1f} | "
              f"{'ACIMA do controle além do piso' if lo > ctrl.det and ll > ctrl.lead_medio_h else 'dentro ou abaixo do piso' if hi >= ctrl.det else 'ABAIXO do controle além do piso'}")


if __name__ == "__main__":
    main()
