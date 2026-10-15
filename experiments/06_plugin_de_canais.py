#!/usr/bin/env python3
"""EXP 06 — PLUG-IN DE CANAIS: um sinal candidato DENTRO da decisão do v2.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. Os EXP 04 e 05b mostraram que a decisão do v2 (CUSUM, níveis A/B, portão sp|vb, escalada, refratário) vale ~4 detecções por notificação. Um candidato de SINAL só mostra seu valor se for medido DENTRO dessa decisão. Este experimento (a) monta e VALIDA o plug-in (a decisão de produção, `ci.prever`, como caixa-preta, rodada no período inteiro) e
(b) troca o canal `t` ou o canal `p` por cada um dos 6 baselines do EXP 04, levados à escala do canal por mapeamento de quantis (`nv.plugin`), e mede o efeito nos DOIS custos (episódios falsos e horas de alarme).

O QUE SE FAZ.
  · `proc` = os sinais causais t, p, sp, vb, a máscara e o `reset` do EXP 05 (caminho de produção, replay semanal); `decide_v2` = `ci.prever` sobre o período inteiro de uma vez.
  · VALIDADE DO PLUG-IN (escrita antes): com os sinais ORIGINAIS a decisão reproduz o controle do harness: detecção 6/7, FP dentro de 1 de 9, e a série de alarme coincide com a do controle em >= 97% dos instantes vigiados. (O controle foi gerado em janelas semanais de 75 d e aqui roda o período de uma vez: o refratário e a escalada podem diferir nas bordas das janelas.)
  · SUBSTITUIÇÃO: para cada baseline b (rz_max, pca_spe, pca_t2, maha_mcd, iforest, knn; n = 20.000, semente 0) e cada canal c em {t, p}: o escore b do EXP 04 -> EWMA de 1 h só em vigiado -> mapeamento de quantis para a distribuição do canal c do v2 (sobre os instantes vigiados do período) -> `proc[c]` -> `ci.prever`.
  · PISO DE RUÍDO PRÓPRIO: as mesmas variantes do EXP 04 (treino de 19.000 e 21.000; sementes 1 e 2 nos estocásticos); o piso é a faixa de detecção e FP entre as variantes.
  · MÉTRICAS: as do harness, nos DOIS custos: FP em episódios e horas de alarme (totais e fora das janelas); 7 eventos.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  E1  a validade do plug-in passa (detecção 6/7, FP 9 +-1, >= 97% de coincidência): ~75%.
  E2  trocar o `t` por rz_max ou iforest mantém detecção >= 5/7 com FP <= 12 (~45%); trocar por pca_spe, maha_mcd ou knn deixa a detecção <= 4/7 (~70%).
  E3  trocar o `p` (o canal mais fraco) por qualquer baseline muda a detecção em no máximo +-1 evento (~60%).
  E4  alguma das 12 combinações (canal x baseline) chega a >= 6/7 com FP <= 9 (~20%) e, se chegar, está dentro do piso de ruído do controle.
  E5  o piso de ruído (faixa de detecção entre as variantes) é >= 1 evento em >= 50% das combinações (~70%).

LIMITES. 7 eventos; o mapa de quantis é global no período (ajuste de escala sem rótulo, mas com a marginal futura); o plug-in troca UM canal por vez (o voto e os portões ficam como estão); o controle foi calibrado para os próprios sinais, e um sinal trocado herda os limiares sem recalibração (uma queda pode ser desencaixe, não falta de informação).

RESULTADO (15/10/2026). VALIDADE DO PLUG-IN: PASSA com folga. A decisão de produção rodada sobre o período inteiro com os sinais originais dá 6/7, início 5, lead 32,7 h, 9 FP, 1.304 h (407 h fora das janelas), IDÊNTICO ao controle, e a série de alarme coincide em 100,00% dos instantes vigiados.
  SUBSTITUIÇÃO DO CANAL t (detecção de 7 | FP episódios | horas de alarme | horas fora das janelas | piso de ruído em detecção): rz_max 3 | 23 | 1.811 | 1.563 | 3-3; pca_spe 3 | 19 | 2.242 | 1.378 | 3-3; pca_t2 3 | 24 | 2.063 | 1.777 | 3-3; maha_mcd 3 | 19 | 2.177 | 1.314 | 3-3; iforest 5 | 16 | 1.859 | 955 | 4-6; knn 2 | 24 | 1.913 | 1.647 | 2-3.
  SUBSTITUIÇÃO DO CANAL p: rz_max 5 | 24 | 1.938 | 1.634 | 5-5; pca_spe 6 | 12 | 2.212 | 1.314 | 6-6; pca_t2 5 | 22 | 1.903 | 1.573 | 5-5; maha_mcd 6 | 11 | 2.011 | 1.764 | 6-6; iforest 6 | 11 | 1.544 | 1.290 | 6-6; knn 6 | 15 | 1.604 | 1.296 | 6-6.
  CONTROLE: 6/7 | 9 FP | 1.304 h | 407 h fora. NENHUMA das 12 combinações chega a >= 6/7 com FP <= 9 (0 de 12).
LEITURA. (1) O canal t do v2 (PCA das temperaturas) é INSUBSTITUÍVEL por qualquer baseline: trocá-lo derruba a detecção a 2-3 de 7 (o iforest chega a 5) e sobe o FP a 16-24. (2) O canal p é o mais fraco e tolera troca na DETECÇÃO (5 a 6 de 7), mas TODA troca custa mais: FP de 11 a 24 episódios (contra 9) e 3x a 4x as horas de alarme fora das janelas (1.290 a 1.764 h contra 407 h). Com a mesma
        distribuição marginal no normal (o mapa de quantis a impõe), os sinais do v2 têm excursões menos concentradas e menos persistentes em operação normal: a ESPECIFICIDADE é o que os canais do v2 têm a mais, e nenhum baseline a replica.
  (3) O piso de ruído dentro da decisão é BAIXO (detecção 0 a 2 de faixa; só 2 de 12 combinações têm faixa >= 1): a consolidação do v2 amortece a variação entre treinos; isso dá ao plug-in uma boa sensibilidade para comparar candidatos.
  (4) CONSEQUÊNCIA PARA A FASE 3: o critério de um sinal novo é ESPECIFICIDADE (FP em episódios e horas fora das janelas à mesma detecção) contra os canais t e p do v2 no plug-in, e não a detecção isolada. Um modelo de normal que não reduza as excursões em operação normal (deriva, regimes) não passa.
PLACAR: ACERTEI: pca_spe, maha_mcd e knn no t deixam <= 4/7 (3, 3, 2); trocar o p muda a detecção em no máximo 1 (5 a 6); nenhuma combinação chega a 6/7 com FP <= 9 (0 de 12). ERREI: previ rz_max e iforest no t mantendo >= 5/7 com FP <= 12 (rz_max 3/7; iforest 5/7 com 16 FP); previ o piso de ruído >= 1 evento em >= 50% das combinações (2 de 12).

Uso:  PYTHONPATH=src .venv/bin/python experiments/06_plugin_de_canais.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado

INICIO = E4.INICIO


def mede(res, reg):
    r = avalia(res["is_anomaly"].astype(bool), res["vigiado"].astype(bool), reg, inicio_periodo=INICIO)
    return r


def linha(r):
    return dict(det=f"{r.det}/{r.n_eventos}", inicio=r.inicio, lead_h=round(r.lead_medio_h, 1), fp=r.fp, horas=round(r.horas_alarme), fora_h=round(r.horas_alarme_fora))


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); tags = [c for c in g.columns if c not in E4.SEM_TAGS]
    al_c, vig_c = carrega_controle("atual"); reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        det = _ci().carregar_modelos(PKG / "modelos")[-1]["detector"]
    vigS = proc["mask"]; fim = vig_c.index[-1]

    print("VALIDADE DO PLUG-IN (a decisão de produção sobre o período inteiro, sinais originais)", flush=True)
    res0 = decide_v2(proc, det); r0 = mede(res0, reg)
    a_novo = res0["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False); v = vig_c
    conc = float((a_novo[v] == al_c[v]).mean())
    ok = r0.det == ctrl.det and abs(r0.fp - ctrl.fp) <= 1 and conc >= 0.97
    print(f"   plug-in: {linha(r0)} | controle: {linha(ctrl)} | coincidência da série de alarme nos instantes vigiados: {100 * conc:.2f}% -> {'PASSA' if ok else 'FALHA'}", flush=True)
    if not ok:
        print("PLUG-IN INVÁLIDO: investigar antes de qualquer substituição."); return

    vig_g = vig_c.reindex(g.index).fillna(False); linhas = []
    for canal in ("t", "p"):
        for n in E4.FABRICAS:
            variantes = [("20000/s0", 20000, 0)] + [(f"{nt}/s0", nt, 0) for nt in (19000, 21000)] + ([(f"20000/s{sm}", 20000, sm) for sm in (1, 2)] if n in E4.ESTOCASTICOS else [])
            res = []
            for rot, nt, sm in variantes:
                s = E4.escore(n, g, tags, est, fim, n_treino=nt, semente=sm)
                e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
                novo = mapeia_quantis(e, proc[canal], vigS)
                res.append((rot, mede(decide_v2(substitui(proc, canal, novo), det), reg)))
            principal = res[0][1]; dets = [r.det for _, r in res]; fps = [r.fp for _, r in res]
            linhas.append(dict(canal=canal, baseline=n, **linha(principal), det_faixa=f"{min(dets)}-{max(dets)}", fp_faixa=f"{min(fps)}-{max(fps)}"))
            print(f"   {canal} <- {n:9s}: {linha(principal)} | piso: detecção {min(dets)}-{max(dets)}, FP {min(fps)}-{max(fps)}", flush=True)
    T = pd.DataFrame(linhas)
    print(f"\nCONTROLE: {linha(ctrl)}\n"); print(T.to_string(index=False))
    ok6 = T[(T.det.str.split("/").str[0].astype(int) >= 6) & (T.fp <= 9)]
    print(f"\ncombinações com >= 6/7 e FP <= 9: {len(ok6)} de {len(T)}")


if __name__ == "__main__":
    main()
