#!/usr/bin/env python3
"""EXP 32 — OS ESCORES MAIS RICOS QUE O PCA, COM A CALIBRAÇÃO REFEITA A CUSTO IGUAL, NA DECISÃO DE PRODUÇÃO (8 eventos).

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE (a hipótese do usuário: 'o PCA é simples demais para ser o melhor'). Os EXP 14 e 20 trocaram o escore de t ou de p mantendo os limiares do v2 (base x k dos níveis A e B, calibrados para a distribuição do resíduo do PCA): um escore diferente herda limiares que não são os dele, e uma queda de detecção pode ser DESENCAIXE, não falta de informação (o próprio LIMITES do EXP 14 dizia isso). O mapa de quantis iguala só a marginal global. Aqui cada alternativa é RECALIBRADA a CUSTO IGUAL: a escala dos dois limiares (A e B) do canal trocado é multiplicada por s, escolhido pelo CUSTO (as horas de alarme fora das janelas de trip ficam em 407 h, a do ponto publicado, com a tolerância de +-10%), nunca pela detecção. A referência é o escore 'fiel' (o PCA(0,95) do bundle) tratado do MESMO jeito.
O QUE SE FAZ. Escores por família, no harness fiel (RobustScaler do sklearn; retreino mensal nas últimas 20.000 amostras estáveis), agora desde 01/02/2025 para cobrir os 8 eventos: fiel, pca3, ridge, gbm, ocsvm, iforest, knn, lof, gmm, ecod; canais t e p (um por vez); n de treino 20.000 e 19.000. EWMA do v2 sobre o sinal cru, mapa de quantis para o canal original, e a decisão `ci.prever` (sinais novos de 8 eventos; INICIO = 01/02/2025). A escala s varia numa grade fixa {0,6; 0,7; 0,8; 0,9; 1,0; 1,1; 1,25; 1,4; 1,6; 1,8; 2,0; 2,5}: em cada célula toma-se o s cujas horas fora ficam mais perto de 407 h (e dentro de +-10% se existir); se nenhum fica, a célula é reportada como 'sem custo igual'.
CRITÉRIOS (fixados antes). Por escore e canal, mediana das duas variantes de n: GANHO = detecção >= a da referência + 1 a custo igual; EQUIVALENTE = a +-1 da referência; PIOR = o resto. Piso de ruído: a referência varia +-1 evento entre os dois n. A pergunta do usuário é respondida SIM só se algum escore dá GANHO em algum canal em AMBAS as variantes de n.
EXPECTATIVAS: a referência (fiel) a custo igual dá 7/8 em t e 6-7/8 em p ~70%; ao recalibrar, >= 3 dos 9 escores alternativos passam de PIOR a EQUIVALENTE em t (o desencaixe explicava parte da queda) ~55%; algum GANHO (>= ref + 1) em t ou p em ambas as variantes de n ~12%; os de resíduo (ridge, gbm) são os mais beneficiados pela recalibração ~45%; o OCSVM é o melhor das densidades ~25%.
INTERPRETAÇÃO PREVISTA. Se nenhum ganha mesmo a custo igual, a limitação não é a simplicidade do escore: o PCA por família com o máximo por sensor é, neste conjunto, o melhor de uma lista honesta, e a melhoria tem de vir de outra camada (a fusão dos canais). Se algum ganha, o próximo passo é o placebo de deslocamento e o episódio por episódio (13b) antes de qualquer conclusão.
LIMITES. 8 eventos e um único tamanho de treino por variante; o custo é medido com os rótulos (as horas fora das janelas), como no EXP 04; a escala s só mexe nos dois níveis do canal trocado (o voto, o CUSUM e os outros canais ficam); nada se adota sem sombra.

RESULTADO (16/10/2026) -- O CUSTO IGUAL SÓ FOI ALCANÇADO PARA 3 DE 40 CÉLULAS (a referência e o pca3 em t): a classificação impressa pelo script SUPERESTIMA A EQUIVALÊNCIA e não vale como veredito; a leitura correta está abaixo.
  CONTROLE (8 eventos): 7/8, 9 FP, 407 h fora. A CUSTO IGUAL (detecção | FP | horas fora | s; n = 20.000 e 19.000). Canal t: fiel 7/8, 9, 407, s = 1,0 | 5/8, 12, 424, s = 0,9; pca3 7/8, 13, 436, s = 0,7 | 6/8, 16, 474 (sem custo igual); ridge 6/8, 17, 500 e 5/8, 17, 501; gbm 3/8, 16, 909 e 5/8, 21, 859; ocsvm 4/8, 14, 475 e 4/8, 16, 513; iforest 4/8, 19, 635 e 6/8, 17, 542; knn 5/8, 14, 470 e 4/8, 17, 510; lof 4/8, 19, 744 e 4/8, 17, 665; gmm 3/8, 17, 495 e 6/8, 17, 456; ecod 5/8, 18, 663 e 7/8, 18, 648 (todos sem custo igual, exceto o pca3 em n = 20.000). Canal p: fiel 7/8, 9, 407, s = 1,0 | 7/8, 9, 406; pca3 6/8, 11, 452 (s = 2,5, o limite da grade); ridge 6/8, 9, 785 e 7/8, 9, 598; gbm 6/8, 10, 482 e 6/8, 11, 486; ocsvm 7/8, 13, 655 e 7/8, 10, 648; iforest 5/8, 13, 872 e 4/8, 14, 847; knn 6/8, 15, 1.328 e 6/8, 12, 1.226; lof 6/8, 16, 1.270 e 6/8, 13, 713; gmm 6/8, 11, 455 e 7/8, 11, 495; ecod 5/8, 9, 656 e 5/8, 10, 660 (nenhum com custo igual: as horas ficam em 450 a 1.328 h mesmo no s mais alto da grade).
  ERRO DO MEU SCRIPT: a classificação (EQUIVALENTE/PIOR) comparou só a detecção, sem exigir o custo igual que o critério pré-registrado pedia ('GANHO = detecção >= ref + 1 A CUSTO IGUAL'). Os rótulos 'EQUIVALENTE' impressos (t: pca3, ridge, iforest, ecod; p: pca3, ridge, gbm, ocsvm, knn, lof, gmm) são, quase todos, de células a 10 a 90% mais horas fora e com 3 a 10 FP a mais que a referência; não são equivalência. Nenhum escore dá GANHO.
  LEITURA. (1) A recalibração por UMA escala nos dois limiares do canal trocado NÃO consegue levar as alternativas ao custo do publicado: no canal p nenhuma o alcança nem com a escala de 2,5 (as horas ficam em 450 a 1.328 h), o que mostra que o excesso de alarme das alternativas não está só em 'limiar mal escalado': os sinais mapeados têm mais episódios persistentes que o resíduo do PCA do bundle. (2) A referência fiel varia +-2 eventos entre n = 20.000 e 19.000 em t (7/8 e 5/8 em custo parecido): o piso de ruído de detecção com 8 eventos é de 2. (3) Onde há custo comparável (o pca3 em t, n = 20.000: 7/8, 13 FP, 436 h) a alternativa tem 4 FP a mais que a referência (9). (4) O experimento NÃO resolve a pergunta do usuário porque o custo igual falhou: a grade de s é curta. Continua em 32b com a grade estendida e a classificação só das células a custo igual.
PLACAR: ACERTEI: nenhum GANHO (~88%); a referência no canal p dá 7/8 (~70%). ERREI: a referência em t é 7/8 (dá 7/8 e 5/8 conforme o n); >= 3 alternativas passam de PIOR a EQUIVALENTE em t (~55%: o rótulo impresso diz 4, mas é artefato do meu script); ridge e gbm os mais beneficiados (~45%: o gbm piora).

Uso:  PYTHONPATH=src .venv/bin/python experiments/32_escores_alternativos_a_custo_igual.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E20 = _carrega("exp20", "20_escores_por_familia_fiel.py"); E6 = E20.E6
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.walk import walk_forward

INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); ROOT = E20.ROOT; CACHE = ROOT / "escores_fiel8"; GL = {}
ESC = ("fiel", "pca3", "ridge", "gbm", "ocsvm", "iforest", "knn", "lof", "gmm", "ecod"); NS = (20000, 19000); S_GRADE = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5); ALVO = 407.0


def escore(a):
    nome, canal, n = a; f = CACHE / f"{canal}_{nome}_{n}.parquet"
    if f.exists():
        return a
    import torch  # noqa
    s = walk_forward(E20.FABRICAS[nome], GL["g"], GL["tags"][canal], GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); return a


def mede(a):
    nome, canal, n, s = a; proc, det, reg = GL["proc"], GL["det"], GL["reg"]
    sc = pd.read_parquet(CACHE / f"{canal}_{nome}_{n}.parquet")["s"]; e = sc.ewm(halflife=pd.Timedelta(det["halflife"][canal]), times=sc.index).mean().where(sc.notna()).reindex(proc.index)
    p2 = substitui(proc, canal, mapeia_quantis(e, proc[canal], proc["mask"])); d2 = copy.deepcopy(det)
    for k in ("k_nivel_a", "k_nivel_b"):
        d2[k][canal] = det[k][canal] * s
    res = decide_v2(p2, d2); al = res["is_anomaly"].astype(bool); r = avalia(al.reindex(GL["idx_c"]).fillna(False), GL["vig"], reg, inicio_periodo=INICIO8)
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, inicio=r.inicio, fpmes=r.fp_mes)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items()}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO8)
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(g=g, est=est, fim=vig_c.index[-1], tags=tags, proc=proc, det=det, reg=reg, vig=vig_c, idx_c=al_c.index)
    print(f"CONTROLE (8 eventos): {ctrl.det}/{ctrl.n_eventos} | FP {ctrl.fp} | fora {ctrl.horas_alarme_fora:.0f} h", flush=True)
    with Pool(4) as p:
        list(p.map(escore, [(nm, c, n) for c in tags for nm in ESC for n in NS], chunksize=1))
        R = dict(p.map(mede, [(nm, c, n, s) for c in tags for nm in ESC for n in NS for s in S_GRADE], chunksize=2))
    print("\nA CUSTO IGUAL (s escolhido pelas horas fora ~ 407 h; detecção | FP | h fora | s)", flush=True)
    tab = {}
    for c in tags:
        for nm in ESC:
            lin = []
            for n in NS:
                cand = sorted(S_GRADE, key=lambda s: abs(R[(nm, c, n, s)]["fora"] - ALVO)); s = cand[0]; r = R[(nm, c, n, s)]; ok = abs(r["fora"] - ALVO) <= 0.1 * ALVO
                tab[(c, nm, n)] = (r["det"], r["fp"], r["fora"], s, ok); lin.append(f"n={n}: {r['det']}/8, FP {r['fp']}, {r['fora']:.0f} h, s={s}{'' if ok else ' (sem custo igual)'}")
            print(f"   {c} <- {nm:7s} | " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO (mediana das variantes de n contra a referência 'fiel' do mesmo canal)", flush=True)
    for c in tags:
        ref = [tab[(c, "fiel", n)][0] for n in NS]; rm = float(np.median(ref)); print(f"   [{c}] referência fiel a custo igual: detecção {ref} (mediana {rm:.1f})", flush=True)
        for nm in ESC[1:]:
            d = [tab[(c, nm, n)][0] for n in NS]; m = float(np.median(d)); ganho = all(x >= y + 1 for x, y in zip(d, ref)); cl = "GANHO" if ganho else "EQUIVALENTE" if abs(m - rm) <= 1 else "PIOR"
            print(f"      {nm:7s}: detecção {d} -> {cl}", flush=True)


if __name__ == "__main__":
    main()
