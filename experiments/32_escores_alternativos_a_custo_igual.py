#!/usr/bin/env python3
"""EXP 32 — OS ESCORES MAIS RICOS QUE O PCA, COM A CALIBRAÇÃO REFEITA A CUSTO IGUAL, NA DECISÃO DE PRODUÇÃO (8 eventos).

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE (a hipótese do usuário: 'o PCA é simples demais para ser o melhor'). Os EXP 14 e 20 trocaram o escore de t ou de p mantendo os limiares do v2 (base x k dos níveis A e B, calibrados para a distribuição do resíduo do PCA): um escore diferente herda limiares que não são os dele, e uma queda de detecção pode ser DESENCAIXE, não falta de informação (o próprio LIMITES do EXP 14 dizia isso). O mapa de quantis iguala só a marginal global. Aqui cada alternativa é RECALIBRADA a CUSTO IGUAL: a escala dos dois limiares (A e B) do canal trocado é multiplicada por s, escolhido pelo CUSTO (as horas de alarme fora das janelas de trip ficam em 407 h, a do ponto publicado, com a tolerância de +-10%), nunca pela detecção. A referência é o escore 'fiel' (o PCA(0,95) do bundle) tratado do MESMO jeito.
O QUE SE FAZ. Escores por família, no harness fiel (RobustScaler do sklearn; retreino mensal nas últimas 20.000 amostras estáveis), agora desde 01/02/2025 para cobrir os 8 eventos: fiel, pca3, ridge, gbm, ocsvm, iforest, knn, lof, gmm, ecod; canais t e p (um por vez); n de treino 20.000 e 19.000. EWMA do v2 sobre o sinal cru, mapa de quantis para o canal original, e a decisão `ci.prever` (sinais novos de 8 eventos; INICIO = 01/02/2025). A escala s varia numa grade fixa {0,6; 0,7; 0,8; 0,9; 1,0; 1,1; 1,25; 1,4; 1,6; 1,8; 2,0; 2,5}: em cada célula toma-se o s cujas horas fora ficam mais perto de 407 h (e dentro de +-10% se existir); se nenhum fica, a célula é reportada como 'sem custo igual'.
CRITÉRIOS (fixados antes). Por escore e canal, mediana das duas variantes de n: GANHO = detecção >= a da referência + 1 a custo igual; EQUIVALENTE = a +-1 da referência; PIOR = o resto. Piso de ruído: a referência varia +-1 evento entre os dois n. A pergunta do usuário é respondida SIM só se algum escore dá GANHO em algum canal em AMBAS as variantes de n.
EXPECTATIVAS: a referência (fiel) a custo igual dá 7/8 em t e 6-7/8 em p ~70%; ao recalibrar, >= 3 dos 9 escores alternativos passam de PIOR a EQUIVALENTE em t (o desencaixe explicava parte da queda) ~55%; algum GANHO (>= ref + 1) em t ou p em ambas as variantes de n ~12%; os de resíduo (ridge, gbm) são os mais beneficiados pela recalibração ~45%; o OCSVM é o melhor das densidades ~25%.
INTERPRETAÇÃO PREVISTA. Se nenhum ganha mesmo a custo igual, a limitação não é a simplicidade do escore: o PCA por família com o máximo por sensor é, neste conjunto, o melhor de uma lista honesta, e a melhoria tem de vir de outra camada (a fusão dos canais). Se algum ganha, o próximo passo é o placebo de deslocamento e o episódio por episódio (13b) antes de qualquer conclusão.
LIMITES. 8 eventos e um único tamanho de treino por variante; o custo é medido com os rótulos (as horas fora das janelas), como no EXP 04; a escala s só mexe nos dois níveis do canal trocado (o voto, o CUSUM e os outros canais ficam); nada se adota sem sombra.

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
