#!/usr/bin/env python3
"""EXP 23 — O QUINTO CANAL NO CAMINHO EXATO DE PRODUÇÃO: a recuperação do 04/11 persiste? a que custo, e os FP novos são de borda?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO. Nos EXP 09, 11, 15 e 20 um quinto canal (AE, ridge, GMM, kNN, pca3...) levou o detector a 7/7 recuperando o evento de 04/11/2025 (um alarme que abre na borda do blackout e é sustentado pelo voto extra: EXP 10b), a +2 a +4 FP. Esses números usaram o EWMA só sobre os instantes vigiados, que (EXP 21) soma ~+2 FP em t contra o EWMA do v2 (sobre o sinal cru). A meta do usuário é FP no nível do controle (<= ~9-10) com 1 a 2 detecções a menos: um quinto canal que recupere o 04/11 a custo de +0 a +2 FP seria útil; a +4, não.
O QUE SE FAZ. Cada candidato é um escore bruto por família do harness fiel (RobustScaler do sklearn; os escores em cache do EXP 20, n de treino 19.000, 20.000 e 21.000 e as sementes dos estocásticos), levado ao canal do v2 por: EWMA do v2 sobre o sinal cru (`ewm(halflife=1 h, times)`, `.where(notna)`) e mapa de quantis para a escala do t (os escores não estão em unidades de base x k). Entra como QUINTO canal com a base e os k do t, voto V1 (3, 2), `ci.prever` como caixa-preta. Candidatos: ridge, gmm, knn, pca3 (t) e UM PCA(0,95) fiel sobre as 26 tags de t e p juntas ('joint'; o escore global que ainda não rodei no harness fiel). Referências: o controle (6/7, 9 FP, 407 h) e o EXP 20 Q (EWMA vigiado).
MEDIDAS por candidato: detecção em cada variante (e QUAL evento muda: o 04/11/2025 é o previsto), FP, horas fora, horas totais; e, olhando os episódios do quinto canal contra os do controle, quantos episódios NASCEM, quantos deles são FP e quantos desses FP nascem na BORDA (a menos de 1 h de uma abertura da máscara de vigilância).
CRITÉRIOS (fixados antes). PERSISTE = detecção 7/7 em >= 2 das 3 variantes de n e o evento ganho é o 04/11/2025. BARATO = mediana do FP <= 11 (<= +2 sobre o controle) e horas fora <= 550. MELHOR (o do EXP 11) = 7/7, FP <= 10 e <= 450 h. 'Os FP novos são de borda' = >= 60% dos FP novos nascem a < 1 h de uma abertura da máscara.
EXPECTATIVAS: PERSISTE para ridge, gmm e knn ~60% (cada um); BARATO em algum candidato ~45%; algum MELHOR ~10%; >= 60% dos FP novos de borda ~55%; o 'joint' (global) PERSISTE ~25% (o escore global dilui por família).
INTERPRETAÇÃO PREVISTA. Se persiste e é BARATO, o quinto canal é hipótese prospectiva para o sombra.py (produção intocada). Se persiste mas custa +3 ou mais FP, o ganho é de borda e se comunica como tal. Se não persiste no caminho exato, a recuperação do 04/11 era artefato do EWMA vigiado.
LIMITES. 7 eventos; a recuperação depende de UM evento (04/11/2025), que cai em qualquer regra de borda; mapa de quantis global; nada se adota sem sombra.

RESULTADO (15/10/2026) -- A RECUPERAÇÃO DO 04/11 NÃO PERSISTE NO CAMINHO EXATO: ERA UM EFEITO DO EWMA VIGIADO DO QUINTO CANAL; o quinto canal fecha.
  Pelo caminho exato (EWMA do v2 sobre o sinal cru, mapa de quantis, voto V1), detecção | FP | h fora (variantes de n e sementes): ridge 6/7 | 11 | 496 h nas 3 variantes | gmm 6/7 | 10 a 13 (mediana 11) | 454 a 501 h | knn 6/7 | 12 a 13 | 474 a 499 h | pca3 6/7 | 12 a 13 | 453 a 461 h | joint (PCA 0,95 fiel sobre as 26 tags) 6/7 | 14 a 19 | 893 a 1.376 h. NENHUM evento a mais em 18 das 19 execuções; só o joint com n = 21.000 ganha o 04/11 (a 19 FP e 1.376 h). Os episódios novos são todos FP e, de 68 FP novos no total, só 9 (joint) e 1 (gmm) nascem na borda: os FP novos NÃO são de borda.
  CRITÉRIOS: PERSISTE falso para os cinco candidatos; BARATO verdadeiro para ridge (FP 11, 496 h) e gmm (FP 11, 458 h), mas sem ganhar nenhum evento: custa +2 FP (e +50 a +90 h fora) por nada; nenhum MELHOR.
LEITURA. (1) Os 7/7 dos EXP 09, 11, 15 e 20 (quinto canal) eram um efeito do EWMA só sobre os instantes vigiados do quinto canal: no EXP 20 o escore 'fiel' (um DUPLICADO do canal t, mesma informação) como quinto canal já dava 7/7 com 13 FP; com o EWMA do v2 ele deixa de recuperar o 04/11. A informação nova dos escores (AE, ridge, GMM, kNN...) não era a causa. (2) Isto corrige a leitura do EXP 10b (o 04/11 como alarme sustentado pelo voto extra): o canal extra sustentava o voto porque o seu EWMA recomeçava limpo na abertura da máscara, enquanto os EWMA de t e p do v2 atravessam o religamento carregando o pico. (3) O que sobra do EWMA: o EWMA vigiado muda a borda (EXP 21: +2 FP em t sem ganhar detecção como canal único); como quinto canal ele ganhava 1 evento a +2 a +4 FP, e é o mesmo trade que a memória 'o-ewma-come-a-partida' descreve em outra régua: ELE troca FP por um evento de borda e não é informação. (4) O quinto canal, com qualquer escore, está fechado como linha de melhoria.
PLACAR: ACERTEI: algum MELHOR (~10%: nenhum); BARATO em algum candidato (~45%: ridge e gmm, mas sem benefício). ERREI: PERSISTE para ridge, gmm e knn (~60% cada: nenhum); >= 60% dos FP novos de borda (~55%: 10 de 68); o joint persiste (~25%: 1 de 3 variantes, a 19 FP).

Uso:  PYTHONPATH=src .venv/bin/python experiments/23_quinto_canal_no_caminho_exato.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E20 = _carrega("exp20", "20_escores_por_familia_fiel.py"); E6 = E20.E6; E5 = E20.E5
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis
from nv.walk import walk_forward

INICIO = E20.INICIO; ROOT = E20.ROOT; JAN = pd.Timedelta(hours=48)
CANDS = ("ridge", "gmm", "knn", "pca3", "joint")


def eps_de(res):
    return episodios_de_serie(res["is_anomaly"].astype(bool).loc[INICIO:])


def main():
    g = E20.nvio.carrega_grade(); est = E20.nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]
    base = decide_v2(proc, det); eps0 = eps_de(base); sobe = proc.index[(proc["mask"] & ~proc["mask"].shift(fill_value=False)).to_numpy()]
    print(f"CONTROLE: {E6.linha(ctrl)} | episódios do controle {len(eps0)}", flush=True)
    E20.GL.update(g=g, est=est, fim=fim, tags=tags)
    juntas = tags["t"] + tags["p"]
    def bruto(nome, n, sem):
        if nome != "joint":
            f = ROOT / "escores_fiel" / f"t_{nome}_{n}_{sem}.parquet"; return pd.read_parquet(f)["s"]
        f = ROOT / "escores_fiel" / f"joint_fiel_{n}_{sem}.parquet"
        if f.exists():
            return pd.read_parquet(f)["s"]
        s = walk_forward(E20.FABRICAS["fiel"], g, juntas, est, pd.Timestamp("2025-03-01", tz="UTC"), fim, n_treino=n, semente=sem); s.to_frame("s").to_parquet(f); return s

    def quinto(s):
        e = s.ewm(halflife=pd.Timedelta(det["halflife"]["t"]), times=s.index).mean().where(s.notna()).reindex(proc.index)
        d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
        for k in ("base", "k_nivel_a", "k_nivel_b"):
            d2[k]["ae"] = det[k]["t"]
        d2["voto_nivel_a"], d2["voto_nivel_b"] = 3, 2; p2 = proc.copy(); p2["ae"] = mapeia_quantis(e, proc["t"], vigS); return decide_v2(p2, d2)
    print("\nQUINTO CANAL pelo caminho exato (EWMA cru + mapa; voto V1)", flush=True)
    for nome in CANDS:
        variantes = [(20000, 0), (19000, 0), (21000, 0)] + ([(20000, 1), (20000, 2)] if nome in E20.ESTOC else [])
        ms = []
        for (n, sem) in variantes:
            res = quinto(bruto(nome, n, sem)); r = E6.mede(res, reg); ev = [k[:10] for k, x in r.por_evento.items() if x["det"] and not ctrl.por_evento.get(k, {}).get("det")]
            E = eps_de(res); novos = [(a, b) for a, b in E if not any(a <= d and b >= c for c, d in eps0)]
            fpn = [(a, b) for a, b in novos if not any(a <= t and b >= t - JAN for t in reg)]; borda = [x for x in fpn if any(0 <= (x[0] - r0).total_seconds() / 3600 < 1 for r0 in sobe)]
            ms.append(dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, ganho=ev, novos=len(novos), fpn=len(fpn), borda=len(borda)))
            print(f"   {nome:6s} n={n} s={sem}: {r.det}/7 | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | evento ganho {ev} | episódios novos {len(novos)}, FP novos {len(fpn)}, de borda {len(borda)}", flush=True)
        d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]; persiste = sum(x >= 7 for x in d[:3]) >= 2 and all("2025-11-04" in m["ganho"] for m in ms if m["det"] >= 7)
        barato = np.median(f) <= 11 and np.median(o) <= 550; melhor = np.median(d) >= 7 and np.median(f) <= 10 and np.median(o) <= 450
        fpn = sum(m["fpn"] for m in ms); bd = sum(m["borda"] for m in ms)
        print(f"   -> {nome}: detecção {np.median(d):.0f}/7 ({min(d)}-{max(d)}) | FP {np.median(f):.0f} ({min(f)}-{max(f)}) | fora {np.median(o):.0f} h | PERSISTE {persiste} | BARATO {barato} | MELHOR {melhor} | FP novos de borda {bd} de {fpn}", flush=True)


if __name__ == "__main__":
    main()
