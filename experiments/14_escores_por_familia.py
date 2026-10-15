#!/usr/bin/env python3
"""EXP 14 — TRILHA B: ESCORES POR FAMÍLIA DE TAGS, residual por sensor contra densidade conjunta (OCSVM, GMM, LOF, iForest, kNN, ECOD, ridge, GBM) DENTRO da decisão do v2.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE (o raciocínio, depois de ler `cabiunas_inference._recon_pca`). O EXP 04/06 concluiu que "nenhum baseline substitui o canal t". Mas aqueles baselines (a) rodaram sobre TODAS as tags juntas e (b) agregaram por soma/distância. O canal t do v2 é outra coisa: o PCA só das 14 temperaturas (o bundle atual tem k = 1 componente: o "modo comum" de carga/regime) e o escore é o MÁXIMO, entre os sensores, do resíduo quadrado normalizado pelo p99 do próprio sensor (a agregação `ScorerMax`: "uma falha em um sensor só some na média de catorze"). Duas hipóteses confundidas e nunca separadas:
  (H8) o que vale é a ESTRUTURA (família de tags + resíduo por sensor + máximo normalizado), e o modelo de base é quase intercambiável entre os de resíduo (PCA, ridge, GBM), enquanto os de densidade conjunta (OCSVM, GMM, LOF, iForest, kNN, ECOD), mesmo por família, perdem por não terem a agregação por sensor;
  (H9) um modelo de resíduo mais flexível (GBM, não linear) acrescenta especificidade ao PCA de k = 1.
DESENHO. Cada escore é calculado POR FAMÍLIA (temperatura: 14 tags; pressão: 12 tags, as do bundle atual), com retreino mensal causal nas últimas 20.000 amostras estáveis (o protocolo do EXP 04), EWMA de 1 h vigiado, mapa de quantis para a distribuição do canal original do v2 (o EXP 06) e `ci.prever` como caixa-preta: troca-se UM canal por vez (t pela escala da temperatura; p pela da pressão). Escores de resíduo: pca1, pca3, ridge (cada tag nas outras), gbm (LightGBM por tag); de densidade: ocsvm (RBF, nu = 0,02), gmm (6 componentes), lof (20 vizinhos), iforest, knn (10), ecod (implementação própria simples). Piso de ruído por escore: treino de 19.000 e 21.000 e, nos estocásticos, sementes 1 e 2.
VALIDADE (a escrever antes): pca1 em t deve reproduzir o controle (detecção >= 5/7 com FP <= 12); se não, o harness por família está descasado do bundle (que treina em janelas próprias) e nada abaixo se lê como veredito.
CRITÉRIOS (os do EXP 06/09, mediana das variantes de cada célula): DIGNO = detecção >= 6/7, FP <= 9 e horas fora <= 407 h; PROMISSOR = detecção >= 5/7, FP <= 12 e horas fora <= 800 h; senão "-". Controle: 6/7 | 9 FP | 407 h fora (1.304 h totais).
EXPECTATIVAS: validade passa ~75%. H8: ridge e gbm em t ficam a +-1 detecção e +-2 FP do pca1 ~55%; TODAS as seis de densidade conjunta em t ficam >= 2 detecções abaixo do pca1 ~65%; ocsvm é a melhor das de densidade ~25% (não há razão para o kernel ganhar sem a agregação por sensor). H9: gbm melhora o pca1 em t (FP -2 ou horas fora -20% com detecção igual) ~20%. No canal p: algum escore DIGNO ~12%; algum PROMISSOR além do pca1 ~45%. Algum DIGNO em qualquer canal ~15%.
INTERPRETAÇÃO PREVISTA. Se H8 vale, o caminho de melhoria é a ESTRUTURA do canal (que famílias, que agregação), não o modelo, e o próximo passo é variar a agregação (máximo, 2º maior, soma truncada) no canal de resíduo. Se H9 vale, entra um GBM por sensor na família de temperatura como substituto, a confirmar com placebo e com o EXP de episódio por episódio (13b). Se as densidades empatam com o resíduo, o v2 não tem vantagem de estrutura e o gargalo é só informação.
LIMITES. 7 eventos; o mapa de quantis é global no período; o harness retreina nas 20.000 últimas amostras do próprio CSV (o bundle usa janelas de datas próprias); um canal por vez, sem recalibrar os limiares do v2 (queda pode ser desencaixe); nenhuma célula se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/14_escores_por_familia.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos.familias import FAMILIA_FABRICAS, ESTOCASTICOS_F
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import walk_forward

INICIO = E4.INICIO; CACHE = Path(__file__).resolve().parents[1] / "data_local" / "escores_fam"; GL = {}
FAMILIAS = {"t": "temperatura", "p": "pressao"}


def tarefa_escore(a):
    nome, canal, n_treino, sem = a; f = CACHE / f"{canal}_{nome}_{n_treino}_{sem}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    t0 = time.time(); g, est, fim = GL["g"], GL["est"], GL["fim"]
    s = walk_forward(FAMILIA_FABRICAS[nome], g, GL["tags"][canal], est, pd.Timestamp("2025-03-01", tz="UTC"), fim, n_treino=n_treino, semente=sem)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} {nome} n={n_treino} s={sem}: {time.time() - t0:.0f} s", flush=True)
    return a, s


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, fim=fim, tags=tags)
    pedidos = []
    for canal in FAMILIAS:
        for nome in FAMILIA_FABRICAS:
            pedidos += [(nome, canal, 20000, 0), (nome, canal, 19000, 0), (nome, canal, 21000, 0)] + ([(nome, canal, 20000, s) for s in (1, 2)] if nome in ESTOCASTICOS_F else [])
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(pedidos)} escores a treinar (retreino mensal por família)", flush=True)
    with Pool(4) as p:
        R = dict(p.map(tarefa_escore, pedidos, chunksize=1))
    print("\nSUBSTITUIÇÃO POR FAMÍLIA (mediana das variantes | faixa de detecção | FP | horas fora | horas totais | lead)", flush=True)
    tab = []
    for canal in FAMILIAS:
        for nome in FAMILIA_FABRICAS:
            ms = []
            for (n, c, nt, sm), s in R.items():
                if n != nome or c != canal:
                    continue
                e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); novo = mapeia_quantis(e, proc[canal], vigS)
                r = E6.mede(decide_v2(substitui(proc, canal, novo), det), reg); ms.append(dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h, principal=(nt == 20000 and sm == 0)))
            d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]; md, mf, mo = np.median(d), np.median(f), np.median(o)
            ver = "DIGNO" if md >= 6 and mf <= 9 and mo <= 407 else "PROMISSOR" if md >= 5 and mf <= 12 and mo <= 800 else "-"
            pr = [m for m in ms if m["principal"]][0]
            print(f"   {canal} <- {nome:8s} ({len(ms)} variantes): detecção {md:.0f}/7 ({min(d)}-{max(d)}) | FP {mf:.0f} ({min(f)}-{max(f)}) | fora {mo:.0f} h | totais {np.median([m['horas'] for m in ms]):.0f} h | lead {np.nanmedian([m['lead'] for m in ms]):.1f} h | principal {pr['det']}/7, {pr['fp']} FP -> {ver}", flush=True)
            tab.append(dict(canal=canal, escore=nome, det=md, det_min=min(d), det_max=max(d), fp=mf, fora=mo, ver=ver))
    pd.DataFrame(tab).to_csv(CACHE.parent / "exp14_tabela.csv", index=False)


if __name__ == "__main__":
    main()
