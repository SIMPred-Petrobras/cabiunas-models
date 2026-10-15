#!/usr/bin/env python3
"""EXP 07 — O PRIMEIRO MODELO DE REDE NEURAL: autoencoder condicional, mascarado e robusto, com histórico expandido, como canal t ou p DENTRO da decisão do v2.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

HIPÓTESE. O que os canais t e p do v2 têm a mais que qualquer baseline é ESPECIFICIDADE (menos excursões persistentes em operação normal; EXP 06). A fonte provável das excursões normais é a deriva e a mudança de regime (a pesquisa mediu "toda semana já está em drift"). Um modelo de "normal" treinado com TODO o histórico estável anterior (não só as últimas
20.000 amostras) e CONDICIONADO ao regime (tempo desde a partida; nível recente das tags) deve produzir menos excursões em operação normal sem perder os eventos.

O MODELO (`nv.modelos.ae_torch`). Autoencoder MLP (d -> 64 -> 32 -> 4 -> 32 -> 64 -> d, GELU), contexto concatenado no codificador e no decodificador; corrupção por máscara (p = 0,15) e ruído (0,05); perda de Huber; escala robusta com piso; escore = máximo entre as tags do erro quadrado normalizado pelo p99 da tag (piso de 0,1 x mediana) e depois pelo p99 do escore no treino (o desenho do v2).
Retreino MENSAL causal sobre as últimas 250.000 amostras estáveis anteriores ao dia 1 (todo o histórico disponível), warm-start do mês anterior (30 épocas no 1.º mês, 6 nos seguintes). Mesmas tags do v2: temperatura (14: TI_0325/0315/0317/0305/0307/0303/0301, TC382_01..06, T5_AVG_A) e pressão (12 tags, incluindo PI_5134001), lidas do bundle.

CONFIGURAÇÕES (um tiro cada, sem varredura): D0 sem contexto; D1 + tau (horas desde a partida: tau/72, tau < 6 h, tau < 24 h); D2 + tau + NÍVEL LENTO (mediana das últimas 7 d de amostras estáveis atrasada em 3 d, escalada com mediana e MAD do período 2024-02 a 2025-02); D0_20k = D0 treinado só nas últimas 20.000 amostras estáveis (isola o efeito do histórico longo).
Sementes 0, 1 e 2 (piso de ruído próprio). Famílias: temperatura -> substitui o canal t; pressão -> substitui o canal p.

AVALIAÇÃO (o plug-in validado no EXP 06): o escore -> EWMA de 1 h só em vigiado -> mapeamento de quantis para a marginal do canal -> decisão de produção; 7 eventos; os DOIS custos (FP em episódios; horas de alarme totais e fora das janelas). Complemento: o escore sob a política simples a 1x o orçamento em horas (semente 0).
CRITÉRIOS (escritos antes), sobre a MEDIANA das 3 sementes: "DIGNO" = detecção >= 6/7 (a do controle) E FP <= 9 E horas fora das janelas <= 407 h (domina o controle em especificidade, igual em detecção). "PROMISSOR" = detecção >= 5/7 E FP <= 12 E horas fora das janelas <= 800 h (os baselines tiveram >= 955 h).

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  E1  o histórico longo (D0) tem menos horas fora das janelas que o D0_20k em pelo menos uma família, por >= 20% (~55%).
  E2  o contexto de partida (D1) reduz os FP em episódios em >= 2 contra o D0 em pelo menos uma família (~45%).
  E3  o nível lento (D2) reduz as horas fora das janelas em >= 20% contra o D1 mas perde >= 1 detecção (~50%): a troca especificidade x sensibilidade da deriva absorvida.
  E4  algum resultado DIGNO: no canal p ~20%; no canal t ~5%. Algum PROMISSOR: p ~45%, t ~15%.
  E5  o piso de ruído entre sementes é <= 1 evento de detecção e <= 3 FP em >= 70% das combinações (~65%).
  E6  sob a política simples a 1x, o melhor D detecta >= 5/7 (~35%).

LIMITES. 7 eventos; arquitetura e hiperparâmetros fixados a priori (k = 4, 64 unidades, 15% de máscara, 6 épocas): sem busca; o mapa de quantis é global no período (ajuste de escala sem rótulo); o contexto de nível lento usa a história de 10 dias, então os primeiros dias de dado não têm contexto (preenchido com 0); treino em CPU e modelos pequenos;
o AE é um MLP por amostra (sem janela temporal): uma versão sequencial (TCN/Transformer) é o passo seguinte se o contexto mostrar valor.

RESULTADO (15/10/2026) -- 0 DIGNOS e 1 PROMISSOR em 8 combinações; A HIPÓTESE DO HISTÓRICO LONGO FOI REFUTADA: o histórico longo PIORA a especificidade. 24 execuções, ~75 s cada (D0_20k ~7 s).
  AE DENTRO DA DECISÃO DO v2 (mediana de 3 sementes; faixa entre parênteses; detecção | FP episódios | horas fora das janelas | lead h):
    canal t:  D0 3/7 (3-3) | 21 | 1.626 | 36,2;   D1 4/7 (3-5) | 21 | 1.526 | 34,9;   D2 6/7 (5-6) | 18 | 1.748 | 25,6;   D0_20k 4/7 (3-5) | 16 | 953 | 34,5.
    canal p:  D0 6/7 (2-6) | 14 | 1.311 | 27,9;   D1 5/7 (4-5) | 16 | 1.057 | 24,6;   D2 5/7 (4-5) | 15 | 1.003 | 26,8;   D0_20k 6/7 (6-6) | 11 (10-12) | 722 (530-1.197) | 32,7  -> PROMISSOR (o único).
  CONTROLE: 6/7 | 9 FP | 407 h fora das janelas | lead 32,7 h. COMPARAÇÃO COM OS BASELINES NO CANAL p (EXP 06): FP 11 a 24 e 1.290 a 1.764 h; o D0_20k no p (FP 11, 722 h) é o MELHOR em especificidade entre todos os substitutos do p (-44% nas horas contra o melhor baseline), mas ainda tem 1,8x as horas fora das janelas do p do v2.
  SOB A POLÍTICA SIMPLES A 1x EM HORAS (semente 0): D2 temperatura 5/7 (13 FP, lead 22,6 h) e D0_20k pressão 5/7 (26 FP): o D2 de temperatura iguala o canal t do v2 (5/7 com 17 FP) com MENOS episódios (13); os demais 2 a 3 de 7.
LEITURA. (1) O HISTÓRICO LONGO É PIOR: D0 (250.000 amostras) tem 1.626 h fora no t e 1.311 no p; D0_20k (só as últimas 20.000), 953 e 722. A normalidade desta máquina NÃO É ESTACIONÁRIA: a FRESCURA do baseline vale mais que o VOLUME (o desenho do v2, 20.000 amostras mensais, está certo), e o histórico antigo, com outros regimes, aumenta as excursões no mês corrente.
  (2) OS CONTEXTOS NÃO AJUDAM: o de partida (D1) não reduz os FP (t 21 -> 21; p 14 -> 16) e o nível lento (D2) aumenta a detecção no t (4 -> 6) mas aumenta também as horas falsas (1.526 -> 1.748): absorver a deriva não cortou a persistência das excursões normais.
  (3) O PISO DE RUÍDO DOS MODELOS NEURAIS É MAIOR QUE O DOS BASELINES: detecção entre sementes de 2 a 6 (D0 p), 3 a 5 (D1 t, D0_20k t) e horas fora de 530 a 1.197 (D0_20k p). Com 3 sementes só se lê a mediana; 5 ou mais seriam necessárias para um veredito.
  (4) POR QUE NÃO PASSA: dentro da decisão do v2 (CUSUM com memória), as excursões do AE em operação normal são MAIS PERSISTENTES que as do p do v2 mesmo com a mesma distribuição marginal imposta pelo mapa de quantis; o CUSUM as acumula em episódios longos (as horas fora triplicam). A PERSISTÊNCIA das excursões normais, e não a sua frequência, é o que o canal precisa ter baixa.
PLACAR: ACERTEI: algum PROMISSOR no canal p (~45%: um); o melhor D sob a política simples detecta >= 5/7 (E6: 5/7); nenhum DIGNO no t (~95%). ERREI: o histórico longo reduz as horas fora (E1: aumentou-as em ~70%); o contexto de partida reduz os FP em >= 2 (E2: não); o nível lento reduz as horas fora em >= 20% mas perde detecção (E3: no t aumentou as duas; no p quase igual);
  o piso de ruído <= 1 evento em >= 70% das combinações (E5: 5 de 8 = 62%).
PRÓXIMOS PASSOS PLAUSÍVEIS (a pré-registrar): (a) modelo SEQUENCIAL (TCN) com a persistência das excursões como alvo de projeto; (b) treino com peso decrescente no tempo (frescura) sobre o histórico; (c) o AE como QUINTO canal (acrescenta informação em vez de substituir); (d) mais sementes (>= 5) para o piso de ruído.

CORREÇÃO (EXP 08, 10 sementes): o "PROMISSOR" do D0_20k no canal p (722 h fora) era sorte de semente; com 10 sementes a mediana é 1.075 h (Q1-Q3 695-1.341), acima do limite de 800 h: NÃO é promissor. A detecção de 6/7 e o FP de 11 a 14 se mantêm.

Uso:  PYTHONPATH=src .venv/bin/python experiments/07_ae_condicional.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, time, warnings
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
from nv.contexto import features_tau, nivel_lento, tau_horas
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk_ae import walk_ae

CACHE = Path(__file__).resolve().parents[1] / "data_local" / "escores_ae"
INICIO = E4.INICIO
CONFIGS = {"D0": ("nenhum", 250000), "D1": ("tau", 250000), "D2": ("tau_nivel", 250000), "D0_20k": ("nenhum", 20000)}
FAMILIAS = {"temperatura": "t", "pressao": "p"}
SEMENTES = (0, 1, 2)
_G = {}


def _contexto(g, est, tags, modo):
    n = len(g)
    if modo == "nenhum":
        return np.zeros((n, 0))
    T = features_tau(tau_horas(g))
    if modo == "tau":
        return T
    L = nivel_lento(g, est, tags).to_numpy("float64"); ref = g.loc["2024-02-01":"2025-02-28", tags].to_numpy("float64")[est[(g.index >= pd.Timestamp("2024-02-01", tz="UTC")) & (g.index <= pd.Timestamp("2025-02-28 23:58", tz="UTC"))]]
    med = np.nanmedian(ref, axis=0); sc = np.maximum(np.maximum(1.4826 * np.nanmedian(np.abs(ref - med), axis=0), 0.05 * np.nanmedian(np.abs(ref), axis=0)), 1e-6)
    return np.hstack([T, np.nan_to_num((L - med) / sc, nan=0.0)])


def tarefa(a):
    cfg, fam, sem = a; f = CACHE / f"{cfg}_{fam}_{sem}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    import torch
    torch.set_num_threads(2)
    g, est, tags_f, fim = _G["g"], _G["est"], _G["tags"][fam], _G["fim"]
    modo, n_max = CONFIGS[cfg]; C = _contexto(g, est, tags_f, modo); t0 = time.time()
    s = walk_ae(g, tags_f, est, C, pd.Timestamp("2025-03-01", tz="UTC"), fim, semente=sem, n_max=n_max)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {cfg} {fam} semente {sem}: {time.time() - t0:.0f} s", flush=True)
    return a, s


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {"temperatura": modelos[-1]["normalizacao"]["temperatura"]["cols"], "pressao": modelos[-1]["normalizacao"]["pressao"]["cols"]}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    _G.update(g=g, est=est, tags=tags, fim=fim)
    print(f"CONTROLE: {E6.linha(ctrl)} | famílias: temperatura {len(tags['temperatura'])} tags, pressão {len(tags['pressao'])} tags", flush=True)
    tarefas = [(c, f, s) for c in CONFIGS for f in FAMILIAS for s in SEMENTES]
    print(f"treinando {len(tarefas)} execuções (retreino mensal, histórico expandido)", flush=True)
    with Pool(3) as p:
        S = dict(p.map(tarefa, tarefas, chunksize=1))
    sinais = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = sinais[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    linhas = []
    for cfg in CONFIGS:
        for fam, canal in FAMILIAS.items():
            res = []
            for sem in SEMENTES:
                e = ewma_vigiado(S[(cfg, fam, sem)], vig_g, "1h").reindex(proc.index)
                res.append(E6.mede(decide_v2(substitui(proc, canal, mapeia_quantis(e, proc[canal], vigS)), det), reg))
            d = [r.det for r in res]; fp = [r.fp for r in res]; fo = [r.horas_alarme_fora for r in res]; hh = [r.horas_alarme for r in res]; ld = [r.lead_medio_h for r in res]
            md, mf, mo = int(np.median(d)), int(np.median(fp)), float(np.median(fo))
            digno = md >= 6 and mf <= 9 and mo <= 407; prom = md >= 5 and mf <= 12 and mo <= 800
            linhas.append(dict(config=cfg, canal=canal, det_mediana=f"{md}/7", det_faixa=f"{min(d)}-{max(d)}", fp_mediana=mf, fp_faixa=f"{min(fp)}-{max(fp)}", fora_h_mediana=round(mo), fora_h_faixa=f"{min(fo):.0f}-{max(fo):.0f}", horas=round(float(np.median(hh))), lead_h=round(float(np.nanmedian(ld)), 1), veredito="DIGNO" if digno else "PROMISSOR" if prom else "-"))
    T = pd.DataFrame(linhas)
    print(f"\nCONTROLE: {E6.linha(ctrl)}\n\nAE DENTRO DA DECISÃO DO v2 (mediana e faixa de 3 sementes)"); print(T.to_string(index=False))
    print("\nCOMPLEMENTO: o escore sob a política simples a 1x o orçamento em horas (semente 0)")
    for cfg in CONFIGS:
        for fam in FAMILIAS:
            r, al, thr = E4.avalia_a_custo(S[(cfg, fam, 0)], vig_g, reg, ctrl.horas_alarme_fora)
            print(f"   {cfg:7s} {fam:11s}: detecção {r.det}/{r.n_eventos}, lead {r.lead_medio_h:.1f} h, FP {r.fp}, fora {r.horas_alarme_fora:.0f} h")
    print(f"\nDIGNOS: {int((T.veredito == 'DIGNO').sum())} | PROMISSORES: {int((T.veredito == 'PROMISSOR').sum())} de {len(T)} combinações")


if __name__ == "__main__":
    main()
