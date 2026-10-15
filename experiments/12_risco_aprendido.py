#!/usr/bin/env python3
"""EXP 12 — TRILHA B: MODELO DE RISCO APRENDIDO sobre os sinais do v2 (fusão supervisionada, validação por blocos de evento).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. Os canais t, p, sp, vb do v2 (já normalizados por base x k) alimentam um voto fixo + CUSUM. Uma fusão APRENDIDA do histórico recente deles (+ tempo desde a partida) estima P(trip nas próximas 48 h) melhor que o voto? É o desenho de Fase 4 do plano (risco e decisão), na versão mais barata possível: sem escores novos.
DADOS. Os sinais causais do v2 existem de 2025-03-05 em diante: 6 grupos de evento (03/17; 04/07 + 04/11 juntos, a 4 dias; 04/29; 11/04; 12/09; 02/26/26). O 27/02/2025 está antes dos sinais e fica fora. É MUITO POUCO: 6 positivos efetivos.
ALVO. y = 1 se o instante está em [T-48 h, T] de um trip (mesma janela do harness); só instantes vigiados e com todos os canais finitos.
FEATURES (causais, 17): para cada canal c em {t,p,sp,vb} (em unidades de base x k_A): valor atual, máximo móvel de 6 h, máximo móvel de 24 h, média móvel de 24 h; mais o 2º maior canal atual (ord2) e log1p(tau em horas desde a partida).
MODELOS: (L) regressão logística L2 forte (C=0,05, balanceada); (G) LightGBM raso (profundidade 2, 60 árvores, taxa 0,05, min_child_samples 300, balanceado). Treino subamostrado a cada 15 min.
VALIDAÇÃO. 6 dobras por GRUPO de evento: o bloco de teste vai das bordas (ponto médio entre os grupos vizinhos) e o treino exclui o bloco e 7 dias além de cada borda. O risco fora-da-dobra concatenado cobre todos os 6 blocos e é pontuado por UMA política fixa: EWMA de 1 h vigiado, dur >= 2 h, refratário 72 h (a de `nv.politica`), com o limiar escolhido para o MESMO orçamento de 407 h de alarme fora das janelas do controle (comparação a custo igual, como no EXP 04).
LIMITES DECLARADOS (otimismo): o treino de cada dobra usa eventos POSTERIORES ao teste (não é causal; é o melhor caso de informação) e o limiar é calibrado no conjunto fora-da-dobra concatenado.
PLACEBO: os rótulos deslocados de 45, 90, 135, 180, 225 e 270 dias (mesmas dobras, mesmo orçamento) para as duas famílias: o que o modelo chega por estrutura de partida/ruído sem o alvo verdadeiro.
REFERÊNCIAS (mesma política, mesmo orçamento): controle do v2 6/7 (voto+CUSUM); canal t sozinho 5/7 e ord2 5/7 (EXP 05); melhor baseline 4/7 (EXP 04).
MÉTRICAS: detecção por evento (7), FP em episódios e horas totais; AUC do risco por bloco (janela de 48 h contra o resto do bloco, só vigiado) e mediana por bloco.
CRITÉRIO: DIGNO = detecção >= 6/7 a 407 h fora e o placebo (mediana dos 6 deslocamentos) <= detecção - 2; PROMISSOR = >= 5/7 com o placebo <= detecção - 1; NÃO SUPERA = o resto (inclui empatar com o canal t sozinho, 5/7, sem margem sobre o placebo).
EXPECTATIVAS: DIGNO ~6%; PROMISSOR ~20%; NÃO SUPERA ~74%. AUC mediana por bloco >= 0,70 em L ~40%; G melhor que L em detecção ~35% (poucos positivos favorecem o modelo simples). O placebo chega a >= 4/7 em alguma família ~45% (a política com orçamento de 407 h acerta janelas de 48 h por cobertura: ver `cobertura_ao_acaso`).

RESULTADO (15/10/2026) -- NÃO SUPERA, com folga. (Nota: são 18 features, não 17; erro de contagem no pré-registro, sem efeito no critério.) 248.208 instantes úteis, 6 dobras, orçamento de 407 h fora.
  [L] logística: detecção 1/7 | início 1 | FP 14 | horas totais 411 | lead 9,3 h | AUC por bloco [0,29; 0,17; 0,83; 0,14; 0,51; 0,37], mediana 0,33. Placebo (det, FP, h fora): (3,22,398) (1,13,407) (0,8,346) (1,11,303) (2,13,409) (3,28,407), mediana de detecção 1,5.
  [G] GBM raso: detecção 0/7 | FP 16 | horas totais 418 | AUC por bloco [0,40; 0,51; 0,89; 0,44; 0,63; 0,56], mediana 0,54. Placebo: (1,11,407) (1,10,407) (0,7,380) (2,11,407) (2,22,422) (1,19,407), mediana 1,0.
  REFERÊNCIAS à mesma política e orçamento: controle (voto+CUSUM) 6/7; canal t sozinho 5/7; melhor baseline 4/7. A fusão aprendida (0 a 1 de 7) NÃO chega nem perto, e o rótulo verdadeiro não se distingue do placebo (L 1 contra 1,5; G 0 contra 1,0).
  LEITURA. Com 6 grupos de evento a fusão supervisionada dos sinais do v2 não aprende uma direção que generalize entre eventos: AUC por bloco abaixo de 0,5 em 4 de 6 blocos na logística (o evento do bloco de teste é o oposto do que o treino viu), o que repete o achado do M31 (AUC LOEO 0,31 e 0,49: os eventos não têm forma comum). O voto + CUSUM do v2 vence por ser NÃO supervisionado: não depende de os eventos se parecerem. O limite é o número de positivos efetivos (6), e nenhum desenho de modelo remove isso.
PLACAR: ACERTEI: NÃO SUPERA (~74%); G melhor que L em detecção ~35% (não: 0 contra 1; mas G tem AUC mediana maior, 0,54 contra 0,33). ERREI: AUC mediana >= 0,70 em L (~40%: deu 0,33). ACERTEI: o placebo chega a >= 4/7 em alguma família (~45%): NÃO chegou (máximo 3); então o orçamento de 407 h não dá cobertura alta ao acaso nesta política.

Uso:  PYTHONPATH=src .venv/bin/python experiments/12_risco_aprendido.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.contexto import tau_horas
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.politica import ewma_vigiado, episodios, alarme_de, calibra_limiar

INICIO = E4.INICIO; JAN = pd.Timedelta(hours=48); BUF = pd.Timedelta(days=7); PASSO = 7  # 15 min
CANAIS = E5.CANAIS


def features(S, det, tau):
    E = pd.DataFrame({c: S[c] / (det["base"][c] * det["k_nivel_a"][c]) for c in CANAIS}).where(S["mask"].astype(bool))
    F = {}
    for c in CANAIS:
        F[f"{c}_0"] = E[c]; F[f"{c}_max6"] = E[c].rolling(180, min_periods=30).max(); F[f"{c}_max24"] = E[c].rolling(720, min_periods=90).max(); F[f"{c}_med24"] = E[c].rolling(720, min_periods=90).mean()
    M = np.sort(np.nan_to_num(E.to_numpy(), nan=-np.inf), axis=1)[:, ::-1]; F["ord2"] = pd.Series(np.where(np.isfinite(M[:, 1]), M[:, 1], np.nan), index=E.index)
    F["log_tau"] = np.log1p(tau.reindex(E.index).clip(lower=0))
    return pd.DataFrame(F)


def rotulos(idx, trips, deslocamento=pd.Timedelta(0)):
    y = np.zeros(len(idx), bool)
    for T in trips:
        T = T + deslocamento; y |= (idx >= T - JAN) & (idx <= T)
    return y


def dobras(trips, idx):
    """Grupos de evento (trips a < 10 d formam um grupo) e blocos entre pontos médios."""
    g = [[trips[0]]]
    for t in trips[1:]:
        (g[-1].append(t) if t - g[-1][-1] < pd.Timedelta(days=10) else g.append([t]))
    bordas = [idx[0]] + [g[i][-1] + (g[i + 1][0] - g[i][-1]) / 2 for i in range(len(g) - 1)] + [idx[-1] + pd.Timedelta("2min")]
    return g, [(bordas[i], bordas[i + 1]) for i in range(len(g))]


def fabrica(nome):
    if nome == "L":
        return lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
    return lambda: lgb.LGBMClassifier(max_depth=2, n_estimators=60, learning_rate=0.05, min_child_samples=300, class_weight="balanced", verbose=-1, n_jobs=2)


def risco_fora_da_dobra(F, ok, trips_rot, blocos, fam):
    idx = F.index; risco = pd.Series(np.nan, index=idx); X = F.to_numpy(float); posic = np.arange(len(idx))
    for (a, b) in blocos:
        teste = ok & (idx >= a) & (idx < b); treino = ok & ((idx < a - BUF) | (idx >= b + BUF)) & (posic % PASSO == 0)
        y = trips_rot
        if y[treino].sum() < 50 or y[treino].all():
            continue
        sc = StandardScaler().fit(X[treino]); m = fabrica(fam)(); m.fit(sc.transform(X[treino]), y[treino])
        risco[teste] = m.predict_proba(sc.transform(X[teste]))[:, 1]
    return risco


def avalia_risco(risco, vig, reg, alvo):
    ini = INICIO - pd.Timedelta(days=10); r = risco.loc[ini:]; v = vig.loc[ini:]
    e = ewma_vigiado(r, v); thr = calibra_limiar(e, v, reg, alvo, inicio=INICIO); al = alarme_de(episodios(e, thr), e.index)
    return avalia(al, v, reg, inicio_periodo=INICIO)


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    al_c, vig_c = carrega_controle("atual"); reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO); alvo = ctrl.horas_alarme_fora
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); tau = tau_horas(g); F = features(S, det, tau); vig = vig_c.reindex(F.index).fillna(False)
    ok = (vig & F.notna().all(axis=1)).to_numpy(bool); idx = F.index
    grupos, blocos = dobras(reg, idx); print(f"CONTROLE: {E6.linha(ctrl)} | {len(F.columns)} features, {int(ok.sum())} instantes úteis | grupos de evento: {[[str(t)[:10] for t in gr] for gr in grupos]}", flush=True)
    resultados = {}
    for fam in ("L", "G"):
        y = rotulos(idx, reg); risco = risco_fora_da_dobra(F, ok, y, blocos, fam); r = avalia_risco(risco, vig, reg, alvo)
        aucs = []
        for (a, b) in blocos:
            m = ok & (idx >= a) & (idx < b)
            if y[m].any() and (~y[m]).any() and risco[m].notna().all():
                aucs.append(roc_auc_score(y[m], risco[m]))
        plac = []
        for d in (45, 90, 135, 180, 225, 270):
            yp = rotulos(idx, reg, pd.Timedelta(days=d)); rp = avalia_risco(risco_fora_da_dobra(F, ok, yp, blocos, fam), vig, reg, alvo); plac.append((rp.det, rp.fp, round(rp.horas_alarme_fora)))
        resultados[fam] = (r, aucs, plac)
        pm = np.median([p[0] for p in plac]); ver = "DIGNO" if r.det >= 6 and pm <= r.det - 2 else "PROMISSOR" if r.det >= 5 and pm <= r.det - 1 else "NÃO SUPERA"
        print(f"\n[{fam}] detecção {r.det}/{r.n_eventos} | início {r.inicio} | FP {r.fp} | horas totais {r.horas_alarme:.0f} | fora {r.horas_alarme_fora:.0f} | lead {r.lead_medio_h:.1f} h | AUC por bloco {np.round(aucs, 2).tolist()} (mediana {np.median(aucs):.2f}) -> {ver}", flush=True)
        print(f"     placebo (det, FP, h fora) por deslocamento: {plac} | mediana de detecção {pm:.1f}", flush=True)


if __name__ == "__main__":
    main()
