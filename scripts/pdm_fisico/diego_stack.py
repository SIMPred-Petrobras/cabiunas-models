#!/usr/bin/env python3
"""O stack de features do EXP7 contra O NOSSO ALVO, com O NOSSO PROTOCOLO.

Pergunta. Ate agora comparamos o numero publicado pelo Diego (92,5% de hit_rate em
alarme de temperatura, 0,35% de falso alerta) com o nosso (7/9 paradas reais, 29,8 h de
alarme/mes). Sao alvos, reguas e protocolos diferentes -- a comparacao nao decide nada.
Este script tira as tres diferencas de uma vez: o stack dele, o nosso alvo, a nossa regua.

O QUE E O STACK DELE (EXP7 itens 1+2, o candidato de referencia dele):
  - features ponto a ponto, sem janelamento sequencial;
  - por sensor e por janela [6min, 1h, 4h, 24h]: media movel, desvio movel e trend_w
    (valor atual menos o de w passos atras);
  - textura nas janelas >= 1h: kurtosis, skewness e crest factor;
  - modelo nao supervisionado sobre isso (iforest e ocsvm, os dois que venceram grids
    dele) e limiar por percentil do escore de TREINO, mais debounce.

O QUE MUDA AQUI (de proposito):
  - ALVO: parada real de maquina (falhas.csv, regra do Francisco), nao alarme de tag.
  - REGUA: avalia.py -- janela de 48 h, episodios agrupados em 2 h, custo em FALSO
    POSITIVO POR MES DE OPERACAO **e em horas**, nunca so em contagem de episodio.
  - MASCARA: a nossa (operacao quente T5>300 + blackout de 6 h pos-partida), nao
    state=="on". Ja medimos que a diferenca entre as duas e o que faz o portao de rampa
    do EXP10b ser nulo aqui.
  - SELECAO: percentil e sustentacao escolhidos SO no treino (<2025-07-01); depois LOEO.
    Nunca se olha o teste para escolher.

DOIS EIXOS DE ABLACAO, para o resultado ter causa atribuivel:

  entrada   'diego'  = TC382_03_A + T5_AVG_A + 10 sondas de vibracao (o grupo dele)
            'nosso'  = os 36 sensores do detector daqui
            -> separa "o metodo dele e melhor" de "ele estava olhando menos sensores"

  ajuste    'estatico'  = um unico fit em tudo antes de 2025-07-01 (o protocolo dele)
            'walk'      = refit mensal nas ultimas 20.000 amostras estaveis (o nosso)
            -> testa DIRETAMENTE a previsao que eu fiz no ultimo turno: como o custo
               deriva de forma monotona (deriva.py), um modelo com fit unico deveria
               degradar mais que um com referencia que se reancora. Previsao minha,
               entao tem que ser medida, nao afirmada.

Armadilhas numericas ja pagas em textura.py e repetidas aqui de proposito:
  - kurtosis/skewness montadas a mao a partir de medias moveis; pandas .rolling().kurt()
    devolve NaN para a janela inteira se houver NaN no meio dela;
  - textura calculada sobre o sinal CENTRADO por linha de base movel de 24 h -- crest
    factor sobre o sinal cru mede o offset DC (fica colado em 1,015), nao a forma.
"""
from __future__ import annotations
import sys, gc, argparse
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C
import avalia as A
from ablacao import canonico, mascara_pontuacao, CORTE

JANELAS = {"6min": 3, "1h": 30, "4h": 120, "24h": 720}      # grade de 2 min
JAN_TEXTURA = ["1h", "4h"]
BASE_H = 24
FIT_N = 20_000
FIT_OCSVM = 5_000
PERCENTIS = [99.0, 99.5, 99.9, 99.95, 99.99]
SUSTENTA_MIN = [2, 10, 30, 60]


def _mp(n):
    """min_periods seguro: metade da janela, nunca menor que 2 nem maior que a janela.
    A janela de 6 min tem so 3 amostras na grade de 2 min."""
    return max(2, min(n, n // 2))


def _mm(x, n):
    return x.rolling(n, min_periods=_mp(n)).mean()


def monta_features(g: pd.DataFrame, sensores: list[str]) -> pd.DataFrame:
    """Features ponto a ponto do EXP7 item 1+2. float32 para caber na memoria."""
    V = g[sensores].astype("float64").interpolate(limit=5)
    D = V - V.rolling(int(BASE_H * 30), min_periods=int(BASE_H * 30) // 4).median()
    saida = {}
    for nome, n in JANELAS.items():
        m = _mm(V, n)
        saida[f"mean_{nome}"] = m
        saida[f"std_{nome}"] = V.rolling(n, min_periods=_mp(n)).std()
        saida[f"trend_{nome}"] = V - V.shift(n)
        if nome in JAN_TEXTURA:
            md = _mm(D, n)
            d = D - md
            v2 = _mm(d ** 2, n)
            sd = v2 ** 0.5
            saida[f"kurt_{nome}"] = _mm(d ** 4, n) / (v2 ** 2).replace(0, np.nan) - 3.0
            saida[f"skew_{nome}"] = _mm(d ** 3, n) / (sd ** 3).replace(0, np.nan)
            saida[f"crest_{nome}"] = (D.abs().rolling(n, min_periods=_mp(n)).max()
                                      / (_mm(D ** 2, n) ** 0.5).replace(0, np.nan))
    blocos = []
    for k, v in saida.items():
        v = v.astype("float32")
        v.columns = [f"{c}__{k}" for c in v.columns]
        blocos.append(v)
    del saida, V, D
    gc.collect()
    F = pd.concat(blocos, axis=1)
    del blocos
    gc.collect()
    return F


def ajusta(Xf: pd.DataFrame, modelo: str, seed=0):
    """Devolve uma funcao que pontua um bloco. Escore maior = mais anomalo."""
    sc = StandardScaler().fit(Xf)
    if modelo == "iforest":
        m = IsolationForest(n_estimators=200, max_samples=min(len(Xf), 8192),
                            random_state=seed, n_jobs=-1).fit(sc.transform(Xf))
        f = lambda Z: -m.score_samples(sc.transform(Z))
    else:                                        # ocsvm RBF, subamostrado no fit
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(Xf), size=min(FIT_OCSVM, len(Xf)), replace=False)
        m = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(sc.transform(Xf.iloc[sel]))
        f = lambda Z: -m.decision_function(sc.transform(Z))
    return f


def pontua(F: pd.DataFrame, idx_fit, modelo: str, alvo=None, seed=0) -> np.ndarray:
    """Ajusta nas linhas idx_fit e pontua 'alvo' (default: F inteiro)."""
    Xf = F.loc[idx_fit].replace([np.inf, -np.inf], np.nan).dropna()
    B = F if alvo is None else F.loc[alvo]
    out = np.full(len(B), np.nan)
    if len(Xf) < 500:
        return out
    Z = B.replace([np.inf, -np.inf], np.nan)
    ok = Z.notna().all(axis=1).to_numpy()
    if ok.any():
        out[ok] = ajusta(Xf, modelo, seed)(Z[ok])
    return out


def roda_escore(F, df, mask, modelo, ajuste, seed=0) -> pd.Series:
    idx = F.index
    estavel = df["stable"].astype(bool)
    s = pd.Series(np.nan, index=idx, dtype="float64")
    if ajuste == "estatico":
        fit = F.loc[estavel & (idx < CORTE)].dropna().tail(FIT_N).index
        s[:] = pontua(F, fit, modelo, seed=seed)
        return s
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta("2min")
        fit = F.loc[estavel & (idx < m0)].dropna().tail(FIT_N).index
        sel = (idx >= m0) & (idx < m1)
        if len(fit) < FIT_N // 4 or not sel.any():
            continue
        s.loc[sel] = pontua(F, fit, modelo, alvo=idx[sel], seed=seed)
        print(f"    {m0.date()} ok", flush=True)
    return s


def alerta_de(s: pd.Series, mask, limiar: float, sust_min: int) -> pd.Series:
    acima = (s > limiar).where(mask, False).fillna(False)
    return A.sustenta(acima, sust_min) & mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", choices=["diego", "nosso"], default="diego")
    ap.add_argument("--modelo", choices=["iforest", "ocsvm"], default="iforest")
    ap.add_argument("--ajuste", choices=["estatico", "walk"], default="estatico")
    a = ap.parse_args()

    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr
    ev_tr, ev_te = falhas[falhas < CORTE], falhas[falhas >= CORTE]

    sensores = (["TC382_03_A", "T5_AVG_A"] + list(C.VIBRATION_TAGS) if a.entrada == "diego"
                else list(dict.fromkeys(C.SENSOR_TAGS)))
    tag = f"{a.entrada}_{a.modelo}_{a.ajuste}"
    print(f"[{tag}] {len(sensores)} sensores", flush=True)
    F = monta_features(g, sensores)
    print(f"[{tag}] {F.shape[1]} features, {F.memory_usage(deep=True).sum()/1e9:.2f} GB",
          flush=True)
    del g; gc.collect()

    s = roda_escore(F, df, mask, a.modelo, a.ajuste)
    del F; gc.collect()
    print(f"[{tag}] escore pronto: {100*s.notna().mean():.1f}% nao-NaN", flush=True)
    s.to_frame("escore").to_parquet(f"escore_{tag}.parquet")

    # limiar por percentil do escore no TREINO estavel, como no EXP6/EXP7
    ref = s[mask & tr].dropna()
    linhas = []
    for p in PERCENTIS:
        lim = float(np.percentile(ref, p))
        for sm in SUSTENTA_MIN:
            al = alerta_de(s, mask, lim, sm)
            d = {"entrada": a.entrada, "modelo": a.modelo, "ajuste": a.ajuste,
                 "percentil": p, "sustenta_min": sm, "limiar": lim}
            for rot, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te),
                               ("tot", pd.Series(True, index=idx), falhas)]:
                am, qm = al[m], (mask & m)[m]
                x = A.avalia(am, ev, qm)
                d.update({f"{rot}_det": x["det"], f"{rot}_n": x["n_ev"], f"{rot}_eps": x["episodios"],
                          f"{rot}_fp": x["fp_mes"], f"{rot}_h": x["h_fp_mes"], f"{rot}_lead": x["lead_med"]})
            d["quais"] = ",".join(A.avalia(al, falhas, mask)["detectados"])
            linhas.append(d)
    t = pd.DataFrame(linhas)
    t.to_csv(f"diego_stack_{tag}.csv", index=False)

    print(f"\n=== [{tag}] grid completo (selecao SO pelo treino) ===")
    print(f"{'perc':>7} {'sust':>5} | {'treino':>7} {'FP/mes':>7} {'h/mes':>7} | "
          f"{'teste':>7} {'FP/mes':>7} {'h/mes':>7} | {'total':>7} {'h/mes':>7}")
    for _, r in t.iterrows():
        print(f"{r.percentil:7.2f} {r.sustenta_min:5d} | {r.tr_det:>3d}/{r.tr_n:<3d} "
              f"{r.tr_fp:7.2f} {r.tr_h:7.1f} | {r.te_det:>3d}/{r.te_n:<3d} {r.te_fp:7.2f} "
              f"{r.te_h:7.1f} | {r.tot_det:>3d}/{r.tot_n:<3d} {r.tot_h:7.1f}")

    ORC = 80.0     # h de alarme/mes do detector daqui no treino, orcamento fixado a priori
    s_ok = t[t.tr_h <= ORC]
    print(f"\nponto escolhido no TREINO (h/mes <= {ORC:.0f}, maior deteccao, empate -> menor h):")
    if s_ok.empty:
        print("  nenhum ponto do grid respeita o orcamento de horas do treino")
    else:
        r = s_ok.sort_values(["tr_det", "tr_h"], ascending=[False, True]).iloc[0]
        print(f"  percentil={r.percentil} sustenta={r.sustenta_min}min")
        print(f"    treino: {r.tr_det}/{r.tr_n}  {r.tr_h:.1f} h/mes  ({r.tr_eps} episodios)")
        print(f"    TESTE : {r.te_det}/{r.te_n}  {r.te_h:.1f} h/mes  ({r.te_eps} episodios)")
        print(f"    quais : {r.quais}")


main()
