#!/usr/bin/env python3
"""As ideias daqui aplicadas ao detector DELE, no alvo DELE, com a regua DELE.

Espelho de diego_stack.py. La levamos o stack do EXP7 para o nosso alvo e nada
melhorou o detector daqui. Aqui o caminho inverso: as construcoes deste projeto
aplicadas ao candidato de temperatura, medidas nos 32 alarmes genuinos do EXP7
(TC382_03_A HI/HIHI + T5_AVG_A HI, OOS 2025-07 a 2026-04) com a regua dele
(janela de +-24 h, detalhamento preditivo/reativo/sem deteccao, custo em
normal_alert_rate).

ETAPA 0 -- CALIBRAR A REPRODUCAO. Isto nao e o modelo dele. O deployado e um
  ocsvm p99,9/debounce=1; aqui e Isolation Forest sobre as mesmas features
  (multi-escala 6min/1h/4h/24h + textura). Antes de medir qualquer ganho e
  preciso saber quao perto do 92,5%/1,94% dele a reproducao chega. Se ficar
  longe, ganho medido em cima dela nao transfere e a coisa certa e mandar as
  ideias para ele testar no modelo real. Isto e o pre-requisito, nao formalidade.

IDEIA 1 -- TETO DE DURACAO. O relatorio dele diz que 72,5% dos 12.782 pontos de
  falso alerta estao em 10 dos 295 episodios. E a mesma distribuicao que
  motivou o teto daqui (69% das horas em 14 episodios de mais de 2 dias), que
  cortou 72% das horas sem custo de deteccao. Ele testou DEBOUNCE, que corta
  pelo COMECO do episodio e por isso nao separa FP curto de precursor curto --
  ele proprio documenta o motivo (duracao mediana do FP residual 2,5 min contra
  25o percentil de 6 min dos precursores). O teto corta pelo FIM: nao toca no
  instante da primeira deteccao, que e o que define hit e antecedencia. Previsao
  falsificavel: derruba horas de alarme sem perder NENHUM preditivo.

IDEIA 2 -- MASCARA DE OPERACAO QUENTE + BLACKOUT POS-PARTIDA. A dele e
  state=="on" mais o piso secundario de 150 degC do EXP10. A daqui exige
  T5>300 degC e apaga 6 h apos cada religamento -- e ja medimos que ela sozinha
  descarta 92,7% das rampas altas, que e o servico do portao de rampa do EXP10b.
  Hipotese: a mascara substitui os dois portoes dele com menos peca movel e
  menos parametro ajustado no OOS.

IDEIA 3 -- PISO DE ESCALA NA NORMALIZACAO. O StandardScaler divide cada feature
  pelo desvio do TREINO; uma feature quieta no treino recebe denominador
  minusculo e domina o escore no OOS. Mesma patologia que o PHI resolve no
  maximo por sensor. Piso = fracao da mediana dos desvios.

Selecao do ponto: percentil x debounce escolhidos NO TREINO (<2025-07), nunca
olhando o OOS -- que e justamente onde o protocolo do EXP10b/10c escorrega.
"""
from __future__ import annotations
import sys, gc, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C
import avalia as A
from ablacao import canonico, mascara_pontuacao, CORTE
from auto_reset import trunca
from diego_stack import monta_features
from quadrante import alvos_diego, regua_diego, JAN_H

SENSORES = ["TC382_03_A", "T5_AVG_A"] + list(C.VIBRATION_TAGS)
OOS0, OOS1 = pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")
PERCENTIS = [99.0, 99.5, 99.9, 99.95, 99.99]
DEBOUNCE_MIN = [2, 6, 12, 30]
TETOS = [1, 2, 4, 8, 12, 24]
PISOS = [0.0, 0.05, 0.15, 0.30]


def mascara_dele(df, g):
    """state == 'on' com o piso secundario do EXP10 (alvo abaixo de 150 degC = off)."""
    op = df["in_operation"].astype(bool)
    return op & (g["T5_AVG_A"] > 150).fillna(False)


def treino_dele(df, g, alarmes_ts, excl_min=60):
    """Conjunto de fit no espirito do EXP6: operando, longe de alarme, antes do OOS."""
    idx = df.index
    m = mascara_dele(df, g) & (idx < OOS0)
    for t in alarmes_ts:
        m &= ~((idx >= t - pd.Timedelta(minutes=excl_min)) & (idx <= t + pd.Timedelta(minutes=excl_min)))
    return m


def escore_com_piso(F, idx_fit, piso, seed=0):
    Xf = F.loc[idx_fit].replace([np.inf, -np.inf], np.nan).dropna()
    sc = StandardScaler().fit(Xf)
    if piso > 0:                       # piso no denominador da normalizacao
        sc.scale_ = np.maximum(sc.scale_, piso * np.median(sc.scale_))
    m = IsolationForest(n_estimators=200, max_samples=min(len(Xf), 8192),
                        random_state=seed, n_jobs=-1).fit(sc.transform(Xf))
    Z = F.replace([np.inf, -np.inf], np.nan)
    ok = Z.notna().all(axis=1).to_numpy()
    out = np.full(len(F), np.nan)
    if ok.any():
        out[ok] = -m.score_samples(sc.transform(Z[ok]))
    return pd.Series(out, index=F.index)


def alerta(s, mask, lim, deb_min):
    return A.sustenta((s > lim).where(mask, False).fillna(False), deb_min) & mask


def nar(al, mask, longe):
    """normal_alert_rate: fracao dos pontos normais do OOS marcados como anomalos."""
    oos = (al.index >= OOS0) & (al.index <= OOS1)
    sel = mask & oos & longe
    return 100 * (al.fillna(False) & sel).sum() / max(sel.sum(), 1)


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    idx = df.index
    reais, under = alvos_diego()
    todos_ts = pd.concat([reais["t"], under["t"]])
    longe = pd.Series(True, index=idx)
    for t in todos_ts:
        longe &= ~((idx >= t - pd.Timedelta(hours=JAN_H)) & (idx <= t + pd.Timedelta(hours=JAN_H)))

    m_dele = mascara_dele(df, g)
    m_nossa = mascara_pontuacao(df)
    fit_idx = df.index[treino_dele(df, g, todos_ts)]
    print(f"alvos: {len(reais)} HI/HIHI genuinos | fit: {len(fit_idx)} amostras "
          f"({len(fit_idx)*2/60:.0f} h) antes de {OOS0.date()}", flush=True)

    print("montando features do EXP7 ...", flush=True)
    F = monta_features(g, SENSORES)
    print(f"{F.shape[1]} features, {F.memory_usage(deep=True).sum()/1e9:.2f} GB", flush=True)
    del g; gc.collect()

    # ---------------- etapa 0 + ideia 3: piso de escala
    linhas = []
    escores = {}
    for piso in PISOS:
        s = escore_com_piso(F, fit_idx, piso)
        escores[piso] = s
        ref = s[m_dele & (idx < OOS0)].dropna()
        print(f"\n{'='*80}\nPISO DE ESCALA = {piso}   (0.0 = reproducao base do EXP7)")
        print(f"{'perc':>7} {'deb':>5} | {'hit':>7} {'pred':>5} {'reat':>5} {'nada':>5} "
              f"{'taxa pred':>10} {'lead':>7} {'NAR':>7}")
        for p in PERCENTIS:
            lim = float(np.percentile(ref, p))
            for d in DEBOUNCE_MIN:
                al = alerta(s, m_dele, lim, d)
                x = regua_diego(al, reais, m_dele)
                print(f"{p:7.2f} {d:4d}m | {x['hit']:6.1f}% {x['pred']:5d} {x['reat']:5d} "
                      f"{x['nada']:5d} {x['taxa_pred']:9.1f}% {x['lead_med']:6.1f}h "
                      f"{nar(al, m_dele, longe):6.2f}%")
                linhas.append(dict(piso=piso, percentil=p, debounce=d, teto=None,
                                   mascara="dele", nar=nar(al, m_dele, longe),
                                   **{k: v for k, v in x.items() if k != "sem_det"}))
    pd.DataFrame(linhas).to_csv("melhora_diego_grid.csv", index=False)

    print(f"\n{'='*80}\nreferencia publicada (EXP7 item1+2, ocsvm p99,9/db1): "
          f"hit 100%, 29 pred, 3 reat, 0 nada, NAR 1,94%")
    print(f"referencia publicada (EXP10c, com as duas portas):            "
          f"hit 100%, 29 pred, 3 reat, 0 nada, NAR 0,35%")

    # ponto escolhido no TREINO: maior deteccao de treino sob orcamento de NAR
    print(f"\n{'='*80}\nPONTO ESCOLHIDO NO TREINO (sem olhar o OOS)")
    s0 = escores[0.0]
    ref = s0[m_dele & (idx < OOS0)].dropna()
    alv_tr = reais.iloc[0:0]  # nao ha alarme genuino antes do OOS nesta amostra
    melhor = None
    for p in PERCENTIS:
        lim = float(np.percentile(ref, p))
        for d in DEBOUNCE_MIN:
            al = alerta(s0, m_dele, lim, d)
            tr = (idx < OOS0) & m_dele
            duty = 100 * (al & tr).sum() / max(tr.sum(), 1)
            if melhor is None or abs(duty - 1.94) < abs(melhor[0] - 1.94):
                melhor = (duty, p, d, lim)
    duty, P, D, LIM = melhor
    print(f"  percentil={P} debounce={D}min  (duty de treino {duty:.2f}%, alvo 1,94% do EXP7)")
    base = alerta(s0, m_dele, LIM, D)
    xb = regua_diego(base, reais, m_dele)
    print(f"  reproducao no OOS: hit {xb['hit']:.1f}%  {xb['pred']} pred  {xb['reat']} reat  "
          f"{xb['nada']} nada  NAR {nar(base, m_dele, longe):.2f}%")

    # ---------------- ideia 1: teto de duracao
    print(f"\n{'='*80}\nIDEIA 1 -- TETO DE DURACAO sobre o ponto acima")
    print(f"{'teto':>6} | {'hit':>7} {'pred':>5} {'reat':>5} {'nada':>5} {'lead':>7} "
          f"{'NAR':>7} {'reducao':>9}")
    n0 = nar(base, m_dele, longe)
    print(f"{'sem':>6} | {xb['hit']:6.1f}% {xb['pred']:5d} {xb['reat']:5d} {xb['nada']:5d} "
          f"{xb['lead_med']:6.1f}h {n0:6.2f}% {'-':>9}")
    fin = []
    for T in TETOS:
        al = trunca(base, T)
        x = regua_diego(al, reais, m_dele); n = nar(al, m_dele, longe)
        print(f"{T:5d}h | {x['hit']:6.1f}% {x['pred']:5d} {x['reat']:5d} {x['nada']:5d} "
              f"{x['lead_med']:6.1f}h {n:6.2f}% {100*(1-n/max(n0,1e-9)):8.1f}%")
        fin.append(dict(ideia="teto", par=T, nar=n, **{k: v for k, v in x.items() if k != "sem_det"}))

    # ---------------- ideia 2: mascara nossa
    print(f"\n{'='*80}\nIDEIA 2 -- MASCARA DE OPERACAO QUENTE + BLACKOUT POS-PARTIDA")
    print(f"{'mascara':>34} | {'hit':>7} {'pred':>5} {'reat':>5} {'nada':>5} {'lead':>7} {'NAR':>7}")
    for rot, mk in [("dele (on + piso 150C)", m_dele),
                    ("nossa (quente T5>300 + blackout 6h)", m_nossa)]:
        ref2 = s0[mk & (idx < OOS0)].dropna()
        lim2 = float(np.percentile(ref2, P))
        al = alerta(s0, mk, lim2, D)
        x = regua_diego(al, reais, mk); n = nar(al, mk, longe)
        print(f"{rot:>34} | {x['hit']:6.1f}% {x['pred']:5d} {x['reat']:5d} {x['nada']:5d} "
              f"{x['lead_med']:6.1f}h {n:6.2f}%")
        fin.append(dict(ideia="mascara", par=rot, nar=n, **{k: v for k, v in x.items() if k != "sem_det"}))
        alt = trunca(al, 12)
        xt = regua_diego(alt, reais, mk); nt = nar(alt, mk, longe)
        print(f"{rot + '  + teto 12h':>34} | {xt['hit']:6.1f}% {xt['pred']:5d} {xt['reat']:5d} "
              f"{xt['nada']:5d} {xt['lead_med']:6.1f}h {nt:6.2f}%")
        fin.append(dict(ideia="mascara+teto", par=rot, nar=nt,
                        **{k: v for k, v in xt.items() if k != "sem_det"}))
    pd.DataFrame(fin).to_csv("melhora_diego.csv", index=False)


main()
