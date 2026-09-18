#!/usr/bin/env python3
"""DRIFT: desacoplar o MODELO da ESCALA -- a correcao que faltava testar.

O DIAGNOSTICO DE RAIZ. O detector usa a MESMA amostra de 20.000 pontos para duas
funcoes com requisitos opostos:

  * o PCA quer dado RECENTE, para representar o normal de agora. E por isso que
    o retreino mensal e obrigatorio: congelar leva 8/8 a 6/8 e 7,15 a 108 h/mes.
  * o `recon_p99` quer dado ABUNDANTE. E um percentil de CAUDA -- o estimador
    mais instavel que existe -- e estima-lo sobre 20.000 pontos cuja composicao
    muda todo mes produz oscilacao de 2,45x ao longo da serie e ate +91% de um
    mes para o outro.

Nao da para otimizar uma amostra para as duas coisas ao mesmo tempo. E por isso
que tudo que se mexe nela quebra: janela menor, janela maior, limpeza, ensemble.

O QUE JA FOI TENTADO E POR QUE FALHOU. `drift_normalizador.py` suavizou o p99 pela
mediana dos ultimos k bundles. Piorou, porque usava o p99 de um PCA ANTIGO com um
PCA NOVO -- escala e modelo descasados. A conclusao de la (o par PCA-p99 e atomico)
continua valendo, e e exatamente o que esta versao respeita.

A CORRECAO. Manter o PCA mensal, e calcular o p99 pontuando um historico LONGO
**com o PCA novo**. A escala continua sendo a daquele modelo -- nada descasa --
mas passa a ser estimada sobre 5 a 10 vezes mais amostras.

O risco, que e o motivo de medir: historico longo carrega regimes antigos, onde o
PCA novo erra muito por drift. Isso INFLA o p99 e pode dessensibilizar o detector.
Se inflar demais, a deteccao cai -- e o teste mostra onde fica o limite.

Duas familias testadas:
  A) p99 sobre janela de calibracao crescente (20k = atual, 60k, 120k, tudo)
  B) trocar o estimador de cauda por um robusto: mediana + k*MAD do erro, que tem
     variancia muito menor que um percentil extremo

RESULTADO: AS DUAS FAMILIAS FALHAM, E A HIPOTESE DE RAIZ ESTAVA ERRADA.

    calibracao da escala        amplitude da escala   deteccao
    20.000  (atual, acoplado)          2,45x            8/8
    60.000                          1279,61x            1/8
    120.000                         1205,36x            1/8
    todo o passado                   267,32x            1/8

O oposto do previsto, por duas ordens de grandeza. Amostra MAIOR deu escala MUITO
mais instavel, nao menos.

POR QUE. O historico longo contem regimes em que o PCA novo erra absurdamente --
o modelo de 2026 aplicado a dado de 2024. O percentil passa a ser dominado por
esses residuos de DRIFT, e varia conforme a proporcao de dado antigo na janela.

A CONSEQUENCIA E QUE A PREMISSA CAI. A variacao de 2,45x do `recon_p99` nao e
ruido de estimador pequeno -- e o SINAL LEGITIMO do drift. O p99 muda porque o
regime muda, e ele DEVE mudar para acompanhar. Estabiliza-lo seria remover a
adaptacao, nao melhorar a estimativa.

Trocar o estimador (med + k*MAD) tambem nao ajuda: com a mesma janela recente a
amplitude fica igual (2,05x contra 2,45x) mas a deteccao cai para 6/8-7/8 e o
custo sobe para 16-70 h/mes. O p99 sobre a janela recente ja e a melhor escolha.

Ver `drift_coerencia.py` para a leitura que substitui esta.

Uso:  PYTHONPATH=. python drift_desacopla.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from cabiunas_pdm import config as C, detector as DET
from ablacao import ScorerMax
from drift_baseline import roda, met, DF, STABLE, IX

FIT = DET.FIT_POINTS
FAM = {"t": C.TEMPERATURE_TAGS, "p": C.PRESSURE_TAGS}


def walkforward(n_calib: int | None, robusto: float | None = None):
    """n_calib = quantos pontos estaveis usar SO para a escala (None = todos).
    robusto = None usa p99; um numero k usa mediana + k*MAD do erro."""
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    n = len(IX)
    out = {"t": np.full(n, np.nan), "p": np.full(n, np.nan)}
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    escalas = {"t": [], "p": []}
    for i, m0 in enumerate(meses):
        passado = STABLE & (IX < m0)
        fit = DF.loc[passado, C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        m1 = meses[i + 1] if i + 1 < len(meses) else IX[-1] + pd.Timedelta("2min")
        s = (IX >= m0) & (IX < m1)
        if not s.any():
            continue
        w = DF.loc[s]
        # a amostra de CALIBRACAO e separada da de ajuste, e maior
        calib = DF.loc[passado, C.SENSOR_TAGS].dropna()
        if n_calib is not None:
            calib = calib.tail(n_calib)
        for c, tags in FAM.items():
            sc = ScorerMax().fit(fit[tags])          # PCA: janela recente
            # escala: o MESMO PCA pontuando o historico longo
            r = sc._raw_scores(calib[tags])[0]
            r = r[np.isfinite(r)]
            if robusto is None:
                esc = float(np.percentile(r, 99))
            else:
                med = float(np.median(r))
                mad = float(np.median(np.abs(r - med)) * 1.4826)
                esc = med + robusto * mad
            esc = max(esc, 1e-9)
            escalas[c].append(esc)
            out[c][s] = sc._raw_scores(w[tags])[0] / esc
        b = DET._spread_mancal(fit)
        med_sp[s] = float(b.median())
        mad_sp[s] = float((b - b.median()).abs().median() * 1.4826)
    est = {c: (np.array(v).max() / np.array(v).min() if len(v) > 1 else float("nan"))
           for c, v in escalas.items()}
    var = {c: (100 * np.median(np.abs(np.diff(v) / np.array(v)[:-1])) if len(v) > 2
               else float("nan")) for c, v in escalas.items()}
    return out["t"], out["p"], med_sp, mad_sp, est, var


def linha(rot, n_calib, robusto=None):
    t, p, ms, ds, est, var = walkforward(n_calib, robusto)
    met(roda(t, p, ms, ds), rot,
        f"   escala t: {est['t']:.2f}x amplitude, {var['t']:.0f}% var/mes")


if __name__ == "__main__":
    print("=" * 122)
    print("A) DESACOPLAR: PCA na janela de 20.000, ESCALA sobre historico maior (mesmo PCA)")
    print("=" * 122)
    linha("20.000  <- atual (acoplado)", FIT)
    for nc in (60_000, 120_000, None):
        linha(f"{'todo o passado' if nc is None else format(nc, ',d').replace(',', '.')}"
              f"{'':>{max(0, 12 - len(str(nc)))}}", nc)

    print()
    print("=" * 122)
    print("B) TROCAR O ESTIMADOR DE CAUDA: mediana + k*MAD no lugar do p99")
    print("   (um percentil extremo e o estimador de maior variancia; o robusto e mais estavel)")
    print("=" * 122)
    for k in (3.0, 5.0, 8.0):
        linha(f"med + {k:.0f}*MAD, calib 20.000", FIT, robusto=k)
    for k in (3.0, 5.0, 8.0):
        linha(f"med + {k:.0f}*MAD, calib total ", None, robusto=k)
