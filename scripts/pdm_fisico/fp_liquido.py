#!/usr/bin/env python3
"""O nosso FP LIQUIDO -- a metrica em que o "5/8 a 0,43" foi medido.

DE ONDE VEIO A PISTA. O notebook 06_serie_completa_anomalias.ipynb imprime
`fp_por_mes` E `fp_por_mes_liquido` ("descontando os que coincidem com alarme ativo"),
e um best_trial.json em cache do ClearML tem a linha inteira:

    eventos_detectados 5   eventos_total 8   dias_avaliados 350,5
    fp_episodios 11   fp_explicados 5   fp_por_mes 0,94   fp_por_mes_liquido 0,51

Conferindo: 11/0,94 = 11,70 meses, e (11-5)/11,70 = 0,513. Bate com 0,51. Entao a
formula e, sem ambiguidade:

    fp_liquido = (fp_episodios - fp_explicados) / meses

onde `fp_explicados` sao os falsos positivos que **coincidem com alarme de planta ja
ativo** -- o argumento e que ali o operador ja estava olhando, entao aquele alarme nosso
nao criou trabalho novo.

POR QUE ISSO IMPORTA. O nosso 1,033 FP/mes e BRUTO. O numero deles e LIQUIDO. Comparar
os dois foi erro nosso: sao definicoes diferentes de custo, e o liquido e sempre menor.
Este script computa o nosso liquido pela mesma formula, com a mesma tabela de alarme.

FUSO. `alarmes_selecionados_turbina_a.csv` vem de export record_*, que pela nota do
CLAUDE.md precisa de correcao de fuso. Em vez de assumir o sinal, o script CALIBRA:
procura o deslocamento que faz os alarmes de nivel baterem com as 8 falhas de
falhas.csv (que foram derivadas dessa mesma tabela). O deslocamento sai medido, nao
suposto -- essa confusao ja custou 6 h de erro ao projeto duas vezes.

DUAS LEITURAS DE "COINCIDE". Sem o codigo do `cabiunas_pdm` (o pacote se perdeu), a
regra exata de `fp_explicados` nao esta disponivel, entao reporto as duas leituras
plausiveis e a faixa entre elas:
  estrita -- ha alarme de planta ATIVO em algum instante do episodio;
  frouxa  -- ha ABERTURA de alarme dentro do episodio ou ate `margem` antes dele
             (o `exclude_alarm_h=1,0` da configuracao deles sugere margem de 1 h).
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
JAN = pd.Timedelta(hours=48)
ALARMES = ("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/"
           "analise_cabiunas/dados/alarmes_selecionados_turbina_a.csv")
MARGEM_H = 1.0
REF = dict(det=5, tot=8, fp_eps=11, fp_expl=5, meses=11.70,
           bruto=0.94, liquido=0.51, lead=21.4)


def alarme_nosso():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, ns, REFRAT_H, DUR_MIN, False)


def carrega_alarmes(desloc_h):
    a = pd.read_csv(ALARMES, usecols=["Data da Ocorrência", "Tag Alarme",
                                      "Descrição Alarme", "Status"])
    a.columns = ["t", "tag", "desc", "status"]
    a["t"] = (pd.to_datetime(a["t"], errors="coerce").dt.tz_localize("UTC")
              + pd.Timedelta(hours=desloc_h))
    return a.dropna(subset=["t"]).sort_values("t")


def calibra_fuso():
    """Acha o deslocamento que alinha as aberturas de alarme com as 8 falhas."""
    melhor, best = None, -1
    for d in (-3, 0, 3):
        a = carrega_alarmes(d)
        on = a[a.status.astype(str).str.startswith("ACT")]
        n = sum(1 for t in alvo
                if ((on.t >= t - pd.Timedelta(hours=1)) &
                    (on.t <= t + pd.Timedelta(minutes=30))).any())
        print(f"    deslocamento {d:+d} h -> {n}/8 falhas com abertura de alarme "
              f"na janela [-1 h, +30 min]")
        if n > best:
            melhor, best = d, n
    return melhor


def intervalos_ativos(a):
    """(inicio, fim) por tag: de uma abertura ACT ate o proximo INACT do mesmo tag."""
    iv = []
    for tag, g in a.groupby("tag"):
        aberto = None
        for t, st in zip(g.t, g.status.astype(str)):
            if st.startswith("ACT") and aberto is None:
                aberto = t
            elif st.startswith("INACT") and aberto is not None:
                iv.append((aberto, t)); aberto = None
        if aberto is not None:
            iv.append((aberto, g.t.iloc[-1]))
    return iv


if __name__ == "__main__":
    al = alarme_nosso()
    m = AV.avalia(al, alvo, mask)
    eps = AV.episodios(al)
    jw = [(t - JAN, t) for t in alvo]
    fps = [(a, b) for a, b in eps
           if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
    meses = m["horas_op"] / 730.0
    print(f"nosso ponto de operacao: {m['det']}/8, {len(eps)} episodios, "
          f"{len(fps)} falsos positivos em {meses:.2f} meses")
    print(f"  FP BRUTO = {len(fps)}/{meses:.2f} = {len(fps)/meses:.3f} FP/mes\n")

    print("calibracao de fuso da tabela de alarme:")
    d = calibra_fuso()
    print(f"  -> adotado {d:+d} h\n")
    a = carrega_alarmes(d)
    a = a[a.desc.astype(str).str.contains("TC_33003A", na=False)]
    on = a[a.status.astype(str).str.startswith("ACT")]
    iv = intervalos_ativos(a)
    print(f"tabela de alarme: {len(a):,} linhas do TC_33003A, {len(on):,} aberturas, "
          f"{len(iv):,} intervalos ativos\n")

    print("=" * 96)
    print("QUAIS DOS NOSSOS FALSOS POSITIVOS COINCIDEM COM ALARME DE PLANTA")
    print("=" * 96)
    mg = pd.Timedelta(hours=MARGEM_H)
    print(f"{'inicio do FP':>17} {'dur':>7} {'estrita':>9} {'frouxa':>8}  alarmes no episodio")
    n_est = n_fro = 0
    for a_, b_ in fps:
        estrita = any(s <= b_ and e >= a_ for s, e in iv)
        dentro = on[(on.t >= a_ - mg) & (on.t <= b_)]
        frouxa = len(dentro) > 0
        n_est += estrita; n_fro += frouxa
        tags = ",".join(sorted(set(dentro.tag.astype(str)))[:3])
        print(f"{a_:%d/%m/%Y %H:%M} {(b_-a_).total_seconds()/3600:6.1f}h "
              f"{'SIM' if estrita else '-':>9} {'SIM' if frouxa else '-':>8}  {tags[:46]}")

    print("\n" + "=" * 96)
    print("O NOSSO NUMERO NAS DUAS DEFINICOES DE CUSTO")
    print("=" * 96)
    print(f"{'':<34} {'FP':>5} {'expl.':>6} {'FP/mes':>9}")
    print(f"  {'BRUTO (o que vinhamos citando)':<32} {len(fps):>5} {'-':>6} "
          f"{len(fps)/meses:>9.3f}")
    print(f"  {'LIQUIDO, leitura estrita':<32} {len(fps):>5} {n_est:>6} "
          f"{(len(fps)-n_est)/meses:>9.3f}")
    print(f"  {'LIQUIDO, leitura frouxa':<32} {len(fps):>5} {n_fro:>6} "
          f"{(len(fps)-n_fro)/meses:>9.3f}")

    print("\n" + "=" * 96)
    print("COMPARACAO COM O TRIAL DE REFERENCIA (best_trial.json em cache)")
    print("=" * 96)
    r = REF
    print(f"{'':<26} {'det':>6} {'FP bruto':>10} {'FP liquido':>11} {'lead':>7}")
    print(f"  {'trial de referencia':<24} {r['det']}/{r['tot']:<4} {r['bruto']:>10.2f} "
          f"{r['liquido']:>11.2f} {r['lead']:>6.1f}h")
    lo = (len(fps)-n_fro)/meses; hi = (len(fps)-n_est)/meses
    print(f"  {'nosso ponto':<24} {m['det']}/8   {len(fps)/meses:>10.2f} "
          f"{min(lo,hi):>5.2f}-{max(lo,hi):<5.2f} {m['lead_med']:>6.1f}h")
    print(f"\n  O liquido do trial de referencia desconta {r['fp_expl']}/{r['fp_eps']} "
          f"= {r['fp_expl']/r['fp_eps']*100:.0f}% dos episodios.")
    print(f"  O nosso desconta {n_est}/{len(fps)} ({n_est/len(fps)*100:.0f}%) na leitura "
          f"estrita e {n_fro}/{len(fps)} ({n_fro/len(fps)*100:.0f}%) na frouxa.")
