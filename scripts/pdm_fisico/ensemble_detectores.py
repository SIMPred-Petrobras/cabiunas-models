#!/usr/bin/env python3
"""ENSEMBLE NO NIVEL DO DETECTOR -- a combinacao que nunca foi testada.

O QUE JA FOI TESTADO E FALHOU: transplantar PECAS entre as maquinas. O canal vb
na varredura do Francisco (960 configs) e a nossa camada de decisao nos canais do
Diego (128 configs). Nos dois casos nada transfere.

O QUE NUNCA FOI TESTADO: combinar as SERIES DE ALARME PRONTAS. E diferente --
nao mexe em nenhuma maquina, tem ZERO parametro livre (uniao e intersecao), e usa
o unico ativo que quatro times independentes produziram: erros possivelmente
DESCORRELACIONADOS.

A evidencia de que podem ser: nos erramos nas duas pontas da distribuicao de lead
(1,3 a 194,9 h) e ele no meio (3,8 a 43,2 h); ele perde 0 eventos na regua de
inicio e nos perdemos 17/03 e 29/04; nos custamos 0,52 FP/mes e ele 2,88.

  UNIAO      -> maximiza deteccao, soma o custo
  INTERSECAO -> minimiza custo, exige concordancia
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import REFRAT_H, DUR_MIN
from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c

DIEGO = "/home/thallys/Documents/projeto-petrobras/wt-diego/canais/merged.parquet"
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0


def mede(al, rot):
    eps = AV.episodios(al)
    banda, leads, ini, depe = 0, [], 0, 0
    for t in alvo:
        c = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
        n = [a for a, _ in eps if t - JAN <= a <= t]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if n: ini += 1
        elif any(a <= t and b >= t - JAN for a, b in eps): depe += 1
    m = AV.avalia(al, alvo, mask); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    return dict(rot=rot, banda=banda, ini=ini, det=m["det"], fp=nfp/meses,
                h=h/meses, lead=np.mean(leads) if leads else np.nan, eps=len(eps))


# --- a nossa serie (ponto publicado) -----------------------------------------
nosso = alarme()

# --- a serie dele, remontada e reamostrada para a nossa grade de 2 min --------
m = pd.read_parquet(DIEGO)
# o indice dele e tz-NAIVE e o nosso e UTC; sem localizar, o reindex devolve tudo
# False e a serie dele some. Os dois lados ja estao em UTC (confirmado em
# verdade.py: o export do PI vem em UTC), entao e localizar, nao converter.
m.index = m.index.tz_localize("UTC")
ns = sum(m[c].astype(int) for c in ("temperatura", "vibracao", "oleo", "alarme"))
v = (ns >= 2)
# filtro de duracao de 45 min e refratario de 48 h, como ele faz
g = (~v).cumsum()[v]
dur = v[v].groupby(g).transform("size") * 30.0 / 60.0        # grade dele e 30 s
vf = v.copy(); vf[v] = dur >= 45.0
al_d = pd.Series(False, index=m.index); bloq = None
for a, b in AV.episodios(vf, gap_h=2.0):
    if bloq is not None and a <= bloq: continue
    al_d.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
# reamostra para a nossa grade: max no bin de 2 min
dele = al_d.resample("2min").max().reindex(idx, fill_value=False).fillna(False).astype(bool)
dele = dele & mask

print("AS DUAS SERIES NA NOSSA REGUA E NA NOSSA GRADE")
print("=" * 104)
print(f"{'':<28}{'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>9}{'h/mes':>8}{'lead':>8}{'eps':>6}")
print("-" * 104)
linhas = [mede(nosso, "nosso (publicado)"), mede(dele, "Diego (reamostrado)")]
for r in linhas:
    print(f"{r['rot']:<28}{r['banda']:5d}/8{r['ini']:6d}/8{r['det']:5d}/8"
          f"{r['fp']:9.3f}{r['h']:8.1f}{(r['lead'] if np.isfinite(r['lead']) else 0):7.1f}h{r['eps']:6d}")

print("-" * 104)
for rot, s in [("UNIAO  (nosso OU dele)", nosso | dele),
               ("INTERSECAO (nosso E dele)", nosso & dele)]:
    r = mede(s, rot)
    print(f"{r['rot']:<28}{r['banda']:5d}/8{r['ini']:6d}/8{r['det']:5d}/8"
          f"{r['fp']:9.3f}{r['h']:8.1f}{(r['lead'] if np.isfinite(r['lead']) else 0):7.1f}h{r['eps']:6d}")

print("\n\nPOR EVENTO -- quem pega o que")
print("=" * 104)
print(f"{'evento':>12} | {'nosso':>22} | {'Diego':>22} | union ganha?")
print("-" * 104)
en, ed = AV.episodios(nosso), AV.episodios(dele)
for t in alvo:
    def q(eps):
        i = [a for a, _ in eps if t - JAN <= a <= t]
        if i:
            l = (t - max(i)).total_seconds()/3600
            return f"nasce {l:5.1f}h" + ("  [banda]" if TMIN <= l <= TMAX else "")
        return "de pe" if any(a <= t and b >= t - JAN for a, b in eps) else "--"
    a, b = q(en), q(ed)
    marca = "  <<<" if ("banda" in b and "banda" not in a) or ("banda" in a and "banda" not in b) else ""
    print(f"{t:%d/%m/%Y} | {a:>22} | {b:>22} |{marca}")


# ---------------------------------------------------------------------------
# A uniao BOOLEANA piora porque os nossos episodios longos se fundem com os
# curtos dele e o inicio combinado sai da janela. Mas em operacao nao se faz OU
# de dois detectores: eles sao DUAS LINHAS DE ALARME que o operador ve em
# paralelo. Medir assim -- uniao no nivel de EVENTO, custo somado.
print("\n\nDUAS LINHAS EM PARALELO -- uniao no nivel de EVENTO")
print("=" * 104)
def banda_de(eps, t):
    i = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
    return (t - max(i)).total_seconds()/3600 if i else None

cob = {}
for t in alvo:
    ln, ld = banda_de(en, t), banda_de(ed, t)
    cob[t] = (ln, ld)
n_nosso = sum(1 for v in cob.values() if v[0] is not None)
n_dele  = sum(1 for v in cob.values() if v[1] is not None)
n_uni   = sum(1 for v in cob.values() if v[0] is not None or v[1] is not None)
n_int   = sum(1 for v in cob.values() if v[0] is not None and v[1] is not None)
leads_u = [min(x for x in v if x is not None) for v in cob.values()
           if any(x is not None for x in v)]

mn, md = mede(nosso, "n"), mede(dele, "d")
print(f"{'':<34}{'banda':>8}{'FP/mes':>10}{'h/mes':>9}{'lead med':>10}")
print("-" * 104)
print(f"{'so o nosso':<34}{n_nosso:6d}/8{mn['fp']:10.3f}{mn['h']:9.1f}"
      f"{np.mean([v[0] for v in cob.values() if v[0] is not None]):9.1f}h")
print(f"{'so o dele':<34}{n_dele:6d}/8{md['fp']:10.3f}{md['h']:9.1f}"
      f"{np.mean([v[1] for v in cob.values() if v[1] is not None]):9.1f}h")
print(f"{'AS DUAS em paralelo':<34}{n_uni:6d}/8{mn['fp']+md['fp']:10.3f}"
      f"{mn['h']+md['h']:9.1f}{np.mean(leads_u):9.1f}h")
print(f"{'   (so quando AMBAS concordam)':<34}{n_int:6d}/8{'--':>10}{'--':>9}{'--':>10}")
print("-" * 104)
print(f"  eventos so o nosso pega : "
      f"{', '.join(t.strftime('%d/%m') for t, v in cob.items() if v[0] is not None and v[1] is None)}")
print(f"  eventos so o dele pega  : "
      f"{', '.join(t.strftime('%d/%m') for t, v in cob.items() if v[1] is not None and v[0] is None)}")
print(f"  ambos                   : "
      f"{', '.join(t.strftime('%d/%m') for t, v in cob.items() if v[0] is not None and v[1] is not None)}")
print(f"  nenhum                  : "
      f"{', '.join(t.strftime('%d/%m') for t, v in cob.items() if v[0] is None and v[1] is None)}")

# sobreposicao dos falsos positivos: os erros sao mesmo descorrelacionados?
cn = classifica_regra_c(en, paradas); cd = classifica_regra_c(ed, paradas)
fn = [(a, b) for a, b, k, _ in cn if k == "FP"]
fd = [(a, b) for a, b, k, _ in cd if k == "FP"]
sobrep = sum(1 for a, b in fn if any(x <= b and y >= a for x, y in fd))
print(f"\n  FP nossos: {len(fn)}   FP dele: {len(fd)}   que se sobrepoem no tempo: {sobrep}")
print(f"  -> {'erros DESCORRELACIONADOS' if sobrep <= 1 else 'erros parcialmente correlacionados'}")
