#!/usr/bin/env python3
"""AUDITORIA dos testes desta rodada -- e ela achou erro.

Pedida depois de a sessao ja ter produzido tres conclusoes erradas que so
apareceram na verificacao. Tres frentes:

 1. CONTROLE: cada script reproduz o v2 publicado no seu baseline? Se nao, tudo
    que foi medido em cima esta deslocado.  -> TODOS OK (5/8, 6/8, 8/8, 0,344).

 2. CONTAMINACAO no `lead_fisico.py`: a referencia de cada evento vai de t-37d a
    t-7d, e a janela de busca de t-7d a t. Cinco dos oito eventos tem OUTRO TRIP
    numa dessas faixas -- e o 11/04 tem o trip de 07/04 dentro da propria janela
    de busca. Refeito so com os 3 eventos limpos (27/02, 04/11, 26/02), os leads
    dao 144,4 / 168,0 / 78,7 h, parecidos com os contaminados: a conclusao
    original (nao ha sobra a destravar) se mantem, agora sobre base menor.

 3. ERRO REAL, e muda uma conclusao. `pericia_fp_v2.py` contou o nivel A como
    "tres canais acenderam em ALGUM momento da primeira hora". O correto e tres
    canais SIMULTANEOS -- que e o que o voto exige. Com o criterio certo:

        entrada    FP   TP   NEUTRO          (reportado antes: so B = 3 FP, 2 TP)
        so A        0    3     0
        so B        3    3     1
        A+B         1    4     6

    E as forcas dos que nascem so por B:

        FP:  170,6 ·   6,2 ·  3,5
        TP:   29,2 · 146,8 ·  1,2   <-- o 26/04, que a contagem errada escondia

    O TP de 26/04 nasce com forca 1,2, MENOR que os dois FP que o piso pretendia
    matar. Nao existe janela que separe os grupos: a separacao que motivou o piso
    de forca era artefato da contagem. Fisher da p = 0,245.

Uso:  PYTHONPATH=. python auditoria_testes.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import reproduz, TMIN_BANDA
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
JAN = pd.Timedelta(hours=48); par = paradas_reais_2h()

def met(fin):
    eps = AV.episodios(fin); det = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, par); mes = det["horas_op"]/730.0
    nfp = sum(1 for _,_,k,_ in cls if k=="FP")
    hfp = sum((b-a).total_seconds()/3600 for a,b,k,_ in cls if k=="FP")
    ban = sum(1 for t in alvo if any(t-JAN<=a<=t-pd.Timedelta(hours=TMIN_BANDA) for a,_ in eps))
    ini = sum(1 for t in alvo if any(t-JAN<=a<=t for a,_ in eps))
    return (ban, ini, det["det"], round(nfp/mes,3), round(hfp/mes,1), len(eps))

REF = met(reproduz(v2=True)[0])
print(f"REFERENCIA (publica_clearml.reproduz): banda {REF[0]}/8 inicio {REF[1]}/8 "
      f"det {REF[2]}/8 FP {REF[3]} h {REF[4]} eps {REF[5]}\n")

# cada script constroi seu proprio baseline -- todos tem de dar o mesmo
import margem_no_v2 as MV
b1 = met(MV.decide(MV.voto_v2(None, "")))
print(f"{'margem_no_v2.decide(voto_v2)':<40}{str(b1):<44}{'OK' if b1==REF else '*** DIVERGE'}")

import corte_por_estabilizacao as CE
b2 = met(CE.decide(CE.voto_base()))
print(f"{'corte_por_estabilizacao.voto_base':<40}{str(b2):<44}{'OK' if b2==REF else '*** DIVERGE'}")

import score_continuo as SCo
b3 = met(SCo.decide(SCo.voto_v2(None, "")))
print(f"{'score_continuo (baseline)':<40}{str(b3):<44}{'OK' if b3==REF else '*** DIVERGE'}")

import piso_no_nascimento as PN
b4 = met(PN.base)
print(f"{'piso_no_nascimento.base':<40}{str(b4):<44}{'OK' if b4==REF else '*** DIVERGE'}")
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import mask, idx, alvo

BUSCA = pd.Timedelta(days=7); REF = pd.Timedelta(days=30)
print("=== A) lead_fisico: a REFERENCIA de cada evento contem outro trip? ===")
print("   (a referencia vai de t-37d a t-7d; outro trip ali dentro a contamina)\n")
ruim = 0
for t in alvo:
    r0, r1 = t - BUSCA - REF, t - BUSCA
    dentro = [x for x in alvo if r0 <= x <= r1 and x != t]
    # e trips DENTRO da janela de busca (t-7d a t) tambem sao um problema:
    na_busca = [x for x in alvo if t - BUSCA <= x < t]
    fl = ""
    if dentro: fl += f"   <<< referencia contem {len(dentro)} trip(s): " + ", ".join(f"{x:%d/%m}" for x in dentro)
    if na_busca: fl += f"   <<< JANELA DE BUSCA contem {len(na_busca)}: " + ", ".join(f"{x:%d/%m}" for x in na_busca)
    if dentro or na_busca: ruim += 1
    print(f"   {t:%Y-%m-%d}  ref [{r0:%d/%m} .. {r1:%d/%m}]{fl}")
print(f"\n   {ruim} de {len(alvo)} eventos tem contaminacao")

print("\n=== B) corte_por_estabilizacao: o corte cai onde deveria? ===")
import corte_por_estabilizacao as CE
base = CE.decide(CE.voto_base())
cort = CE.encerra_estabilizado(base, 6, 0.20)
# todo episodio cortado deve comecar no MESMO instante e terminar antes
eb = AV.episodios(base); ec = AV.episodios(cort)
ini_b = {a for a, _ in eb}; ini_c = {a for a, _ in ec}
print(f"   episodios base {len(eb)}, cortado {len(ec)}, mesmos inicios: {ini_b == ini_c}")
mais_longo = [(a, b) for a, b in ec
              if any(a == x and b > y for x, y in eb)]
print(f"   episodios que FICARAM MAIS LONGOS apos o corte: {len(mais_longo)} (deve ser 0)")
dur_b = {a: (b-a).total_seconds()/3600 for a, b in eb}
dur_c = {a: (b-a).total_seconds()/3600 for a, b in ec}
viol = [a for a in dur_c if dur_c[a] > dur_b.get(a, 1e9) + 1e-6]
print(f"   violacoes de duracao: {len(viol)} (deve ser 0)")

print("\n=== C) piso_no_nascimento: 'so nivel B' esta sendo detectado certo? ===")
import piso_no_nascimento as PN
from margem_no_v2 import A, B
from publica_clearml import SIN, VOTO_LO
nA = sum(A[c].astype(int) for c in SIN)
vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
n_so_b = 0
for a, b in AV.episodios(PN.base):
    jan = idx[(idx >= a) & (idx <= min(b, a + pd.Timedelta(hours=1)))]
    if not bool(vA.loc[jan].any()):
        n_so_b += 1
print(f"   episodios que nascem so por B: {n_so_b} de {len(AV.episodios(PN.base))}")
print(f"   (a pericia reportou 5: 3 FP + 2 TP -- confere?)")

print("\n=== D) score_continuo: o voto contínuo entra no MESMO pos-processamento? ===")
import score_continuo as SCo
s = SCo.score("soma")
th = float(np.nanpercentile(s[mask].dropna(), 95))
v = pd.Series(s >= th, index=idx).fillna(False) & mask
print(f"   voto continuo p95: {int(v.sum())} instantes ({100*v.mean():.1f}%)")
print(f"   voto do v2       : {int(SCo.voto_v2(None,'').sum())} instantes "
      f"({100*SCo.voto_v2(None,'').mean():.1f}%)")
print(f"   -> a comparacao e justa so se o pos-processamento for o mesmo: "
      f"decide() e o mesmo objeto? {SCo.decide is CE.decide}")
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import SIN, VOTO_LO, VOTO_HI
from margem_no_v2 import A, B, FORCA, decide, voto_v2
from regra_c_com_teto import classifica

print("=== C) CONTAGEM CORRETA: quem nasce so pelo nivel B ===")
print("   criterio: o nivel A precisa de 3 canais SIMULTANEOS, nao 3 que")
print("   acenderam em algum momento da primeira hora.\n")
nA = sum(A[c].astype(int) for c in SIN)
nB = sum(B[c].astype(int) for c in SIN)
vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"])
base = decide(voto_v2(None, ""))
cls = classifica(AV.episodios(base), None)

tab = {"so A": {"FP":0,"TP":0,"NEUTRO":0}, "so B": {"FP":0,"TP":0,"NEUTRO":0},
       "A+B": {"FP":0,"TP":0,"NEUTRO":0}}
det = []
for a, b, k, d in cls:
    jan = idx[(idx >= a) & (idx <= min(b, a + pd.Timedelta(hours=1)))]
    ea, eb = bool(vA.loc[jan].any()), bool(vB.loc[jan].any())
    ent = "A+B" if (ea and eb) else ("so A" if ea else "so B")
    tab[ent][k] += 1
    det.append((a, k, ent, float(FORCA.loc[jan].max())))
print(f"{'entrada':<8}{'FP':>5}{'TP':>5}{'NEUTRO':>8}")
for e in ("so A", "so B", "A+B"):
    print(f"{e:<8}{tab[e]['FP']:>5}{tab[e]['TP']:>5}{tab[e]['NEUTRO']:>8}")
print("\n   forcas no nascimento, entre os que nascem SO por B:")
for a, k, e, f in det:
    if e == "so B":
        print(f"      {a:%Y-%m-%d %H:%M}  {k:<7} forca {f:8.1f}")

from scipy.stats import fisher_exact
t2 = [[tab["so B"]["FP"], tab["so A"]["FP"]+tab["A+B"]["FP"]],
      [tab["so B"]["TP"], tab["so A"]["TP"]+tab["A+B"]["TP"]]]
print(f"\n   tabela so-B x resto / FP x TP: {t2}  -> Fisher p = {fisher_exact(t2)[1]:.3f}")

print("\n\n=== A) LEAD FISICO com referencia LIMPA ===")
print("   exclui do calculo qualquer evento cuja referencia ou janela de busca")
print("   contenha outro trip -- sobram os nao contaminados.\n")
import lead_fisico as LF
BUSCA, REF = LF.BUSCA, LF.REF
limpos = [t for t in alvo
          if not [x for x in alvo if x != t and (t-BUSCA-REF <= x <= t-BUSCA or t-BUSCA <= x < t)]]
print(f"   eventos nao contaminados: {len(limpos)} de {len(alvo)} -> "
      + ", ".join(f"{t:%d/%m/%y}" for t in limpos))
for t in limpos:
    l = {c: LF.inicio_desvio(c, t) for c in LF.SIN}
    fis = np.nanmax(list(l.values()))
    print(f"      {t:%Y-%m-%d}  " +
          "  ".join(f"{c}={l[c]:.0f}h" if np.isfinite(l[c]) else f"{c}=--" for c in LF.SIN)
          + f"   max {fis:.1f}h")
