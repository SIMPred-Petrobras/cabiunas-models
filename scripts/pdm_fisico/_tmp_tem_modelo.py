"""O modelo APRENDIDO faz trabalho, ou a estatistica robusta faz tudo?

t e p sao erro de reconstrucao de PCA -- modelo ajustado mensalmente, aprendido.
sp e vb sao z robusto contra mediana/MAD -- estatistica classica, nada aprendido.

Tira o PCA inteiro e ve o que sobra. Com (kb,kv) revarrido em cada ablacao, para
nao confundir "perdeu o modelo" com "perdeu a calibracao"."""
import sys; sys.path.insert(0, ".")
import pandas as pd, avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

paradas = paradas_reais_2h(); JAN = pd.Timedelta(hours=48)
jw = [(t-JAN, t) for t in alvo]
CACHE = {(kb, kv): partes(kb, kv) for kb in (1.1,1.3,1.5,1.7,2.0,2.4) for kv in (1.8,2.2,2.8)}

def melhor(usar, rotulo):
    top = None
    for (kb, kv), ON in CACHE.items():
        n = sum(ON[c].astype(int) for c in usar)
        v = pd.Series(n >= 2, index=idx) & mask
        obrig = [c for c in ("sp", "vb") if c in usar]
        if obrig:
            o = ON[obrig[0]].copy()
            for c in obrig[1:]: o = o | ON[c]
            v = v & o
        al = pos(v, n, REFRAT_H, DUR_MIN, False)
        eps = AV.episodios(al)
        if not eps: continue
        m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
        cl = classifica_regra_c(eps, paradas)
        n_fp = sum(1 for a,b,c,l in cl if c == "FP")
        h = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c == "FP")
        det = sum(any(a<=t1 and b>=t0 for a,b in eps) for t0,t1 in jw)
        if n_fp/meses > 1.0: continue
        if top is None or (det, -h/meses) > (top[0], -top[2]):
            top = (det, n_fp/meses, h/meses, m["lead_med"])
    if top is None:
        print(f"  {rotulo:<44} nada dentro do teto de 1 FP/mes"); return
    print(f"  {rotulo:<44} {top[0]}/8   {top[1]:6.3f} FP/mes   {top[2]:6.2f} h/mes   lead {top[3]:5.1f}h")

print("O QUE CADA PARTE CARREGA (limiares revarridos, teto de 1 FP/mes)")
print("=" * 96)
melhor(SIN,                "COMPLETO: PCA (t,p) + estatistica (sp,vb)")
melhor(["sp", "vb"],       "SO estatistica robusta -- sem o PCA")
melhor(["t", "p"],         "SO o PCA aprendido -- sem a estatistica")
print()
melhor(["t", "sp", "vb"],  "PCA de temperatura + estatistica")
melhor(["p", "sp", "vb"],  "PCA de pressao + estatistica")

print("\nEM QUANTOS DOS 8 EVENTOS CADA CANAL ESTA ACESO (ponto de producao)")
print("=" * 96)
ON = CACHE[(1.7, 2.2)]
ns = sum(ON[c].astype(int) for c in SIN)
v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
eps = AV.episodios(pos(v, ns, REFRAT_H, DUR_MIN, False))
tps = [(a, b) for a, b in eps if any(a <= t1 and b >= t0 for t0, t1 in jw)]
cont = {c: 0 for c in SIN}
for a, b in tps:
    jan = (idx >= a) & (idx <= b)
    for c in SIN:
        if ON[c].to_numpy()[jan].any(): cont[c] += 1
for c in SIN:
    tipo = "PCA aprendido" if c in ("t", "p") else "estatistica robusta"
    print(f"  {c:>4} ({tipo:<20}) aceso em {cont[c]}/8 deteccoes")
pca_algum = sum(1 for a, b in tps
                if (ON["t"].to_numpy()[(idx>=a)&(idx<=b)].any()
                    or ON["p"].to_numpy()[(idx>=a)&(idx<=b)].any()))
print(f"\n  deteccoes com AO MENOS UM canal de PCA aceso: {pca_algum}/8")
