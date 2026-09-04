"""Em que percentil o limiar fixo cai, MES A MES, no sinal que o detector ve.
O agregado (p88) e o de um mes quieto (p99,99) discordam -- entao a pergunta
"em que percentil o limiar esta" nao tem UMA resposta, tem uma distribuicao."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pos_processamento import EW, mask, idx, alvo
from publica_clearml import SIN, BASE, K, T0

E = {c: EW[c].where(mask) for c in SIN}
meses = pd.date_range(T0, idx.max(), freq="MS", tz="UTC")
linhas = []
for i, m0 in enumerate(meses):
    m1 = meses[i+1] if i+1 < len(meses) else idx.max() + pd.Timedelta("2min")
    selm = (idx >= m0) & (idx < m1) & mask
    if selm.sum() < 500:
        continue
    r = {"mes": m0}
    for c in SIN:
        v = E[c][selm].dropna()
        r[c] = 100.0*(v < BASE[c]*K[c]).mean() if len(v) else np.nan
    r["n"] = int(selm.sum())
    linhas.append(r)
T = pd.DataFrame(linhas)

print(f"{'mes':>9} {'dias':>6} " + "".join(f"{c:>10}" for c in SIN))
for r in T.itertuples():
    print(f"{r.mes:%Y-%m} {r.n*2/60/24:6.1f} " +
          "".join(f"{getattr(r,c):9.2f}%" for c in SIN))

print(f"\n{'':>16} " + "".join(f"{c:>10}" for c in SIN))
for nome, f in (("mediana", np.nanmedian), ("minimo", np.nanmin), ("maximo", np.nanmax)):
    print(f"{nome:>16} " + "".join(f"{f(T[c]):9.2f}%" for c in SIN))
print(f"{'meses acima de p99':>16} " +
      "".join(f"{int((T[c]>=99).sum()):>6}/{len(T):<3}" for c in SIN))
