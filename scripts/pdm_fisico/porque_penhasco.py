"""Por que a deteccao desaba a partir de 2000h?

Hipotese: a referencia e montada com tail(N) de amostras ESTAVEIS, e operacao
estavel e intermitente -- entao N grande faz a janela abranger muitos MESES de
calendario, atravessando regimes diferentes da maquina. A variancia da
referencia infla, o escore normalizado encolhe, e nada mais cruza o limiar:
o detector nao fica 'seletivo', fica CEGO.

Mede-se: (a) quantos dias de calendario a referencia abrange, (b) a escala
resultante (p99 do erro por sensor), (c) o escore tipico e o escore no pico
pre-evento -- se ambos encolhem juntos, e cegueira por escala, nao seletividade.
"""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
from cabiunas_pdm import config as C, detector as DET
from ablacao import canonico, mascara_pontuacao, ScorerMax

df = canonico(); stable = df["stable"].astype(bool); idx = df.index
mask = mascara_pontuacao(df)
EV = pd.Timestamp("2026-02-26 15:34", tz="UTC")   # trip de oleo perdido a partir de 2000h
m0 = pd.Timestamp("2026-02-01", tz="UTC")

print(f"{'fit_h':>6} {'amostras':>9} {'span_calendario':>16} {'p99_escala':>11} "
      f"{'escore_mediano':>15} {'pico_48h_pre_evento':>20} {'razao_pico/limiar':>18}")
for fh in [667., 1333., 2000., 2667., 3500.]:
    npts = int(fh*30)
    fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS].dropna().tail(npts)
    if len(fit) < npts//4:
        print(f"{fh:6.0f} {npts:9d}  referencia insuficiente"); continue
    span_d = (fit.index[-1] - fit.index[0]).total_seconds()/86400
    sc = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
    escala = float(np.nanmedian(sc.sens_p99_))
    w = df.loc[(idx >= m0) & (idx < EV + pd.Timedelta(hours=2))]
    s = pd.Series(sc.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy(), index=w.index)
    s = s.where(mask.reindex(s.index).fillna(False))
    ew = s.ewm(halflife=pd.Timedelta("1h"), times=s.index).mean()
    pre = ew[(ew.index >= EV - pd.Timedelta(hours=48)) & (ew.index < EV)]
    limiar = DET.THR_FAM * 1.3
    print(f"{fh:6.0f} {npts:9d} {span_d:14.0f}d {escala:11.4f} {ew.median():15.3f} "
          f"{pre.max():20.3f} {pre.max()/limiar:18.2f}")
print(f"\n(limiar = THR_FAM * k_base = {DET.THR_FAM*1.3:.2f}; razao >= 1 significa que cruza)")
