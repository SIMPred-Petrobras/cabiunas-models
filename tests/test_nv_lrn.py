import numpy as np
import pandas as pd

from nv.causal import auditoria_do_futuro
from nv.lrn import lrn


def _serie(n=24 * 60 * 30, semente=0):
    idx = pd.date_range("2025-01-01", periods=n, freq="2min", tz="UTC"); rng = np.random.default_rng(semente)
    return pd.Series(1.0 + 0.2 * rng.standard_normal(n), index=idx), pd.Series(True, index=idx)


def test_lrn_tira_o_patamar_persistente_e_preserva_o_pico_novo():
    e, v = _serie(); e.iloc[10000:15000] += 3.0                     # patamar elevado e PERSISTENTE (>= 7 d depois some da referência? aqui dura 7 d)
    z = lrn(e, v, janela_h=240, guarda_h=24)
    assert z.iloc[10050:10100].mean() > 5                            # logo depois do degrau o desvio aparece (a referência é atrasada)
    e2, v2 = _serie(); e2.iloc[6000:10000] += 3.0                    # patamar de ~5,5 d que começa depois de haver referência
    z2 = lrn(e2, v2, janela_h=240, guarda_h=24)
    assert z2.iloc[6100:6200].mean() > 5 and z2.iloc[14000:14100].mean() < 1      # no início o desvio aparece; quando o patamar domina a referência, é absorvido


def test_lrn_causal():
    e, v = _serie(); d = pd.DataFrame({"e": e, "v": v.astype(float)})
    f = lambda x: lrn(x["e"], x["v"].astype(bool), janela_h=240, guarda_h=24).to_frame("z")
    assert auditoria_do_futuro(f, d, [d.index[12000], d.index[30000]])["passa"]
