import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from sota import protocolo as P


def _serie():
    idx = pd.date_range(P.INICIO, P.FIM, freq="2min"); vig = pd.Series(True, index=idx); al = pd.Series(False, index=idx)
    trips = [pd.Timestamp("2025-03-10", tz="UTC"), pd.Timestamp("2025-06-10", tz="UTC")]
    al.loc[trips[0] - pd.Timedelta(hours=60):trips[0] - pd.Timedelta(hours=10)] = True      # detecta, nasce antes da janela: lead censurado
    al.loc["2025-08-01 00:00":"2025-08-01 05:00"] = True                                     # um FP
    return al, vig, trips


def test_mede_lead_censurado_e_f1():
    al, vig, trips = _serie(); m = P.mede(al, vig, trips)
    assert (m["det"], m["inicio"], m["fp"], m["n_censurado"]) == (1, 0, 1, 1)
    assert m["lead_mediano"] == 48.0 and abs(m["f1"] - 0.5) < 1e-9          # precisão 1/2, recall 1/2


def test_compara_consigo_mesmo_nao_difere():
    al, vig, trips = _serie(); d = P.compara(al, al, vig, trips, n=200)
    assert d["d_det"] == 0 and not (d["difere_det"] or d["difere_fp"] or d["difere_horas"])
