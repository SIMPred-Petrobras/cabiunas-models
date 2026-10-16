#!/usr/bin/env python3
"""EXP 27 — DE ONDE VEM O FP 'REMOVÍVEL'? (máquina ligada sem combustão ou regime pós-parada longa/manutenção?)

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO. EXP 13b, 25c e 26: as 'melhorias' que apareceram (escalada_idade_h = 192, consenso de vizinhança) removem os mesmos FP do ponto publicado: o de 15/10/2025 (135 h) e o de 09/01/2026 (34 h) (e, no E192, o de 26/10/2025, 112 h). Eu levantei a hipótese de que fossem a máquina 'ligada sem combustão' do regime ago-out/2025; a memória 'running-a-e-a-mascara' já diz que o trecho sem combustão (RUNNING_A ligada com a maioria dos sinais dizendo parada) é de 19 a 24/08/2025, fora destas datas. Duas hipóteses concorrentes, testadas aqui SEM mexer no detector:
  H-COMB  os FP removíveis ocorrem com a máquina ligada mas sem combustão (T5 baixo, ou a tag oscilando).
  H-MANUT os FP removíveis ocorrem no regime depois de uma PARADA LONGA (>= 72 h, p. ex. manutenção), quando o ponto de operação mudou ('drift-abrupto-pos-manutencao') e o bundle do mês ainda é o do regime antigo.
O QUE SE FAZ. Os 9 episódios de FP do ponto publicado (caminho exato; sinais do EXP 05) mais os 5 episódios que cobrem trips. Para cada um: (i) mediana do T5_AVG_A no episódio e fração do tempo com T5 < 400 °C; (ii) número de transições da RUNNING_A por dia dentro do episódio; (iii) dias desde o fim da última parada longa (>= 72 h) até o início do episódio; (iv) o canal que mais tempo fica aceso (nível A: sinal >= base x k_nivel_a) e a fração de aceso de cada canal. Linha de base: a fração dos instantes vigiados do período avaliado que está a <= 30 dias de uma parada longa (a taxa de base de 'estar no regime pós-parada').
CRITÉRIOS (fixados antes). H-COMB se confirma para um episódio se a mediana do T5 < 400 °C ou a fração de T5 < 400 °C >= 30% ou >= 4 transições da RUNNING_A por dia; confirma para o conjunto se >= 50% dos FP removíveis (15/10/2025, 26/10/2025, 09/01/2026) a satisfazem. H-MANUT se confirma para um episódio se o início está a <= 30 dias de uma parada longa; confirma para o conjunto se >= 2 dos 3 FP removíveis a satisfazem E a fração de FP (de todos os 9) nessa condição é >= 2x a taxa de base (enriquecimento).
EXPECTATIVAS: H-COMB refutada (<= 1 dos 3 FP removíveis com T5 normal e tag estável) ~85%; H-MANUT confirmada para o conjunto ~45% (o de 15/10 e o de 26/10 vêm depois de uma parada longa de outubro? e o de 09/01 depois de uma de dezembro?); enriquecimento >= 2x ~40%; o canal dominante nos FP removíveis é o vb ~35% e o p ~35%.
INTERPRETAÇÃO PREVISTA. Se H-MANUT confirma, o FP removível é um FP DE REGIME PÓS-MANUTENÇÃO e o remédio é de apresentação/governança (um período de graça ou retreino extra depois de parada longa, já avaliado como 'cabeça de aquecimento' e 'recentragem reprovada'), não de limiar. Se nenhuma confirma, os dois FP são só os dois FP, sem explicação física identificada, e o ganho de removê-los é só amostra.
LIMITES. 9 + 5 episódios; uma parada longa é definida por mim (>= 72 h); a taxa de base e o enriquecimento são descritivos (poucos eventos); nada se adota.

Uso:  PYTHONPATH=src .venv/bin/python experiments/27_de_onde_vem_o_fp_removivel.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6; E5 = E25.E5
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG

INICIO = E25.INICIO; GL = E25.GL; JAN = pd.Timedelta(hours=48)
REMOVIVEIS = ("2025-10-15", "2025-10-26", "2026-01-09")


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual")
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    pub = E25.alarme({}); eps = episodios_de_serie(pd.Series(pub, index=idx).loc[INICIO:])
    r = g["RUNNING_A"] > 0.5; t5 = g["T5_AVG_A"]; ch = r.ne(r.shift()).cumsum()
    # paradas longas (>= 72 h): instante de FIM da parada (religamento)
    seg = pd.DataFrame({"lig": r, "ch": ch}).groupby("ch").agg(lig=("lig", "first"), ini=("lig", lambda s: s.index[0]), fim=("lig", lambda s: s.index[-1]), n=("lig", "size"))
    paradas = seg[(~seg.lig) & (seg.n * 2 / 60 >= 72)]; fins = list(paradas.fim + pd.Timedelta("2min"))
    print(f"paradas longas (>= 72 h) no período: {[(str(a.date()), round(n * 2 / 60)) for a, n in zip(paradas.ini, paradas.n) if a >= pd.Timestamp('2025-01-01', tz='UTC')]}", flush=True)
    desde = lambda t: min([(t - f).total_seconds() / 86400 for f in fins if f <= t], default=np.nan)
    v = GL["vig"].loc[INICIO:]; base = float(np.mean([desde(t) <= 30 for t in v.index[v.to_numpy()][::30]]))
    print(f"taxa de base: fração dos instantes vigiados a <= 30 dias do fim de uma parada longa = {base:.2f}\n", flush=True)
    kA = det["k_nivel_a"]; bs = det["base"]; linhas = []
    for a, b in eps:
        tr = [t for t in reg if a <= t and b >= t - JAN]; tipo = ("trip " + str(tr[0])[:10]) if tr else "FP"
        sl = slice(a, b); x5 = t5.loc[sl]; rr = r.loc[sl]; trans = int((rr != rr.shift()).sum() - 1); dias = max((b - a).total_seconds() / 86400, 1e-9)
        aceso = {c: float((proc[c].loc[sl] / (bs[c] * kA[c]) >= 1).mean()) for c in ("t", "p", "sp", "vb")}
        linhas.append(dict(inicio=f"{a:%Y-%m-%d %H:%M}", dur_h=round((b - a).total_seconds() / 3600 + 2 / 60, 1), tipo=tipo, T5_mediana=round(float(x5.median())), frac_T5_baixo=round(float((x5 < 400).mean()), 2), transicoes_dia=round(trans / dias, 1), dias_desde_parada_longa=round(desde(a), 1), **{f"aceso_{c}": round(v_, 2) for c, v_ in aceso.items()}))
    T = pd.DataFrame(linhas); print(T.to_string(index=False), flush=True)
    fp = T[T.tipo == "FP"]; rem = fp[fp.inicio.str[:10].isin(REMOVIVEIS)]
    comb = lambda row: row.T5_mediana < 400 or row.frac_T5_baixo >= 0.3 or row.transicoes_dia >= 4; manut = lambda row: row.dias_desde_parada_longa <= 30
    print(f"\nH-COMB: FP removíveis que a satisfazem: {int(sum(comb(row) for row in rem.itertuples()))} de {len(rem)} (limite >= 50%) | entre todos os FP: {int(sum(comb(row) for row in fp.itertuples()))} de {len(fp)}", flush=True)
    nm = int(sum(manut(row) for row in rem.itertuples())); tf = float(np.mean([manut(row) for row in fp.itertuples()]))
    print(f"H-MANUT: FP removíveis a <= 30 dias de uma parada longa: {nm} de {len(rem)} | fração de todos os FP nessa condição: {tf:.2f} contra a taxa de base {base:.2f} (enriquecimento {tf / max(base, 1e-9):.1f}x; limite >= 2x e >= 2 dos 3)", flush=True)


if __name__ == "__main__":
    main()
