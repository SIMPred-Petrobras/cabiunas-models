#!/usr/bin/env python3
"""EXP 29 — O HARNESS COM OS 8 EVENTOS: replay causal estendido para antes de março de 2025 (o 27/02/2025 volta ao conjunto).

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Os EXP 04-27 avaliam a partir de 10/03/2025 e por isso 7 dos 8 trips-alvo (o de 27/02/2025 fica antes dos sinais causais, que só existem desde 05/03/2025 porque o CSV bruto começa em 01/01/2025 e a produção precisa de ~60 d de janela). A série do controle (ledger 'atual') começa em 08/01/2025 e dá, com os 8: 7/8, início 6, 9 FP, 0,807 FP/mês, 407 h fora.
O QUE SE FAZ. (i) dados de 30 s de set/2024 a fev/2025 dos xlsx (fuso +3 h = UTC; `scripts/constroi_raw30s_2024q4.py`) concatenados ANTES do CSV bruto (que tem prioridade a partir de 01/01/2025); (ii) o mesmo replay causal semanal de produção (`sinais_causais`, janela de 75 d, trips só até o corte), começando em 08/01/2025 (como o ledger do controle); cache novo em data_local/sinais_v2_8ev.parquet (o antigo fica intacto); (iii) as validações abaixo; (iv) `ci.prever` como caixa-preta sobre os sinais novos, avaliado a partir de 01/02/2025 com os 8 eventos.
VALIDADE (escrita antes; todas precisam passar para o harness de 8 eventos valer):
  V1 FONTE: a grade de produção (`preparar_grade`) calculada dos xlsx de jan-fev/2025 coincide com a calculada do CSV bruto no mesmo período: erro relativo mediano (entre tags) <= 1e-3.
  V2 CONTINUIDADE: os sinais novos coincidem com os do cache antigo (data_local/sinais_v2.parquet) de 01/04/2025 em diante: diferença absoluta máxima <= 1e-6 em t, p, sp e vb, e a máscara idêntica.
  V3 CONTROLE: o `prever` sobre os sinais novos reproduz o controle com 8 eventos: detecção 7/8, FP 9 (+-1) e a série de alarme coincide com a do controle em >= 97% dos instantes vigiados desde 01/02/2025.
EXPECTATIVAS: V1 passa ~70% (a interpolação do xlsx muda pouco o 2 min); V2 passa para t, p e sp ~75% e para o vb ~55% (a janela do vb é rolante e as semanas de março podem diferir); V3 passa ~60%; o evento de 27/02/2025 é detectado com início ~75%.
INTERPRETAÇÃO PREVISTA. Se V1-V3 passam, o harness passa a ter os 8 eventos (cache novo) e os resultados anteriores seguem válidos como recorte de 7. Se V2 falha só nas primeiras semanas de março (janela mais curta no cache antigo), a diferença é do cache antigo e não do novo: reportado. Se V1 falha, a junção xlsx + CSV não é fiel e o 27/02 não entra por essa via.
LIMITES. A junção usa xlsx 'interpolated' antes de 01/01/2025 e CSV bruto depois; a produção foi desenhada para o bruto; uma diferença de interpolação pode mudar o vb nas semanas iniciais; nenhum resultado anterior é refeito aqui.

Uso:  PYTHONPATH=src .venv/bin/python experiments/29_harness_com_oito_eventos.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); CACHE8 = ROOT / "sinais_v2_8ev.parquet"


def main():
    ci = _ci(); x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet")
    csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv); modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    print("V1 — FONTE (xlsx x CSV bruto, jan-fev/2025)", flush=True)
    ini, fim = pd.Timestamp("2025-01-05", tz="UTC"), pd.Timestamp("2025-02-25", tz="UTC")
    with contextlib.redirect_stdout(io.StringIO()):
        g1 = ci.preparar_grade(x.loc[ini:fim, raw.columns]); g2 = ci.preparar_grade(raw.loc[ini:fim])
    j = g1.index.intersection(g2.index); cols = [c for c in g1.columns if c in g2.columns and g1[c].dtype.kind == "f"]
    a, b = g1.loc[j, cols], g2.loc[j, cols]; ok = a.notna() & b.notna(); rel = ((a.where(ok) - b.where(ok)).abs() / (b.where(ok).abs() + 1e-9)).median()
    print(f"   {len(j)} instantes, {len(cols)} colunas; erro relativo mediano (mediana entre tags) {rel.median():.2e}; máximo entre tags {rel.max():.2e}; fração de NaN: xlsx {a.isna().mean().mean():.3f}, CSV {b.isna().mean().mean():.3f}", flush=True)
    v1 = rel.median() <= 1e-3
    S8 = sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=CACHE8, extra=x); S0 = pd.read_parquet(E5.CACHE)
    print(f"\nsinais novos: {S8.index[0]} a {S8.index[-1]} ({len(S8)} instantes); antigos: {S0.index[0]} a {S0.index[-1]}", flush=True)
    print("V2 — CONTINUIDADE (de 01/04/2025 em diante)", flush=True)
    j = S8.index.intersection(S0.index); j = j[j >= pd.Timestamp("2025-04-01", tz="UTC")]; v2 = True
    for c in ("t", "p", "sp", "vb"):
        d = (S8.loc[j, c] - S0.loc[j, c]).abs(); n_nan = int((S8.loc[j, c].isna() != S0.loc[j, c].isna()).sum()); print(f"   {c}: diferença absoluta máxima {np.nanmax(d):.2e} (mediana {np.nanmedian(d):.2e}); NaN diferentes: {n_nan}", flush=True); v2 &= bool(np.nanmax(d) <= 1e-6 and n_nan == 0)
    dm = int((S8.loc[j, "mask"].astype(bool) != S0.loc[j, "mask"].astype(bool)).sum()); print(f"   máscara diferente em {dm} instantes", flush=True); v2 &= dm == 0
    for ini_, fim_ in (("2025-03-05", "2025-04-01"),):
        jj = S8.index.intersection(S0.index); jj = jj[(jj >= pd.Timestamp(ini_, tz="UTC")) & (jj < pd.Timestamp(fim_, tz="UTC"))]
        print(f"   [março de 2025, o início do cache antigo] diferença máxima t {np.nanmax((S8.loc[jj,'t']-S0.loc[jj,'t']).abs()):.2e}, vb {np.nanmax((S8.loc[jj,'vb']-S0.loc[jj,'vb']).abs()):.2e}", flush=True)
    print("\nV3 — CONTROLE COM 8 EVENTOS", flush=True)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    proc = S8[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    res = decide_v2(proc, det); al = res["is_anomaly"].astype(bool); r = avalia(al.reindex(al_c.index).fillna(False), vig_c, reg, inicio_periodo=INICIO8)
    ref = avalia(al_c, vig_c, reg, inicio_periodo=INICIO8); v = vig_c.loc[INICIO8:]; conc = float((al.reindex(al_c.index).fillna(False).loc[INICIO8:][v] == al_c.loc[INICIO8:][v]).mean())
    print(f"   novo : {r.det}/{r.n_eventos} | início {r.inicio} | FP {r.fp} | FP/mês {r.fp_mes:.3f} | fora {r.horas_alarme_fora:.0f} h | totais {r.horas_alarme:.0f} h", flush=True)
    print(f"   ledger: {ref.det}/{ref.n_eventos} | início {ref.inicio} | FP {ref.fp} | FP/mês {ref.fp_mes:.3f} | fora {ref.horas_alarme_fora:.0f} h | totais {ref.horas_alarme:.0f} h | coincidência da série nos instantes vigiados: {100 * conc:.2f}%", flush=True)
    print("   por evento (detectado, início):", {k[:10]: (x_['det'], x_['inicio']) for k, x_ in r.por_evento.items()}, flush=True)
    v3 = r.det == 7 and abs(r.fp - 9) <= 1 and conc >= 0.97
    print(f"\nV1 {'PASSA' if v1 else 'FALHA'} | V2 {'PASSA' if v2 else 'FALHA'} | V3 {'PASSA' if v3 else 'FALHA'}", flush=True)


if __name__ == "__main__":
    main()
