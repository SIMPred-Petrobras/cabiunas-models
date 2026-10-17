#!/usr/bin/env python3
"""EXP 39b — A TERCEIRA VIA COM EVENTOS DE ATIVAÇÃO DO DIÁRIO (falha de instrumento e suprimento de gás) EM JANELA ANTERIOR AO DISPARO.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR. (Corrige o EXP 39: lá o ESTADO reconstruído do diário ficou ativo em 84% e 94% do tempo, inutilizável.)

FAMÍLIAS (as do EXP 39, fixadas pelo texto da `Descrição Alarme`), só as ativações ACT/UNACK de 2025-2026: F_falha = descrição contém 'Falha' (308 ativações; 282 são o PI_0319 'Falha PI Gás Motor p/Partida', 8 do PDIT_0305 'Falha PDI Vazam. Selo Prim.', e poucas de PI_0307, PDI_0301, PDI_0317, PI_0308, PDI_0338, PI_0339, PDI_0302, PDIT_6240304); F_gas = descrição contém 'Gás Comb' ou 'Gás Linha' sem 'Falha' (281 ativações; 233 do PI_0315 'Pressão Baixa Gás Comb.' e 48 do PDI_0302 'PrDif.Bx.Gás Linha Balanc.'). Como 92% do F_falha é o PI_0319, ele é reportado também à parte, SEM virar uma terceira família de teste.
O QUE SE FAZ. Para os 18 episódios do publicado (8 eventos; 9 FP, 9 trips) cobertos pelo diário: se a família teve >= 1 ativação nas L horas ANTES do nascimento, com L em {6, 24} (duas janelas fixadas; corrige-se por 2 famílias x 2 janelas = 4 testes: o p exigido é 0,10/4 = 0,025 para chamar de enriquecimento). PERMUTAÇÃO: o vetor de ativações da família deslocado circularmente por 2.000 deslocamentos aleatórios de >= 14 dias: P(ao acaso >= o número observado de FP). Probabilidade de base: a fração dos instantes vigiados com >= 1 ativação da família nas L horas anteriores.
CRITÉRIOS (fixados antes). ENRIQUECIDA se >= 3 dos 9 FP têm a ativação, <= 1 dos 9 trips, e p da permutação <= 0,025. O veto no episódio (descartar o episódio se há ativação da família nas L h antes do nascimento) é ÚTIL se, enriquecida, a detecção fica >= 7/8 e os FP <= 7; NOCIVO se perde detecção sem cortar FP.
EXPECTATIVAS: probabilidade de base do F_falha em 24 h entre 25% e 55% ~65% (o PI_0319 dispara perto de uma vez a cada dois dias); F_falha NÃO enriquecida nos FP ~75%; F_gas NÃO enriquecida ~80%; algum veto ÚTIL ~5%.
INTERPRETAÇÃO PREVISTA. Se nenhuma família se enriquece, a informação do diário (instrumento declarado em falha e baixa pressão de gás) não separa FP de acerto neste conjunto, e a terceira via com informação da planta fica sem sinal utilizável; o que sobra é perguntar à operação. Se alguma se enriquece, é um veto fisicamente justificado a confirmar na sombra.
LIMITES. 18 episódios; o diário termina em 18/04/2026; o PI_0319 'Falha' domina o F_falha e pode ser só a sequência de partida; 4 testes corrigidos; nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/39b_veto_por_evento_da_planta.py
"""
from __future__ import annotations
import contextlib, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48)
LOG = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv")


def main():
    d = pd.read_csv(LOG); d["t"] = pd.to_datetime(d["Data da Ocorrência"]).dt.tz_localize("UTC"); fim_log = d.t.max(); a = d[(d.Status == "ACT/UNACK") & (d.t >= "2025-01-01")].copy(); ds = a["Descrição Alarme"].astype(str)
    fam = {"F_falha": a[ds.str.contains("Falha", case=False)], "F_falha sem PI_0319 (descritivo)": a[ds.str.contains("Falha", case=False) & ~a["Tag Alarme"].str.contains("PI_0319")], "F_gas": a[ds.str.contains("Gás Comb|Gás Linha", case=False, regex=True) & ~ds.str.contains("Falha", case=False)]}
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); eps = [(x, y) for x, y in episodios_de_serie(al.loc[INICIO8:]) if x <= fim_log]; tipo = [("trip" if any(x <= t and y >= t - JAN for t in reg) else "FP") for x, y in eps]
    pub = avalia(al.reindex(al_c.index).fillna(False), vig.reindex(al_c.index).fillna(False), reg, inicio_periodo=INICIO8); print(f"PUBLICADO: {pub.det}/8, FP {pub.fp}; {len(eps)} episódios cobertos ({tipo.count('FP')} FP, {tipo.count('trip')} trip)", flush=True)
    vm = proc["mask"].to_numpy(bool) & (idx >= INICIO8) & (idx <= fim_log); nasc = np.array([idx.searchsorted(x) for x, y in eps]); rng = np.random.default_rng(20261016); n = len(idx); i_ini = idx.searchsorted(INICIO8); i_fim = idx.searchsorted(fim_log); L_ = i_fim - i_ini
    for nome, ev in fam.items():
        imp = np.zeros(n); np.add.at(imp, np.clip(idx.searchsorted(ev.t), 0, n - 1), 1.0); print(f"\n{nome}: {len(ev)} ativações", flush=True)
        for Lh in (6, 24):
            Lp = Lh * 30; cs = np.cumsum(imp); tem = lambda arr, ii: (arr[ii] - arr[np.maximum(ii - Lp, 0)]) > 0
            obs = tem(cs, nasc); kfp = int(sum(1 for x, t in zip(obs, tipo) if x and t == "FP")); ktr = int(sum(1 for x, t in zip(obs, tipo) if x and t == "trip")); base = float(tem(cs, np.flatnonzero(vm)).mean()); sh = []
            for _ in range(2000):
                off = int(rng.integers(14 * 720, L_ - 14 * 720)); i2 = np.zeros(n); i2[i_ini:i_fim] = np.roll(imp[i_ini:i_fim], off); c2 = np.cumsum(i2); o2 = tem(c2, nasc); sh.append(int(sum(1 for x, t in zip(o2, tipo) if x and t == "FP")))
            p = float(np.mean(np.array(sh) >= kfp)); enr = kfp >= 3 and ktr <= 1 and p <= 0.025
            print(f"   {Lh:2d} h antes: FP com a ativação {kfp} de {tipo.count('FP')} | trips {ktr} de {tipo.count('trip')} | base {100 * base:.1f}% dos instantes vigiados | permutação P(>= {kfp} FP) = {p:.3f} (média ao acaso {np.mean(sh):.2f}) -> {'ENRIQUECIDA' if enr else 'não enriquecida'} | episódios com ela: {[f'{x:%m-%d} {t}' for (x, y), t, o in zip(eps, tipo, obs) if o]}", flush=True)
            if not nome.endswith("(descritivo)"):
                al2 = al.copy()
                for (x, y), o in zip(eps, obs):
                    if o:
                        al2.loc[x:y] = False
                r = avalia(al2.reindex(al_c.index).fillna(False), vig.reindex(al_c.index).fillna(False), reg, inicio_periodo=INICIO8); print(f"      veto no episódio: {r.det}/8, FP {r.fp} (publicado {pub.det}/8, FP {pub.fp}) -> {'ÚTIL' if enr and r.det >= 7 and r.fp <= 7 else 'NOCIVO' if r.det < pub.det and r.fp >= pub.fp else 'NEUTRO/INCONCLUSIVO'}", flush=True)


if __name__ == "__main__":
    main()
