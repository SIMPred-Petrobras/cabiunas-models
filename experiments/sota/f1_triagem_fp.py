#!/usr/bin/env python3
"""SOTA FASE 1 — TRIAGEM DOS 9 FP DO PCA v2: FP x causa provável, cruzando com o diário de alarmes da planta. Tag ClearML `sota-baseline`.

Fixado em 22/10/2026, ANTES de rodar. DESCRITIVO. DECLARADO: boa parte dos fatos já é conhecida (EXP 38: nenhum FP tem alarme analógico no disparo, e durante o episódio
2 dos 9 têm — 15/10/2025 e 08/04/2026, TC382_03_A HI/HIHI; EXP 60: 6 dos 9 nascem na borda da vigilância, 6,5 h após a partida; 21/10: o FP de 26/10/2025 é seguido de
parada e 20 h de manutenção). Esta tabela CONSOLIDA esses fatos numa regra única, fixada antes, com as tags que puxaram cada alarme (H6).
PARA CADA FP (episódio da base atual, 01/02/2025 a 21/04/2026, que não intersecta janela de trip):
  ALARME DA PLANTA: ativações (ACT) do diário em [nascimento − 2 h, fim] com a máquina em operação (RUNNING_A > 0,5 no instante); analógicas = HI, HIHI, LOLO, UNDER, OVER;
    discretas = CFN (só contexto). O diário está em UTC (EXP 38).
  PARTIDA: horas desde a última partida até o nascimento; BORDA = nasce até 30 min depois da abertura da vigilância.
  PARADA: a máquina para em [fim, fim + 2 h]; MANUTENÇÃO DEPOIS: modo de manutenção (HSX >= 24 h, regra do pacote) começa em [fim, fim + 7 d].
  INSTRUMENTO: alguma das 3 tags do topo (H6) acusada pela regra de "sensor travado" do pacote (`_sensores_travados`, 13 semanas até o fim do episódio).
  CARGA: L na notificação (`nv.prioridade`). ORDENS: ordens/operações de manutenção do SAP da turbina com data em [nascimento − 7 d, fim + 7 d] (auxiliar; datas planejadas; só onde as
    planilhas estão — elas NÃO vão para o worker, então lá a coluna sai "n/d").
CAUSA PROVÁVEL (precedência fixada antes): (1) ALARME REAL SEM TRIP se houve ACT analógica com a máquina operando; senão (2) INSTRUMENTO se uma tag do topo está travada;
senão (3) PARTIDA/PARADA se nasce na borda ou termina em parada; senão (4) SEM EXPLICAÇÃO. Todas as colunas ficam na tabela, não só a causa.
EXPECTATIVA (com os fatos já conhecidos): alarme real 2; partida/parada 4 a 6; instrumento 0 ou 1; sem explicação 1 a 3.
LIMITES: 9 episódios; o diário pode ter alarmes suprimidos; coincidência não é causa; as ordens do SAP têm datas planejadas.

Uso local:  PYTHONPATH=src:experiments .venv/bin/python -m sota.f1_triagem_fp
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp f1_triagem_fp --tags sota-baseline --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv import io as nvio
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2
from nv.prioridade import carga_nascimento
from sota import protocolo as P
from sota import h6_atribuicao as H6

ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "f1_triagem_fp"
ALARMES = Path(os.environ.get("ALARMES_CSV", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv"))
SAP = Path(os.environ.get("SAP_DIR", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models"))
ANALOG = {"HI", "HIHI", "LOLO", "UNDER", "OVER"}; DUAS_H = pd.Timedelta(hours=2)


def sap():
    """Ordens e operações de manutenção (datas e texto breve), se os arquivos existirem."""
    linhas = []
    for arq, cols in (("Historico Ordens Turbina A.xlsx", ("Data-base iníc.", "Data-base fim")), ("Histórico Operações TSC33003A.xlsx", ("Dta.início real", "1ª data início", "Dt.real fim"))):
        f = SAP / arq
        if not f.exists():
            continue
        d = pd.read_excel(f)
        for c in cols:
            if c in d:
                dt = pd.to_datetime(d[c], errors="coerce", dayfirst=True)
                linhas += [(t.tz_localize("UTC"), str(x)) for t, x in zip(dt, d["Texto breve"]) if pd.notna(t)]
    return linhas


def main():
    t0 = time.time(); ci = _ci(); SAIDA.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; trips = P.trips_alvo(carrega_registro())
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        R = decide_v2(proc, det)
    top = H6.SAIDA / ("h6_top3_teste.csv" if H6.TESTE else "h6_top3.csv")
    if not top.exists():
        print("H6 ainda não calculada aqui: rodando a atribuição (replay por tag)...", flush=True); H6.main()
    A = pd.read_csv(top)
    g = nvio.carrega_grade(); mk = nvio.mascaras(g); op = mk["ligada"]; est = mk["estavel"]; t5 = g["T5_AVG_A"].where(est)
    partidas = g.index[mk["partida"].to_numpy(bool)]; vig = proc["mask"]; v = vig.to_numpy(); aberturas = vig.index[np.flatnonzero(v[1:] & ~v[:-1]) + 1]
    paradas = g.index[np.flatnonzero(op.to_numpy()[:-1] & ~op.to_numpy()[1:]) + 1]
    manut = ci._manutencoes(g)
    d = pd.read_csv(ALARMES); d["t"] = pd.to_datetime(d["Data da Ocorrência"]).dt.tz_localize("UTC"); d = d[d["Status"] == "ACT/UNACK"]
    d["operando"] = op.reindex(d.t.dt.floor("2min")).fillna(False).to_numpy()
    ordens = sap(); tem_sap = any((SAP / f).exists() for f in ("Historico Ordens Turbina A.xlsx", "Histórico Operações TSC33003A.xlsx")); fim_diario = d.t.max(); linhas = []
    ini = H6.INICIO_REPLAY + pd.Timedelta(days=7) if H6.TESTE else P.INICIO
    for a, b in episodios_de_serie(R["is_anomaly"].astype(bool).loc[ini:P.FIM]):
        if any(a <= t and b >= t - JANELA for t in trips):
            continue
        loc = (a - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M"); h6 = A[A.inicio_local == loc]
        w = d[(d.t >= a - DUAS_H) & (d.t <= b) & d.operando]; an = w[w["Condição do Alarme"].isin(ANALOG)]; di = w[w["Condição do Alarme"] == "CFN"]
        ult_p = partidas[partidas <= a]; k = aberturas.searchsorted(a, side="right") - 1
        borda = bool(k >= 0 and a - aberturas[k] <= pd.Timedelta(minutes=30)); para = bool(((paradas >= b) & (paradas <= b + DUAS_H)).any())
        man = [(x, y) for x, y in manut if b <= x <= b + pd.Timedelta(days=7)]
        tops = [s.split(" ")[0] for s in h6[["top1", "top2", "top3"]].iloc[0].fillna("").tolist() if s] if len(h6) else []
        cols = [c for c in g.columns if c.replace("954005_624_", "") in tops]
        trav = ci._sensores_travados(g.loc[b - pd.Timedelta(weeks=13):b], est.loc[b - pd.Timedelta(weeks=13):b], cols) if cols else []
        o = sorted({x for t, x in ordens if a - pd.Timedelta(days=7) <= t <= b + pd.Timedelta(days=7)})
        causa = ("ALARME REAL SEM TRIP" if len(an) else "INSTRUMENTO" if trav else "PARTIDA/PARADA" if (borda or para) else "SEM EXPLICAÇÃO")
        linhas.append(dict(inicio_local=loc, dur_h=round((b - a).total_seconds() / 3600, 1), causa_provavel=causa,
                           alarmes_analogicos="; ".join(sorted({f"{x.replace('954005_624_', '')}:{c}" for x, c in zip(an["Tag Alarme"], an["Condição do Alarme"])})) or "-",
                           n_discretos=len(di), discretos="; ".join(sorted({x.replace("954005_624_", "") for x in di["Tag Alarme"]}))[:80] or "-",
                           h_desde_partida=round((a - ult_p[-1]).total_seconds() / 3600, 1) if len(ult_p) else np.nan, borda=borda, termina_em_parada=para,
                           manutencao_ate_7d=(f"{man[0][0]:%d/%m} ({(man[0][1] - man[0][0]).total_seconds() / 3600:.0f} h)" if man else "-"),
                           sensor_travado="; ".join(x["tag"].replace("954005_624_", "") for x in trav) or "-", carga_L=round(carga_nascimento(t5, a), 2),
                           canais=h6.canais_nascimento.iloc[0] if len(h6) else "", top1=h6.top1.iloc[0] if len(h6) else "", top2=h6.top2.iloc[0] if len(h6) else "",
                           top3=h6.top3.iloc[0] if len(h6) else "", ordens_sap_7d=(" | ".join(o)[:120] or "-") if tem_sap else "n/d", coberto_pelo_diario=a <= fim_diario))
    T = pd.DataFrame(linhas)
    if H6.TESTE:
        print(f"TESTE: encanamento ok ({len(T)} FP na janela curta; colunas {list(T.columns)}; saída suprimida, nada gravado) em {time.time() - t0:.0f} s", flush=True); return
    T.to_csv(SAIDA / "triagem_fp.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
    print(f"TRIAGEM DOS {len(T)} FP DO PCA v2 (base atual)\n", flush=True)
    print(T[["inicio_local", "dur_h", "causa_provavel", "alarmes_analogicos", "n_discretos", "h_desde_partida", "borda", "termina_em_parada", "manutencao_ate_7d", "sensor_travado", "carga_L"]].to_string(index=False), flush=True)
    print("\n" + T[["inicio_local", "canais", "top1", "top2", "top3"]].to_string(index=False), flush=True)
    print("\n" + T[["inicio_local", "discretos", "ordens_sap_7d"]].to_string(index=False), flush=True)
    print(f"\nRESUMO: {T.causa_provavel.value_counts().to_dict()} | tempo {time.time() - t0:.0f} s; tabela em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
