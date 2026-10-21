#!/usr/bin/env python3
"""EXP 60 — NOTIFICAÇÃO ADIADA PARA O ALARME DE BORDA: o episódio que nasce até 30 min depois de a vigilância abrir vira "atenção pós-partida" e só vira ALARME se durar 24 h.

PRÉ-REGISTRO (21/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp60.py`), branch exp/modos-de-falha.

POR QUÊ (medido em 21/10, `fp-duracao-e-partida-21out`). Nos 18 episódios do controle desde 01/02/2025, 6 dos 9 FP nascem na BORDA (6,5 h após a partida, 28 min depois de a
máscara abrir — o degrau mínimo), contra 2 dos 9 episódios de trip; e os 2 de trip são os episódios "travados" de 140 h e 648 h, cujos trips também são pegos por episódios
posteriores. Dos 6 FP de borda, 4 duram menos de 1 dia (14 a 20 h). É prática padrão de gestão de alarmes (ISA-18.2, EEMUA 191: alarme por estado, supressão em transitório
de partida/parada) não tratar como alarme o que acende na transição. As regras de borda anteriores (blackout longo, EWMA vigiado, regimes, transição observada, veto k=4)
VETAVAM ou escondiam o episódio e perdiam as detecções de regime longo; esta só ADIA a notificação — o episódio longo continua virando alarme.
ATENÇÃO, DECLARADO: no controle atual eu JÁ VI o resultado (FP 9 -> 5, detecção igual). Ali a medida não é evidência; a evidência está nas OUTRAS bases (G168 e B800, cujas bordas
eu não olhei, e o retreino n = 19.000), na vizinhança de X, no controle "24 h para todos" e, depois, na sombra.

REGRA (fixada antes; só pós-processa a série de alarme já produzida — o detector não muda):
  EPISÓDIO = o de produção (`ci._episodios`, une trechos a menos de 2 h). BORDA = o episódio nasce até 30 min depois da última abertura da vigilância (subida da máscara: fim do
  blackout de partida, retorno do T5 > 300 ou volta de dado). Episódio de borda com duração < X -> "atenção pós-partida" (sai do alarme); com duração >= X -> alarme a partir de
  nascimento + X. X = 24 h (um dia de operação); vizinhança 12 e 36 h.
  CONTROLE "X PARA TODOS": a mesma exigência aplicada a TODOS os episódios — mede se o efeito é da borda ou de persistência genérica (o EXP 43 já mostrou que persistência longa
  global perde detecção).
BASES: atual (ledger do controle), G168 e B800 (ledgers das variantes da sombra) e atual com retreino n = 19.000 (t e p do harness fiel, `escores_sub34`, pela decisão de produção).
MEDIDAS: detecção de 8, início, banda, FP, FP/mês, horas totais e fora, antecedência por trip (e quanto a regra atrasa a notificação), episódios rebaixados a atenção.
CRITÉRIOS (fixados antes):
  GANHO = na base atual com X = 24: FP cai >= 3, detecção igual, início e banda caem no máximo 1; E em >= 2 das 3 outras bases FP cai >= 2 com detecção igual; E com X = 12 e 36 a
  detecção da base atual fica igual; E o controle "X para todos" perde detecção (senão o efeito não é da borda).
  PIOR = detecção cai >= 1 na base atual com X = 24. EQUIVALENTE no resto.
EXPECTATIVAS: base atual com X = 24: FP 9 -> 5 e detecção 7 (~95%, já visto — não conta); FP cai >= 2 com detecção igual em >= 2 das 3 outras bases (~55%); início e banda
inalterados na base atual (~85%); X = 12 igual ao original na base atual (~80%: nenhum FP de borda dura menos de 12 h); X = 36 igual a X = 24 (~80%); "24 h para todos" perde >= 2
detecções na base atual (~90%); algum trip tem a notificação atrasada >= 12 h em alguma base (~30%).
INTERPRETAÇÃO PREVISTA: GANHO -> candidata da v2.1 na CAMADA DE NOTIFICAÇÃO (não no detector) e, por ser pós-processamento da série, aplicável por fora aos ledgers da sombra
sem tocar no congelado; a validação de verdade é prospectiva. EQUIVALENTE ou PIOR -> a borda fica só como rótulo de apresentação ("alarme nascido na partida").
LIMITES: 8 eventos; o X e a janela de 30 min foram escolhidos depois de ver os 9 FP do controle (por isso as outras bases e a vizinhança); 4 bases não são independentes (mesmo
dado); nada se adota sem a sombra.
VERIFICAÇÕES: V1 sem a regra, cada base reproduz as suas medidas de referência (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12); V2 a mesma reconstrução da base n = 19.000,
feita com n = 20.000, coincide com o ledger do controle em >= 99,9% dos instantes vigiados; V3 a regra com X = 0 devolve a série original.

Uso local (encanamento, sem tabelas):  EXP60_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/60_borda_adiada.py
No worker:  PYTHONPATH=src python scripts/worker_exp60.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, substitui

TESTE = bool(os.environ.get("EXP60_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC")
BORDA = pd.Timedelta(minutes=30); XS = (12, 24, 36); X0 = 24
REF = {"atual": (7, 9), "G168": (8, 14), "B800": (8, 12)}
GL = {}


def regra(al, vig, X, so_borda=True):
    """Pós-processa a série de alarme: episódio de borda (ou todos) com duração < X sai; com duração >= X vira alarme só a partir de nascimento + X."""
    ci = GL["ci"]; a = al.copy(); v = vig.reindex(al.index).fillna(False).to_numpy(bool)
    aberturas = al.index[np.flatnonzero(v[1:] & ~v[:-1]) + 1]; toc = []
    if X <= 0:
        return a, toc
    for ini, fim in ci._episodios(al):
        k = aberturas.searchsorted(ini, side="right") - 1
        borda = k >= 0 and (ini - aberturas[k]) <= BORDA
        if so_borda and not borda:
            continue
        if (fim - ini) < pd.Timedelta(hours=X):
            a.loc[ini:fim] = False; toc.append((ini, fim, "atenção"))
        else:
            a.loc[ini:ini + pd.Timedelta(hours=X) - pd.Timedelta(minutes=2)] = False; toc.append((ini, fim, f"alarme +{X} h"))
    return a, toc


def mede(al, vig):
    r = avalia(al.reindex(GL["idx"]).fillna(False), vig, GL["trips"], inicio_periodo=INICIO8)
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora,
                lead={k[:10]: v["lead_h"] for k, v in r.por_evento.items()}, ev={k[:10]: int(v["det"]) for k, v in r.por_evento.items()})


def base_n(ci, det, proc, n):
    px = proc.copy()
    for c in ("t", "p"):
        s = pd.read_parquet(ROOT / "escores_sub34" / f"{c}_A_{n}.parquet")["s"]
        px = substitui(px, c, s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna()))
    with contextlib.redirect_stdout(io.StringIO()):
        return decide_v2(px, det)["is_anomaly"].astype(bool)


def main():
    t0 = time.time(); ci = _ci(); GL["ci"] = ci; ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    GL["trips"] = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    B = {}
    for v in ("atual", "G168", "B800"):
        al, vig = carrega_controle(v); B[v] = (al, vig)
    GL["idx"] = B["atual"][0].index
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    B["atual n=19000"] = (base_n(ci, det, proc, 19000).reindex(GL["idx"]).fillna(False), B["atual"][1])
    a20 = base_n(ci, det, proc, 20000).reindex(GL["idx"]).fillna(False); vg = B["atual"][1].reindex(GL["idx"]).fillna(False)
    v2 = float((a20[vg.loc[INICIO8:].reindex(GL["idx"]).fillna(False)] == B["atual"][0][vg.loc[INICIO8:].reindex(GL["idx"]).fillna(False)]).mean())
    ok["V2"] = v2 >= 0.999
    print(f"V2 — reconstrução com n=20.000 (escores_sub34) x ledger do controle: coincidência {100 * v2:.2f}% dos instantes vigiados -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    for v, (dr, fr) in REF.items():
        m = mede(*B[v]); ok[f"V1_{v}"] = (m["det"], m["fp"]) == (dr, fr)
        print(f"V1 [{v}] sem a regra: {m['det']}/8, FP {m['fp']} (referência {dr}/8, FP {fr}) -> {'PASSA' if ok[f'V1_{v}'] else 'FALHA'}", flush=True)
    a0, _ = regra(*B["atual"], 0); ok["V3"] = bool((a0 == B["atual"][0]).all())
    print(f"V3 — regra com X = 0 devolve a série original -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    m19 = mede(*B["atual n=19000"]); print(f"   base n=19000 sem a regra: {m19['det']}/8, FP {m19['fp']} (informativo)", flush=True)
    R = {}
    for v, (al, vig) in B.items():
        R[(v, "original", 0)] = (mede(al, vig), [])
        for X in XS:
            a, toc = regra(al, vig, X); R[(v, "borda", X)] = (mede(a, vig), toc)
        a, toc = regra(al, vig, X0, so_borda=False); R[(v, "todos", X0)] = (mede(a, vig), toc)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(R, B, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(R, B, ok, t0, escreve=True)


def relata(R, B, ok, t0, escreve):
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    print("\nRESULTADOS (det/8 | início | banda | FP | FP/mês | horas | fora | rebaixados a atenção)", flush=True)
    for v in B:
        for regra_, X in (("original", 0), ("borda", 12), ("borda", 24), ("borda", 36), ("todos", 24)):
            L, toc = R[(v, regra_, X)]
            nome = "original" if regra_ == "original" else (f"borda, X={X} h" if regra_ == "borda" else "24 h para TODOS (controle)")
            print(f"   {v:14s} {nome:28s} | {L['det']}/8 | {L['inicio']} | {L['banda']} | FP {L['fp']:2d} | {L['fp_mes']:.3f} | {L['horas']:5.0f} h | {L['fora']:4.0f} h | {sum(1 for t in toc if t[2] == 'atenção')}", flush=True)
        print("", flush=True)
    print("EPISÓDIOS TOCADOS (borda, X = 24 h) e ANTECEDÊNCIA de cada trip (h; original -> com a regra)", flush=True)
    for v in B:
        L0 = R[(v, "original", 0)][0]; L1, toc = R[(v, "borda", 24)]
        print(f"   {v}: " + "; ".join(f"{loc(a)} ({(b - a).total_seconds() / 3600:.0f} h) -> {o}" for a, b, o in toc), flush=True)
        fh = lambda x: "—" if x is None else f"{x:.1f}"
        print("      antecedência: " + " | ".join(e[8:10] + "/" + e[5:7] + " " + fh(L0["lead"][e]) + "->" + fh(L1["lead"][e]) for e in L0["lead"]), flush=True)
    print("\nCLASSIFICAÇÃO", flush=True)
    o = R[("atual", "original", 0)][0]; b = R[("atual", "borda", X0)][0]
    p1 = (o["fp"] - b["fp"] >= 3) and b["det"] == o["det"] and (o["inicio"] - b["inicio"] <= 1) and (o["banda"] - b["banda"] <= 1)
    outras = [v for v in B if v != "atual"]
    p2 = sum((R[(v, "original", 0)][0]["fp"] - R[(v, "borda", X0)][0]["fp"] >= 2) and R[(v, "borda", X0)][0]["det"] == R[(v, "original", 0)][0]["det"] for v in outras)
    p3 = all(R[("atual", "borda", X)][0]["det"] == o["det"] for X in (12, 36))
    p4 = R[("atual", "todos", X0)][0]["det"] < o["det"]
    cl = "GANHO" if (p1 and p2 >= 2 and p3 and p4) else "PIOR" if b["det"] < o["det"] else "EQUIVALENTE"
    print(f"   base atual (X=24): FP {o['fp']} -> {b['fp']}, det {o['det']} -> {b['det']}, início {o['inicio']} -> {b['inicio']}, banda {o['banda']} -> {b['banda']} -> {'ok' if p1 else 'não'}", flush=True)
    print(f"   outras bases com FP -2 e detecção igual: {p2}/3 -> {'ok' if p2 >= 2 else 'não'} | vizinhança X=12/36 sem perda de detecção: {'ok' if p3 else 'não'} | '24 h para todos' perde detecção: {'sim' if p4 else 'NÃO'}", flush=True)
    print(f"   -> {cl}{' | META DO USUÁRIO (FP <= 6 e det >= 6)' if (b['fp'] <= 6 and b['det'] >= 6) else ''}", flush=True)
    if escreve:
        pd.DataFrame([dict(base=v, regra=r_, X=X, **{k: L[k] for k in ('det', 'inicio', 'banda', 'fp', 'fp_mes', 'horas', 'fora')}, rebaixados=sum(1 for t in toc if t[2] == 'atenção'))
                      for (v, r_, X), (L, toc) in R.items()]).to_csv(ROOT / "exp60_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
