#!/usr/bin/env python3
"""Prioridade pela carga sobre os ledgers da sombra: só LÊ o ledger, o CSV e o registro de trips; escreve numa pasta à parte. O pacote e o ledger não mudam.

HIPÓTESES PROSPECTIVAS, PRÉ-REGISTRADAS EM 22/10/2026, ANTES DE HAVER DADO PROSPECTIVO (o período da sombra começa em 22/04/2026 e o export de lá em diante ainda não chegou):
  H-CARGA      (estática, EXP 62): o episódio cuja carga na notificação (L, `nv.prioridade`) fica abaixo de 0,5 é atenção, não alarme.
  H-CARGA-DIN  (dinâmica, EXP 65): o mesmo, mas o episódio de carga baixa é promovido a alarme quando a carga passa de 0,5 enquanto ele está aceso.
  Parâmetros fixos: θ 0,5, T5 estável, referência de 30 d, janela de 2 h. Nenhum se reescolhe depois de ver dado prospectivo.
QUANDO DECIDIR: no mesmo momento dos critérios da sombra (>= 12 meses de operação OU >= 8 trips-alvo no período prospectivo, o que vier depois). Antes disso só se descreve.
CRITÉRIO (no molde do P1b da sombra), por ledger e por hipótese, sem combinar: CONFIRMADA se nenhum trip do período prospectivo detectado pelo detector deixa de ser
detectado pelos alarmes de alta prioridade, E o FP de alta prioridade é menor que o FP total. Um trip perdido a reprova.
O QUE A CONFIRMAÇÃO AUTORIZA: propor a prioridade como regra de notificação na v2.1, com aprovação da operação. Até lá ela é só CONTEXTO na tela (alta/baixa).

Uso:
  PYTHONPATH=src python scripts/prioridade_sombra.py --ledger LEDGER [LEDGER ...] --csv ARQ [ARQ ...] [--extra PARQUET] [--registro CSV] --saida PASTA [--verifica]
`--verifica` compara o T5 do caminho de produção com a grade da pesquisa e o histórico com o EXP 62 (atual 7/8 FP 4; G168 e B800 7/8 FP 6). A referência do EXP 62 é o
replay SEMANAL (`data_local/controle_v2`, congelado em 05/03/2025): verificar com `--ledger data_local/controle_v2/<v> --desde 2026-04-22`. Os ledgers reais registraram o
histórico numa execução só (linha de base) e diferem do replay semanal em 3 episódios (na base atual pegam o 04/11: 8/8 contra 7/8); só o período prospectivo decide.
"""
from __future__ import annotations
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "src"))
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.controle_sinais import _ci, PKG
from nv.prioridade import aplica, carga_nascimento, prioriza

INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); ALVO0 = pd.Timestamp("2025-02-27", tz="UTC")
REF62 = {"atual": (7, 4), "G168": (7, 6), "B800": (7, 6)}


def t5_de_producao(ci, csvs, extra):
    with contextlib.redirect_stdout(io.StringIO()):
        raw = pd.concat([ci.carregar_dados(Path(c)) for c in csvs]).sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        if extra is not None:
            x = pd.read_parquet(extra); raw = pd.concat([x.loc[x.index < raw.index[0], raw.columns], raw]).sort_index()
        g = ci.preparar_grade(raw)
    _, estavel, _, _ = ci._mascara(g)
    return g["T5_AVG_A"].where(estavel)


def detectados(al, trips):
    eps = episodios_de_serie(al)
    return {t for t in trips if any(a <= t and b >= t - JANELA for a, b in eps)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ledger", nargs="+", required=True); ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--extra", default=None); ap.add_argument("--registro", default=str(PKG / "registro_trips.csv")); ap.add_argument("--saida", required=True)
    ap.add_argument("--verifica", action="store_true"); ap.add_argument("--desde", default=None, help="sobrepõe o `desde` do congelado.json (só para verificação)")
    a = ap.parse_args(); ci = _ci(); saida = Path(a.saida); saida.mkdir(parents=True, exist_ok=True)
    t5 = t5_de_producao(ci, a.csv, a.extra); trips = sorted(ci.carregar_trips(Path(a.registro)))
    print(f"T5 estável de produção: {t5.index[0]:%Y-%m-%d} a {t5.index[-1]:%Y-%m-%d %H:%M} | {len(trips)} trips no registro", flush=True)
    if a.verifica:
        from nv import io as nvio
        g = nvio.carrega_grade(); t5p = g["T5_AVG_A"].where(nvio.mascaras(g)["estavel"])
        j = t5.index.intersection(t5p.index); d = (t5.loc[j] - t5p.loc[j]).abs()
        print(f"VERIFICA T5: produção x grade da pesquisa em {len(j)} instantes: diferença máx {np.nanmax(d.to_numpy()):.2e}, NaN em lados diferentes {int((t5.loc[j].isna() != t5p.loc[j].isna()).sum())}", flush=True)
    for led in map(Path, a.ledger):
        nome = led.name; desde = pd.Timestamp(a.desde, tz="UTC") if a.desde else pd.Timestamp(json.load(open(led / "congelado.json"))["desde"])
        al, vig = carrega_controle(nome, pasta=led.parent)
        eps = ci._episodios(al); tab = prioriza(eps, t5)
        ids = pd.read_csv(led / "episodios.csv", parse_dates=["inicio"]) if (led / "episodios.csv").exists() else None
        if ids is not None:
            ids["inicio"] = pd.to_datetime(ids["inicio"], utc=True); tab = tab.merge(ids[["id", "inicio", "notificavel_em"]], on="inicio", how="left")
        tab.to_csv(saida / f"{nome}_prioridade.csv", index=False)
        series = {"detector": al, "H-CARGA": aplica(al, tab, "estatica"), "H-CARGA-DIN": aplica(al, tab, "dinamica")}
        print(f"\n[{nome}] congelado desde {desde:%Y-%m-%d}; {len(tab)} episódios no ledger ({(tab['estatica'] == 'baixa').sum()} de carga baixa, {(tab['dinamica'] == 'promovida').sum()} promovidos, "
              f"{(tab['estatica'] == 'pendente').sum()} pendentes)", flush=True)
        for rot, ini, fim, tr in (("histórico (descritivo)", INICIO8, desde - pd.Timedelta(minutes=2), [t for t in trips if ALVO0 <= t < desde]),
                                  ("PROSPECTIVO", desde, al.index[-1], [t for t in trips if t >= desde])):
            if fim <= ini:
                print(f"   {rot}: sem dado", flush=True); continue
            meses = float(vig.loc[ini:fim].sum()) * 2 / 60 / 730.5
            print(f"   {rot}: {meses:.1f} meses de operação vigiada, {len(tr)} trips", flush=True)
            det0 = detectados(series["detector"].loc[ini:fim], tr)
            for k, s in series.items():
                r = avalia(s, vig, tr, inicio_periodo=ini, fim_periodo=fim); perd = det0 - detectados(s.loc[ini:fim], tr)
                print(f"      {k:12s} {r.det}/{len(tr)} | FP {r.fp:2d} ({r.fp_mes:.3f}/mês) | {r.horas_alarme:5.0f} h" + (f" | perde {[f'{t:%d/%m/%Y}' for t in sorted(perd)]}" if perd else ""), flush=True)
                if a.verifica and rot.startswith("histórico") and k == "H-CARGA" and nome in REF62:
                    print(f"      VERIFICA {nome}: H-CARGA no histórico {r.det}/8 FP {r.fp} (EXP 62: {REF62[nome][0]}/8 FP {REF62[nome][1]}) -> {'PASSA' if (r.det, r.fp) == REF62[nome] else 'DIFERE'}", flush=True)
            if rot == "PROSPECTIVO":
                pronto = meses >= 12 or len(tr) >= 8
                if not pronto:
                    print(f"      decisão: AINDA NÃO (>= 12 meses ou >= 8 trips; há {meses:.1f} meses e {len(tr)} trips) — só descrição", flush=True)
                else:
                    fp0 = avalia(series["detector"], vig, tr, inicio_periodo=ini, fim_periodo=fim).fp
                    for k in ("H-CARGA", "H-CARGA-DIN"):
                        perd = det0 - detectados(series[k].loc[ini:fim], tr); fpk = avalia(series[k], vig, tr, inicio_periodo=ini, fim_periodo=fim).fp
                        print(f"      {k}: {'CONFIRMADA' if (not perd and fpk < fp0) else 'REPROVADA'} (trips perdidos {len(perd)}; FP {fpk} contra {fp0})", flush=True)
    print(f"\ntabelas em {saida}", flush=True)


if __name__ == "__main__":
    main()
