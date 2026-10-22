#!/usr/bin/env python3
"""EXP 64 — O REFRATÁRIO FANTASMA: episódio que NÃO vira alarme (dura menos que a duração mínima) arma o bloqueio de 72 h e silencia os seguintes. Supressor sistemático ou acaso?

PRÉ-REGISTRO (21/10/2026), COMMITADO ANTES DE RODAR. DESCRITIVO: nada se adota daqui. Roda no worker do ClearML (`scripts/worker_exp64.py`), branch exp/modos-de-falha.

POR QUÊ (EXP 63, f6a75ec). O `prever` aplica o refratário ANTES da duração mínima: um episódio de voto que nasce fora do bloqueio arma 72 h de refratário mesmo que depois
seja descartado por durar menos de 120 min — ninguém é notificado, e o que vier nas 72 h seguintes é descartado. No EXP 63 isso tirou o 11/04/2025 (um pedaço de 56 min em
10/04 armou o bloqueio). No controle o mesmo mecanismo bloqueia 7 episódios de 6 armamentos, TODOS FP. A pergunta é se o "9 FP" do controle depende de um supressor que
funciona por construção (o voto curto aparece quando o detector está agitado, antes de um episódio longo e falso) ou de sorte, e se ele custa detecção em outras bases.
ATENÇÃO, DECLARADO: na base atual eu JÁ VI o inventário (6 armamentos, 7 FP bloqueados, nenhum TP). Ali é descrição, não evidência; a evidência está nas outras 3 bases e no placebo.

DEFINIÇÕES (fixadas antes; a cauda do `prever` reimplementada a partir do voto e da força que ele próprio devolve):
  VOTO = severity != "normal" (já com a escalada por idade); EPISÓDIO = `ci._episodios(voto)`; NOTIFICÁVEL = dura >= duracao_min, ou forte e >= duracao_min_forte.
  FANTASMA = episódio que passa o refratário (arma o bloqueio) e NÃO é notificável.
  SEM FANTASMA (contrafactual) = a mesma cauda, mas o episódio não notificável não arma o bloqueio (o refratário passa a ser armado só por alarme notificado).
  BLOQUEADO PELO FANTASMA = episódio descartado porque nasce dentro do bloqueio armado por um fantasma.
BASES: atual (cache causal `sinais_v2_8ev`), G168 e B800 (replay causal de produção com as constantes da sombra, VB_GUARDA_H 168 e VB_HORAS_BASE 800, mesmo início e
complemento de 30 s do EXP 63) e atual com retreino n = 19.000 (t e p do harness fiel, `escores_sub34`).
MEDIDAS: M1 inventário — armamentos fantasma, quantos bloqueiam algo, episódios bloqueados (TP/FP, duração, força, se nasce na borda = até 30 min da abertura da máscara,
se a máscara fechou entre o armamento e o bloqueado). M2 contrafactual — detecção, início, banda, FP, FP/mês, horas, fora, com e sem fantasma; Δ = com − sem; lista dos
episódios que mudam (cascata). M3 placebo — P2 (PRINCIPAL): o mesmo número G de bloqueios de 72 h armados no fim de episódios de voto sorteados entre os que, no mundo sem
fantasma, NÃO armam bloqueio (controla "o voto curto aparece quando o detector está agitado"); P1: G bloqueios em instantes vigiados uniformes. 2.000 sorteios cada, aplicados
ao mundo sem fantasma. p = fração dos sorteios com ΔFP <= ΔFP real E Δdet >= Δdet real.
CLASSIFICAÇÃO (fixada antes, em ordem):
  RISCO = em alguma base o fantasma custa detecção (Δdet < 0).
  ACASO = p(P2) > 0,10 em >= 2 das 4 bases.
  CONSISTENTE = em >= 3 das 4 bases ΔFP <= −2 com Δdet = 0 e p(P2) <= 0,10.
  MISTO no resto.
EXPECTATIVAS: V1-V3 passam (~85%); base atual sem fantasma com FP >= 12 (~85%, inventário já visto) e a mesma detecção (~80%); FP/mês sem fantasma na base atual >= 1,1
(~60%); o fantasma custa detecção em alguma base (~35%); p(P2) <= 0,10 em >= 3 bases (~30%); >= 50% dos bloqueados (somando as bases) nascem na borda (~55%) e têm a máscara
fechando entre o armamento e eles (~60%). Classificação: RISCO ~35%, ACASO ~35%, CONSISTENTE ~20%, MISTO ~10%.
INTERPRETAÇÃO PREVISTA: CONSISTENTE -> o "9 FP" depende de um supressor não projetado mas que funciona por construção: documentar como parte do ponto de operação, NÃO corrigir
a ordem, e toda correção de dado futura (como a faixa do EXP 63) tem de ser avaliada com ele. ACASO -> o FP honesto do detector é o do mundo sem fantasma; registrar esse
número e levar a ordem refratário/duração para a discussão da v2.1 (corrigir sobe o FP, mas torna a decisão coerente). RISCO -> argumento para corrigir a ordem, ao custo de FP;
nenhuma mudança sem experimento próprio e sombra. MISTO -> documentar, sem ação.
LIMITES: 8 eventos; as 4 bases dividem o mesmo dado (não são independentes); o fantasma foi achado pós-hoc na base atual; o placebo P1 ignora que o detector agitado concentra
episódios (por isso o P2 é o principal).
VERIFICAÇÕES: V1 cada base reconstruída reproduz a sua referência (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12; n=19.000 5/8 FP 11) e, nas 3 com ledger, coincide com ele
em >= 99,5% dos instantes vigiados (medido antes de fixar, só a reconstrução: atual 100%, B800 100%, G168 99,65% — um FP de jan/2026 termina 28 h antes no replay do que no
ledger, 11/01 19:52 contra 13/01 00:18, sem mudar contagem; a medida usa o replay, que tem voto e força); V2 a cauda reimplementada devolve exatamente o `is_anomaly` do `prever` nas 4 bases; V3 com duração mínima zero, "com" e "sem" fantasma
são idênticos (o contrafactual só toca o fantasma).

RESULTADO (21/10/2026; worker do ClearML, tarefa 2acadf647f8745faa8f822d8c82828ed; pré-registro 493609a). V1-V3 PASSAM nas 4 bases. RISCO (pela ordem pré-registrada), mas
o fantasma é também CONSISTENTE em 3 das 4 bases: é um supressor de FP sistemático, que custou uma detecção numa base.
    base        com fantasma (o detector)            sem fantasma (contrafactual)         Δ          placebo P2 / P1
    atual       7/8 · FP  9 · 0,807/mês · 1.457 h    7/8 · FP 15 · 1,345/mês · 1.825 h    FP −6      p 0,000 / 0,000
    G168        8/8 · FP 14 · 1,256/mês · 1.967 h    8/8 · FP 20 · 1,794/mês · 2.153 h    FP −6      p 0,000 / 0,000
    B800        8/8 · FP 12 · 1,076/mês · 1.712 h    8/8 · FP 17 · 1,525/mês · 2.009 h    FP −5      p 0,000 / 0,000
    n=19000     5/8 · FP 11 · 0,986/mês · 1.505 h    6/8 · FP 15 · 1,345/mês · 1.692 h    FP −4, det −1 (o 11/04/2025)   p 0,000 / 0,002
  Os bloqueios sorteados no fim de outros votos (P2) tiram em média 1,1-1,4 FP e custam 0,3-0,6 detecção; o fantasma tira 4-6 FP e custa 0 em 3 bases. Os armamentos são os
  mesmos eventos nas 4 bases (13/08, 21/09, 08/10, 01/12/2025, 01/01 e 09/03/2026: o dado é o mesmo). Somando as bases, 30 bloqueados, 1 TP (o 11/04, que também caiu no
  EXP 63 pelo mesmo caminho). Cascata: em todas as bases o 14/10/2025 (135 h, FP) só existe COM o fantasma (sem ele, o 08/10 notificado bloqueia o 14/10).
  MECANISMO (contra a minha expectativa): não é borda nem parada — só 5 de 30 bloqueados nascem na borda e em 6 de 30 a máscara fecha entre o armamento e eles. O fantasma é um
  voto curto (10-82 min) com a máquina em operação contínua, 3 a 40 h antes de um episódio longo (até 164 h): o refratário trata o episódio longo como repetição do curto, que é
  o propósito do refratário ("repetir não é notícia"). NADA fica escondido: o `prever` mostra o voto curto e o episódio bloqueado como "atenção" na tela; o fantasma rebaixa o
  episódio longo de alarme para atenção.
PLACAR: ACERTEI: V1-V3 (~85%); base atual sem fantasma com FP >= 12 (~85%; 15) e mesma detecção (~80%); FP/mês sem fantasma >= 1,1 (~60%; 1,345); o fantasma custa detecção
  em alguma base (~35%; n=19.000); p(P2) <= 0,10 em >= 3 bases (~30%; 4 de 4). ERREI: >= 50% dos bloqueados na borda (~55%; 5/30) e com a máscara fechando entre (~60%; 6/30).
VEREDITO (como pré-registrado): RISCO -> há argumento para corrigir a ordem refratário/duração, mas ele é FRACO e contra a meta: corrigir custa +4 a +6 FP (+0,45 a +0,54
  FP/mês) e +187 a +368 h de alarme para ganhar 1 detecção em 1 das 4 bases (0 nas outras 3). O fantasma fica, documentado como parte do ponto de operação: o FP de 0,807/mês
  do controle depende dele, e o FP "sem fantasma" é 1,345/mês. Toda correção futura de dado ou de voto (como a faixa do EXP 63) tem de ser avaliada sabendo que partir ou fundir
  votos arma ou desarma o fantasma.

Uso local (encanamento, janela curta, sem tabelas):  EXP64_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/64_refratario_fantasma.py
No worker:  PYTHONPATH=src python scripts/worker_exp64.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import JANELA
from nv.causal import carrega_registro
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, substitui

TESTE = bool(os.environ.get("EXP64_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); BORDA = pd.Timedelta(minutes=30)
INICIO_REPLAY = pd.Timestamp("2025-01-08 07:34", tz="UTC") + pd.Timedelta(weeks=42 if TESTE else 0)   # no teste, um corte do calendário semanal do replay completo
VARIANTES = {"G168": {"VB_GUARDA_H": 168.0}, "B800": {"VB_HORAS_BASE": 800.0}}
REF = {"atual": (7, 9), "G168": (8, 14), "B800": (8, 12), "n=19000": (5, 11)}
NSORT = 40 if TESTE else 2000
GL = {}


def decide(proc, det):
    p = proc[["t", "p", "sp", "vb", "mask", "reset"]].copy(); p["mask"] = p["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        return decide_v2(p, det)


def cauda(res, det, sem_fantasma=False, blocos=()):
    """A cauda do `prever` (refratário com furo forte-e-velho, depois duração mínima), episódio a episódio. Devolve (alarme, registros)."""
    ci = GL["ci"]; voto = res["severity"].ne("normal"); forca = res["forca"]; idx = res.index
    al = pd.Series(False, index=idx); bloq = ini_bloq = None; arma = None; regs = []
    ref_h = pd.Timedelta(hours=det["refratario_h"])
    for a, b in ci._episodios(voto):
        fmax = float(forca.loc[a:b].max()); forte = fmax > det["escalada_abs"]
        velho = ini_bloq is not None and (a - ini_bloq).total_seconds() / 3600 >= det["escalada_idade_h"]
        if bloq is not None and a <= bloq and not (forte and velho):
            regs.append(dict(a=a, b=b, estado="bloqueado", por=arma, forca=fmax)); continue
        if any(u0 <= a <= u1 for u0, u1 in blocos):
            regs.append(dict(a=a, b=b, estado="placebo", por=None, forca=fmax)); continue
        dur = (b - a).total_seconds() / 60 + 2
        notif = dur >= det["duracao_min"] or (forte and dur >= det["duracao_min_forte"])
        if sem_fantasma and not notif:
            regs.append(dict(a=a, b=b, estado="curto", por=None, forca=fmax)); continue
        bloq = b + ref_h; ini_bloq = a; arma = (a, b, notif)
        regs.append(dict(a=a, b=b, estado="notificado" if notif else "fantasma", por=None, forca=fmax))
        if notif:
            al.loc[a:b] = True
    return al, regs


def mede(al, nome):
    vig = GL["vig"][nome]; a = al.reindex(vig.index).fillna(False)
    r = avalia(a, vig, GL["trips"], inicio_periodo=INICIO8)
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora)


def rotulo(a, b):
    return "TP" if any(a <= t and b >= t - JANELA for t in GL["todos_trips"]) else "FP"


def sorteio(arg):
    nome, tipo, semente = arg
    res, det, G, cand, vig_t = GL["S"][nome]; rng = np.random.default_rng(semente)
    if tipo == "P2":
        fins = rng.choice(cand, size=min(G, len(cand)), replace=False); blocos = [(f + pd.Timedelta(minutes=1), f + pd.Timedelta(hours=det["refratario_h"])) for f in fins]
    else:
        us = rng.choice(vig_t, size=G, replace=False); blocos = [(u, u + pd.Timedelta(hours=det["refratario_h"])) for u in us]
    al, _ = cauda(res, det, sem_fantasma=True, blocos=blocos); m = mede(al, nome)
    return nome, tipo, m["fp"], m["det"]


def main():
    t0 = time.time(); ci = _ci(); GL["ci"] = ci; ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    GL["todos_trips"] = sorted(carrega_registro()); GL["trips"] = [t for t in GL["todos_trips"] if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    led = {v: carrega_controle(v) for v in ("atual", "G168", "B800")}
    GL["vig"] = {"atual": led["atual"][1], "G168": led["G168"][1], "B800": led["B800"][1], "n=19000": led["atual"][1]}
    proc0 = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")
    R = {"atual": decide(proc0, det)}
    x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet")
    for v, const in VARIANTES.items():
        padrao = {k: getattr(ci, k) for k in const}
        for k, val in const.items():
            setattr(ci, k, val)
        try:
            S = sinais_causais(INICIO_REPLAY, cache=ROOT / f"sinais_64_{v}{'_teste' if TESTE else ''}.parquet", extra=x)
        finally:
            for k, val in padrao.items():
                setattr(ci, k, val)
        R[v] = decide(S, det); print(f"replay {v}: {len(S)} instantes ({time.time() - t0:.0f} s)", flush=True)
    px = proc0[["t", "p", "sp", "vb", "mask", "reset"]].copy()
    for c in ("t", "p"):
        s = pd.read_parquet(ROOT / "escores_sub34" / f"{c}_A_19000.parquet")["s"]
        px = substitui(px, c, s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna()))
    R["n=19000"] = decide(px, det)

    # ── verificações ──
    ini_v = max(INICIO8, INICIO_REPLAY + pd.Timedelta(days=7))
    for nome, res in R.items():
        al = res["is_anomaly"].astype(bool); m = mede(al, nome); bom = (m["det"], m["fp"]) == REF[nome]
        txt = f"V1 [{nome}] {m['det']}/8 FP {m['fp']} (referência {REF[nome][0]}/8 FP {REF[nome][1]})"
        if nome in led:
            lal, lvg = led[nome]; j = lvg.index[lvg.to_numpy(bool) & (lvg.index >= ini_v)].intersection(al.index)
            co = float((al.loc[j].to_numpy() == lal.loc[j].to_numpy()).mean()); bom &= co >= 0.995; txt += f"; coincide com o ledger em {100 * co:.2f}% dos vigiados"
        ok[f"V1_{nome}"] = bool(bom); print(f"{txt} -> {'PASSA' if bom else 'FALHA'}", flush=True)
        a2, _ = cauda(res, det); ok[f"V2_{nome}"] = bool((a2.to_numpy() == al.to_numpy()).all())
        d0 = dict(det); d0["duracao_min"] = 0; d0["duracao_min_forte"] = 0
        ok[f"V3_{nome}"] = bool((cauda(res, d0)[0] == cauda(res, d0, sem_fantasma=True)[0]).all())
        print(f"V2 [{nome}] cauda reimplementada = is_anomaly -> {'PASSA' if ok[f'V2_{nome}'] else 'FALHA'} | V3 com duração zero, com = sem -> {'PASSA' if ok[f'V3_{nome}'] else 'FALHA'}", flush=True)
    if TESTE:
        ok = {k: v for k, v in ok.items() if not k.startswith("V1_") or k == "V1_atual"}   # janela curta: as bases replayadas não fecham a referência inteira

    # ── M1/M2: com e sem fantasma ──
    OUT, BLQ, MUD = {}, [], {}
    for nome, res in R.items():
        mask = res["vigiado"].astype(bool).to_numpy(); ab = res.index[np.flatnonzero(mask[1:] & ~mask[:-1]) + 1]
        a_com, regs = cauda(res, det); a_sem, regs_sem = cauda(res, det, sem_fantasma=True)
        mc, ms = mede(a_com, nome), mede(a_sem, nome)
        fant = [r for r in regs if r["estado"] == "fantasma" and r["b"] >= INICIO8 - pd.Timedelta(hours=det["refratario_h"])]
        blq = [r for r in regs if r["estado"] == "bloqueado" and r["por"] is not None and not r["por"][2] and r["a"] >= INICIO8]
        for r in blq:
            k = ab.searchsorted(r["a"], side="right") - 1
            fecha = bool((~res["vigiado"].astype(bool).loc[r["por"][1]:r["a"]]).any())
            BLQ.append(dict(base=nome, armou=r["por"][0], armou_min=(r["por"][1] - r["por"][0]).total_seconds() / 60 + 2, nasce=r["a"], dur_h=(r["b"] - r["a"]).total_seconds() / 3600,
                            forca=r["forca"], rotulo=rotulo(r["a"], r["b"]), borda=bool(k >= 0 and r["a"] - ab[k] <= BORDA), mascara_fecha=fecha))
        from nv.avalia.metricas import episodios_de_serie
        ec = set(episodios_de_serie(a_com.loc[INICIO8:])); es = set(episodios_de_serie(a_sem.loc[INICIO8:]))
        MUD[nome] = ([(a, b, rotulo(a, b)) for a, b in sorted(es - ec)], [(a, b, rotulo(a, b)) for a, b in sorted(ec - es)])
        arma_sem = {(r["a"], r["b"]) for r in regs_sem if r["estado"] == "notificado"}
        cand = [r["b"] for r in regs_sem if (r["a"], r["b"]) not in arma_sem and r["b"] >= INICIO8 - pd.Timedelta(hours=det["refratario_h"])
                and (r["a"], r["b"]) not in {(f["a"], f["b"]) for f in fant}]
        vig_t = res.index[mask & (res.index >= INICIO8 - pd.Timedelta(hours=det["refratario_h"]))]
        OUT[nome] = dict(com=mc, sem=ms, G=len(fant), G_bloq=len({(r["por"][0]) for r in blq}))
        GL.setdefault("S", {})[nome] = (res, det, len(fant), np.array(cand, dtype=object), np.array(vig_t, dtype=object))

    # ── M3: placebo ──
    tarefas = [(nome, tipo, 1000 * i + j) for nome in R for i, tipo in enumerate(("P1", "P2")) for j in range(NSORT)]
    with Pool(int(os.environ.get("EXP64_PROCS", 4))) as pool:
        PL = list(pool.imap_unordered(sorteio, tarefas, chunksize=20))
    for nome in R:
        dfp = OUT[nome]["com"]["fp"] - OUT[nome]["sem"]["fp"]; ddet = OUT[nome]["com"]["det"] - OUT[nome]["sem"]["det"]; OUT[nome].update(dfp=dfp, ddet=ddet)
        for tipo in ("P1", "P2"):
            xs = [(fp - OUT[nome]["sem"]["fp"], d - OUT[nome]["sem"]["det"]) for n_, t_, fp, d in PL if n_ == nome and t_ == tipo]
            OUT[nome][f"p_{tipo}"] = float(np.mean([(f <= dfp) and (d >= ddet) for f, d in xs]))
            OUT[nome][f"dfp_{tipo}"] = float(np.mean([f for f, _ in xs])); OUT[nome][f"ddet_{tipo}"] = float(np.mean([d for _, d in xs]))
    GL.update(OUT=OUT, BLQ=BLQ, MUD=MUD)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (janela curta; saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(ok, t0, escreve=True)


def relata(ok, t0, escreve):
    OUT, BLQ, MUD = GL["OUT"], GL["BLQ"], GL["MUD"]
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    linha = lambda nome, m: print(f"   {nome:14s} {m['det']}/8 | início {m['inicio']} | banda {m['banda']} | FP {m['fp']:2d} | {m['fp_mes']:.3f}/mês | {m['horas']:5.0f} h | fora {m['fora']:4.0f} h", flush=True)
    print("\nM2 — com e sem fantasma, por base", flush=True)
    for nome, o in OUT.items():
        print(f" [{nome}] armamentos fantasma no período: {o['G']} (bloqueiam algo: {o['G_bloq']})", flush=True)
        linha("com fantasma", o["com"]); linha("sem fantasma", o["sem"])
        print(f"   Δ (com − sem): det {o['ddet']:+d}, FP {o['dfp']:+d} | placebo P2 p = {o['p_P2']:.3f} (ΔFP médio {o['dfp_P2']:+.2f}, Δdet médio {o['ddet_P2']:+.2f}) | "
              f"P1 p = {o['p_P1']:.3f} (ΔFP médio {o['dfp_P1']:+.2f}, Δdet médio {o['ddet_P1']:+.2f})", flush=True)
        sai, entra = MUD[nome]
        print("   só SEM fantasma (o fantasma suprime): " + ("; ".join(f"{loc(a)} ({(b - a).total_seconds() / 3600:.0f} h, {r})" for a, b, r in sai) or "nenhum"), flush=True)
        print("   só COM fantasma (cascata): " + ("; ".join(f"{loc(a)} ({(b - a).total_seconds() / 3600:.0f} h, {r})" for a, b, r in entra) or "nenhum"), flush=True)
    print("\nM1 — episódios bloqueados por fantasma", flush=True)
    for r in BLQ:
        print(f"   [{r['base']:7s}] armou {loc(r['armou'])} ({r['armou_min']:.0f} min) -> bloqueou {loc(r['nasce'])} ({r['dur_h']:.0f} h, força {r['forca']:.1f}, {r['rotulo']}, "
              f"borda {'sim' if r['borda'] else 'não'}, máscara fecha entre {'sim' if r['mascara_fecha'] else 'não'})", flush=True)
    nb = len(BLQ)
    if nb:
        print(f"   somando as bases: {nb} bloqueados, {sum(r['rotulo'] == 'TP' for r in BLQ)} TP; nascem na borda {sum(r['borda'] for r in BLQ)}/{nb}; máscara fecha entre "
              f"{sum(r['mascara_fecha'] for r in BLQ)}/{nb}", flush=True)
    risco = any(o["ddet"] < 0 for o in OUT.values()); acaso = sum(o["p_P2"] > 0.10 for o in OUT.values()) >= 2
    consist = sum(o["dfp"] <= -2 and o["ddet"] == 0 and o["p_P2"] <= 0.10 for o in OUT.values()) >= 3
    cl = "RISCO" if risco else ("ACASO" if acaso else ("CONSISTENTE" if consist else "MISTO"))
    print(f"\nCLASSIFICAÇÃO: {cl} (custa detecção em alguma base: {risco}; p(P2) > 0,10 em >= 2 bases: {acaso}; consistente em >= 3: {consist})", flush=True)
    if escreve:
        pd.DataFrame([dict(base=n, mundo=w, **o[w]) for n, o in OUT.items() for w in ("com", "sem")]
                     + [dict(base=n, mundo="placebo", **{k: v for k, v in o.items() if k.startswith(("p_", "dfp", "ddet", "G"))}) for n, o in OUT.items()]).to_csv(ROOT / "exp64_resultados.csv", index=False)
        pd.DataFrame(BLQ).to_csv(ROOT / "exp64_bloqueados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
