#!/usr/bin/env python3
"""SOTA H7 — BLACKOUT PÓS-PARTIDA MAIS LONGO E LIMPEZA DO CANAL p (sem a PI_5134001, tag de outra unidade). Tag ClearML `sota-h7`.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. Pedido do usuário depois da triagem dos 9 FP (5 nascem 6,5 h após a partida, na abertura da vigilância; 1 é puxado
pela PI_5134001, do ar de instrumentação — outra unidade).
O QUE JÁ SE SABE (declarado): EXP 52 — tirar a PI_5134001 do p deu 7/8, FP 9-10, mas +135 h de alarme (piora a carga); EXP 54 — blackout de 12 h na base atual só tirou 1 FP
(9 -> 8); EXP 60 e "relógio de partida" — os trips logo depois da partida (04/11, partida a frio, trip 15 h depois; 07/04, trip 13 h depois de religar) foram perdidos
nas bases G168 e B800 quando a borda foi adiada; e 5 dos 8 trips acontecem até 31 h após a partida. NÃO olhei os horários de partida antes de cada trip para este pré-registro.
VARIANTES: blackout pós-partida X em {6 (atual), 7, 8, 10, 12} h (a máscara do pacote: RUNNING_A > 0,5 E T5 > 300 E fora de X h depois de cada partida; reset do CUSUM
em ~máscara ou partida), cada uma COM e SEM a PI_5134001 no canal p (o p sem ela = PCA fiel, 11 tags, retreino mensal causal com as últimas 20.000 amostras estáveis, como o
harness que reproduz a produção). O EWMA dos sinais não muda (roda sobre o sinal inteiro); só a decisão vê a máscara nova. BASES: atual (a que decide), G168, B800 e
atual com retreino n = 19.000 (robustez).
REGRA DE DECISÃO (do usuário, fixada antes), na base atual, com detecção "de pé" e FP/mês do protocolo:
  PASSO 1 — PI: "sem PI_5134001" (com X = 6) é aceito se não perde nenhum trip que o atual detecta, REDUZ o FP/mês e não perde trip em LOEO.
  PASSO 2 — JANELA (com a escolha do passo 1): a candidata X é aceitável se não perde nenhum trip que o X = 6 detecta e REDUZ o FP/mês; escolhe-se a MENOR aceitável
  (empate -> menor janela); sem nenhuma, fica 6 h. LOEO: em cada um dos 8 recortes (um trip escondido), a regra escolhe a janela só com os outros 7 (o trip escondido conta
  como inexistente: o episódio dele vira FP) e verifica se a janela escolhida perde o trip escondido. A escolha só é ACEITA se nenhum trip escondido é perdido em LOEO.
ROBUSTEZ (reportada; não muda a regra, muda a recomendação): a configuração escolhida em G168, B800 e n = 19.000 contra o X = 6 de cada uma (trips perdidos, ΔFP, Δhoras);
PLACEBO do blackout: as mesmas horas extras mascaradas, em blocos de X − 6 h no mesmo número de partidas, começando em instantes vigiados sorteados (200 sorteios):
p = fração com FP <= o real e detecção >= a real; PLACEBO da PI: tirar do p, uma de cada vez, cada uma das outras 11 tags de pressão: p = fração com ΔFP <= o real e
detecção >= a real. IC por bootstrap de eventos e de meses e comparação pareada contra o X = 6 com PI (protocolo único); horas de alarme sempre.
CHECAGEM POR TRIP (pedido do usuário): para cada trip, a última partida antes dele, o início do episódio que o detecta no X = 6 e se esse início cai dentro da janela
nova [partida, partida + X); atenção a 07/04 e 11/04/2025. TABELA ANTES/DEPOIS DOS 9 FP: o destino de cada FP do X = 6 em cada configuração (sai, encurta, fica) e os FP novos.
RECOMENDAÇÃO PREVISTA: regra ACEITA + robustez sem trip perdido + placebo p <= 0,10 -> variante para a SOMBRA (nada vai direto para produção). Regra aceita com robustez ou
placebo falhando -> registrar como frágil, não adotar. Nada aceito -> o blackout fica em 6 h e a PI_5134001 fica no p (a higiene não paga).
EXPECTATIVAS: V1-V3 passam (~85%); sem PI não reduz FP/mês (~75%, EXP 52); alguma janela de 7-12 h reduz o FP/mês na base atual (~70%); alguma janela de 7-12 h perde
trip na base atual (~45%); a regra aceita alguma janela > 6 h (~35%); se aceitar, a robustez perde trip em G168 ou B800 (~60%); placebo do blackout p <= 0,10 (~40%).
LIMITES: 8 eventos (7 detectados na base atual); X em 5 valores; a regra é do usuário e foi fixada antes; nada se adota sem a sombra.
VERIFICAÇÕES: V1 com X = 6 a máscara recalculada aqui coincide com a do cache causal em 100% dos instantes do PERÍODO AVALIADO (01/02/2025 a 21/04/2026), e cada base
reproduz a sua referência (DECLARADO: a primeira versão desta verificação pedia >= 99,99% de TODOS os instantes e deu 99,954%; os 181 instantes diferentes estão todos em
out-nov/2024, antes do período: 180 são as primeiras 6 h do cache, porque o replay começa em 25/10/2024 e a primeira linha da janela conta como partida, e 1 é um instante
com RUNNING_A exatamente 0,5; corrigido antes de qualquer resultado) (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12; n=19.000 5/8 FP 11); V2 o p do harness com as 12 tags reproduz o `escores_sub34` (<= 1e-9); V3 o placebo com zero blocos devolve o real.

RESULTADO (22/10/2026; worker do ClearML, tarefa d32245dab5404dbebbb6b1c1237c49ad; pré-registro 8fb3e52). V1-V3 PASSAM. NADA ACEITO: o blackout fica em 6 h e a PI_5134001
fica no canal p.
  BASE ATUAL (X = 6 com PI: 7/8, FP 9, 0,807/mês, 1.457 h):
    com PI   X = 7: 6/8 FP 11 1.422 h | 8: 6/8 FP 10 1.332 h | 10: 6/8 FP 12 633 h | 12: 6/8 FP 8 (0,717) 1.229 h — TODAS perdem o 11/04; só a de 12 h reduz o FP (−1)
    sem PI   X = 6: 7/8 FP 10 1.593 h (+136 h, como no EXP 52) | 7-12: 6/8, FP 9 a 14, todas perdem o 11/04
  PASSO 1 (PI): não reduz o FP (+1) -> FICA. Placebo da PI (tirar cada outra tag do p): ΔFP de −1 a +1; p = 0,818 — tirar a PI_5134001 não difere de tirar qualquer tag.
  PASSO 2 (janela): nenhuma janela de 7-12 h passa (todas perdem o 11/04); LOEO: com o 11/04 escondido a regra escolheria 12 h e perderia exatamente o 11/04 -> fica 6 h.
  ROBUSTEZ: G168 perde 2 trips com 7-10 h e 3 com 12 h; B800 perde 1 com 7-8 h e 3 com 10-12 h; n=19.000 GANHA 1 (5 -> 6) com 7, 8 e 12 h — o efeito troca de sinal com a base.
  PLACEBO DO BLACKOUT: as mesmas horas mascaradas em blocos sorteados dão MAIS FP (mediana 16-24) que o blackout real em todas as bases (p = 0,000): mascarar quebra
    episódios em mais episódios; mascarar depois da partida é menos ruim que mascarar ao acaso, mas não melhora o 6 h.
  CHECAGEM POR TRIP (última partida antes do trip; início do episódio que o detecta):
    27/02 partida 20/02 23:14, episódio às 6,5 h (na borda) — sobrevive em todas as janelas (o episódio de 140 h continua depois da janela);
    17/03 episódio 14,6 h após a partida; 07/04 partida 07/04 05:16 (13 h antes do trip), mas o episódio que detecta nasceu em 06/04 10:10, ANTES da partida — sobrevive em todas;
    11/04 partida 10/04 16:08 (21,9 h antes), episódio às 17,8 h — FORA de qualquer janela, e mesmo assim PERDIDO com X >= 7. MECANISMO (verificado depois, com a cauda do
    EXP 64): com 6 h, o voto de 08/04 19:00 a 10/04 16:04 é um bloco de 45 h nascido dentro do refratário do alarme de 06/04 e descartado; com X >= 7 a máscara maior parte
    esse bloco, e um pedaço curto em 10/04 13:10 nasce fora do refratário, arma um REFRATÁRIO FANTASMA de 72 h (dura menos que os 120 min mínimos) e silencia o 11/04 —
    o mesmo caminho do EXP 63; 29/04 e 09/12 episódios nascidos antes da partida; 04/11 não detectado na base atual; 26/02/2026 episódio às 6,5 h (borda), sobrevive.
  OS 9 FP: as janelas mais longas ENCURTAM 5-6 dos 9 e eliminam 1-3 (15/06 com 8 h ou mais; 08/04 com 7, 8 e 12 h; 23/08 com 12 h), mas criam 2-4 FP NOVOS. Sem a PI some
    o 16/01 e nascem 2 novos. O 26/10 (112 h) não muda em nenhuma configuração.
PLACAR: ACERTEI: V1-V3 (~85%); sem PI não reduz o FP (~75%); alguma janela de 7-12 h perde trip na base atual (~45%; todas perdem o 11/04); se aceita, a robustez perderia
  trip em G168/B800 (perde em todas). ERREI: alguma janela reduz o FP na base atual (~70%; só a de 12 h, −1); placebo do blackout p <= 0,10 (~40%; p = 0,000, mas no sentido
  de o placebo ser PIOR, sem que o real melhore o 6 h). A regra não aceitou nada (dei ~65% para isso).
VEREDITO (como pré-registrado): nada aceito -> o blackout fica em 6 h e a PI_5134001 fica no p (a higiene não paga). O 11/04 é perdido pelo refratário fantasma, não pela
  máscara: qualquer mudança que parta os votos perto de uma parada esbarra nele.

Uso local (encanamento):  SOTA_TESTE=1 PYTHONPATH=src:experiments .venv/bin/python -m sota.h7_partida_e_canal
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp h7_partida_e_canal --tags sota-h7 --procs 24 --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward
from sota import protocolo as P

TESTE = bool(os.environ.get("SOTA_TESTE"))
ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "h7"
XS = (6, 7, 8, 10, 12); PI = "PI_5134001"; N_PLAC = 10 if TESTE else 200; POR_HORA = 30
REF = {"atual": (7, 9), "G168": (8, 14), "B800": (8, 12), "n=19000": (5, 11)}
VARIANTES = {"G168": {"VB_GUARDA_H": 168.0}, "B800": {"VB_HORAS_BASE": 800.0}}
INICIO_REPLAY = pd.Timestamp("2025-01-08 07:34", tz="UTC")
GL = {}


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, EXP / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


def mascara(op, t5, X, idx):
    estavel = op & (t5 > 300); partida = op & ~op.shift(fill_value=False)
    bl = partida.rolling(int(X * POR_HORA), min_periods=1).max().astype(bool)
    m = (estavel & ~bl).reindex(idx).fillna(False); pa = partida.reindex(idx).fillna(False)
    return m, (~m) | pa


def decide(proc, det):
    with contextlib.redirect_stdout(io.StringIO()):
        return decide_v2(proc, det)["is_anomaly"].astype(bool)


def walk(arg):
    nome, tags, n = arg
    E46 = GL["E46"]; s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), GL["g"], tags, GL["est"], P.INICIO, GL["fim"], n_treino=n, semente=0)
    return nome, s


def placebo_blackout(arg):
    base, pi, X, semente = arg
    proc = GL["PROC"][(base, pi)].copy(); idx = proc.index; m6 = GL["MASK"][6][0].reindex(idx).fillna(False); r6 = GL["MASK"][6][1].reindex(idx).fillna(True); rng = np.random.default_rng(semente)
    cand = np.flatnonzero(m6.to_numpy() & (idx >= P.INICIO) & (idx <= P.FIM)); n_bl = int((X - 6) * POR_HORA); m = m6.to_numpy().copy(); r = r6.to_numpy().copy()
    for i in rng.choice(cand, size=GL["n_partidas"], replace=False) if X > 6 else []:
        m[i:i + n_bl] = False; r[i:i + n_bl] = True
    proc["mask"] = m; proc["reset"] = r; al = decide(proc, GL["det"]); x = P.mede(al, GL["vig_c"], GL["trips"])
    return base, pi, X, x["fp"], x["det"]


def detectados(al, trips):
    eps = episodios_de_serie(al.loc[P.INICIO:P.FIM])
    return {t for t in trips if any(a <= t and b >= t - JANELA for a, b in eps)}


def fp_mes(al, vig, trips):
    return P.mede(al, vig, trips)["fp_mes"]


def regra_janela(AL, vig, trips, pi):
    """Menor X aceitável (não perde trip que o X = 6 detecta e reduz o FP/mês) com os trips dados; 6 se nenhuma."""
    d6 = detectados(AL[(pi, 6)], trips); f6 = fp_mes(AL[(pi, 6)], vig, trips)
    for X in XS[1:]:
        if d6 <= detectados(AL[(pi, X)], trips) and fp_mes(AL[(pi, X)], vig, trips) < f6:
            return X
    return 6


def main():
    t0 = time.time(); ci = _ci(); SAIDA.mkdir(parents=True, exist_ok=True); ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; trips = P.trips_alvo(carrega_registro()); GL.update(det=det, trips=trips, E46=_carrega("exp46", "46_optuna_modelos_loeo_placebo.py"))
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); al_c, vig_c = carrega_controle("atual")
    GL.update(g=g, est=est, fim=vig_c.index[-1], vig_c=vig_c)
    tags_p = list(modelos[-1]["normalizacao"]["pressao"]["cols"]); outras = [t for t in tags_p if t != PI]
    if TESTE:
        outras = outras[:2]
    assert PI in tags_p, tags_p
    jobs = [("p12_20000", tags_p, 20000), ("semPI_20000", [t for t in tags_p if t != PI], 20000), ("semPI_19000", [t for t in tags_p if t != PI], 19000)]
    jobs += [(f"plac_{t}", [x for x in tags_p if x != t], 20000) for t in outras]
    with Pool(min(int(os.environ.get("SOTA_PROCS", 8)), len(jobs))) as pool:
        W = dict(pool.imap_unordered(walk, jobs, chunksize=1))
    ref = pd.read_parquet(ROOT / "escores_sub34" / "p_A_20000.parquet")["s"]; j = ref.index.intersection(W["p12_20000"].index); j = j[(j >= P.INICIO) & (j <= P.FIM)]
    a, b = ref.loc[j].to_numpy(), W["p12_20000"].loc[j].to_numpy(); mm = np.isfinite(a) & np.isfinite(b)
    ok["V2"] = bool(np.max(np.abs(a[mm] - b[mm])) <= 1e-9 and (np.isfinite(a) == np.isfinite(b)).all())
    print(f"V2 — p do harness (12 tags) x escores_sub34: dif. máx {np.max(np.abs(a[mm] - b[mm])):.1e} -> {'PASSA' if ok['V2'] else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    ew = lambda s, c: s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna())
    proc0 = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc0["mask"] = proc0["mask"].astype(bool)
    BASES = {"atual": proc0}
    x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet")
    for v, const in VARIANTES.items():
        pad = {k: getattr(ci, k) for k in const}
        for k, val in const.items():
            setattr(ci, k, val)
        try:
            S = sinais_causais(INICIO_REPLAY, cache=ROOT / f"sinais_64_{v}.parquet", extra=x)
        finally:
            for k, val in pad.items():
                setattr(ci, k, val)
        pv = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); pv["mask"] = pv["mask"].astype(bool); BASES[v] = pv
    p19 = proc0.copy()
    for c in ("t", "p"):
        s = pd.read_parquet(ROOT / "escores_sub34" / f"{c}_A_19000.parquet")["s"]; p19 = substitui(p19, c, ew(s, c))
    BASES["n=19000"] = p19
    idx = proc0.index; op = (g["RUNNING_A"] > 0.5).fillna(False); t5 = g["T5_AVG_A"]
    MASK = {X: mascara(op, t5, X, idx) for X in XS}; GL["MASK"] = MASK
    per = (idx >= P.INICIO) & (idx <= P.FIM)
    co = float((MASK[6][0] == proc0["mask"])[per].mean()); cr = float((MASK[6][1] == proc0["reset"].astype(bool))[per].mean())
    PROC, AL = {}, {}
    for base, pr in BASES.items():
        for pi in ("com", "sem"):
            q = pr.copy()
            if pi == "sem":
                q = substitui(q, "p", ew(W["semPI_19000" if base == "n=19000" else "semPI_20000"], "p"))
            PROC[(base, pi)] = q
            for X in XS:
                z = q.copy(); m, r = MASK[X]; z["mask"] = m.reindex(z.index).fillna(False).to_numpy(); z["reset"] = r.reindex(z.index).fillna(True).to_numpy()
                AL[(base, pi, X)] = decide(z, det)
    GL["PROC"] = PROC
    v1 = {b: (P.mede(AL[(b, "com", 6)], vig_c, trips)["det"], P.mede(AL[(b, "com", 6)], vig_c, trips)["fp"]) for b in BASES}
    ok["V1"] = co == 1.0 and cr == 1.0 and all(v1[b] == REF[b] for b in BASES)
    print(f"V1 — máscara X = 6 recalculada x cache no período avaliado: {100 * co:.4f}% (reset {100 * cr:.4f}%); referências {v1} -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)
    GL["n_partidas"] = int(((op & ~op.shift(fill_value=False)).loc[P.INICIO:P.FIM]).sum())
    v3 = placebo_blackout(("atual", "com", 6, 0)); ok["V3"] = (v3[4], v3[3]) == REF["atual"]
    print(f"V3 — placebo sem blocos devolve o real: {v3[4]}/8 FP {v3[3]} -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    # ── métricas, IC e comparação pareada contra X = 6 com PI ──
    M = {}
    for (b, pi, X), al in AL.items():
        m = P.mede(al, vig_c, trips); n = 200 if TESTE else P.N_BOOT
        M[(b, pi, X)] = dict(m, ic=P.ic(al, vig_c, trips, n=n), cmp=P.compara(al, AL[(b, "com", 6)], vig_c, trips, n=n), det_set=detectados(al, trips))
    # ── regra (base atual) ──
    A = {(pi, X): AL[("atual", pi, X)] for pi in ("com", "sem") for X in XS}
    d6 = detectados(A[("com", 6)], trips); f6 = fp_mes(A[("com", 6)], vig_c, trips)
    passo1 = d6 <= detectados(A[("sem", 6)], trips) and fp_mes(A[("sem", 6)], vig_c, trips) < f6
    loeo1 = []
    for k in trips:
        tr = [t for t in trips if t != k]; dk = detectados(A[("com", 6)], tr)
        aceita = dk <= detectados(A[("sem", 6)], tr) and fp_mes(A[("sem", 6)], vig_c, tr) < fp_mes(A[("com", 6)], vig_c, tr)
        loeo1.append(aceita and (k in detectados(A[("com", 6)], [k])) and (k not in detectados(A[("sem", 6)], [k])))
    pi_ok = passo1 and not any(loeo1); pi_esc = "sem" if pi_ok else "com"
    Xesc = regra_janela(A, vig_c, trips, pi_esc); loeo2 = []
    for k in trips:
        tr = [t for t in trips if t != k]; Xk = regra_janela(A, vig_c, tr, pi_esc)
        loeo2.append((k, Xk, (k in detectados(A[(pi_esc, 6)], [k])) and (k not in detectados(A[(pi_esc, Xk)], [k]))))
    aceita = Xesc > 6 and not any(p for _, _, p in loeo2)
    GL.update(M=M, AL=AL, BASES=list(BASES), passo1=passo1, loeo1=loeo1, pi_esc=pi_esc, Xesc=Xesc, loeo2=loeo2, aceita=aceita, op=op)
    # ── placebos ──
    tarefas = [(b, "com", X, 1000 * X + i) for b in BASES for X in XS[1:] for i in range(N_PLAC)]
    with Pool(int(os.environ.get("SOTA_PROCS", 8))) as pool:
        PB = list(pool.imap_unordered(placebo_blackout, tarefas, chunksize=10))
    GL["PB"] = PB
    mpi = M[("atual", "sem", 6)]; d_real = mpi["fp"] - M[("atual", "com", 6)]["fp"]; PPI = []
    for nome, s in W.items():
        if nome.startswith("plac_"):
            q = substitui(BASES["atual"].copy(), "p", ew(s, "p")); al = decide(q, det); m = P.mede(al, vig_c, trips); PPI.append((nome[5:], m["fp"] - M[("atual", "com", 6)]["fp"], m["det"]))
    GL["PPI"] = PPI; GL["d_pi"] = (d_real, mpi["det"])
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(escreve=False)
        print(f"TESTE: encanamento ok em {time.time() - t0:.0f} s (saída suprimida). {ok}", flush=True); return
    relata(escreve=True); print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


def relata(escreve):
    M, AL, trips, op = GL["M"], GL["AL"], GL["trips"], GL["op"]
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    print("\nCONFIGURAÇÕES (detecção de pé | início | banda | FP | FP/mês [IC] | horas | contra X = 6 com PI: Δdet [IC], ΔFP/mês [IC], Δh/mês [IC])", flush=True)
    for b in GL["BASES"]:
        for pi in ("com", "sem"):
            for X in XS:
                m = M[(b, pi, X)]; c = m["cmp"]
                print(f" [{b:7s}] PI {pi} X={X:2d} | {m['det']}/8 | {m['inicio']} | {m['banda']} | FP {m['fp']:2d} | {m['fp_mes']:.3f} [{m['ic']['fp_mes'][0]:.2f}-{m['ic']['fp_mes'][1]:.2f}] | {m['horas']:5.0f} h | "
                      f"Δdet {c['d_det']:+d} [{c['ic_d_det'][0]:+.0f}, {c['ic_d_det'][1]:+.0f}] ΔFP {c['d_fp_mes']:+.2f} [{c['ic_d_fp_mes'][0]:+.2f}, {c['ic_d_fp_mes'][1]:+.2f}] "
                      f"Δh {c['d_horas_mes']:+.1f} [{c['ic_d_horas_mes'][0]:+.1f}, {c['ic_d_horas_mes'][1]:+.1f}]"
                      + ("" if b != "atual" else f" | perde {sorted(f'{t:%d/%m}' for t in M[(b, 'com', 6)]['det_set'] - m['det_set'])}"), flush=True)
    print("\nCHECAGEM POR TRIP (base atual, X = 6 com PI): última partida antes do trip, início do episódio que o detecta e se ele cai dentro de [partida, partida + X)", flush=True)
    partidas = op.index[(op & ~op.shift(fill_value=False)).to_numpy()]; eps = episodios_de_serie(AL[("atual", "com", 6)].loc[P.INICIO:P.FIM]); linhas = []
    for t in trips:
        pa = partidas[partidas < t][-1]; e = [(a, b) for a, b in eps if a <= t and b >= t - JANELA]
        ini = e[0][0] if e else None; dentro = {X: (ini is not None and pa <= ini < pa + pd.Timedelta(hours=X)) for X in XS}
        det_X = {X: t in M[("atual", "com", X)]["det_set"] for X in XS}
        linhas.append(dict(trip=loc(t), ultima_partida=loc(pa), h_partida_ate_trip=round((t - pa).total_seconds() / 3600, 1), inicio_episodio=loc(ini) if ini is not None else "não detectado",
                           h_partida_ate_inicio=round((ini - pa).total_seconds() / 3600, 1) if ini is not None else np.nan, **{f"inicio_na_janela_{X}h": dentro[X] for X in XS[1:]},
                           **{f"detecta_{X}h": det_X[X] for X in XS}))
    T = pd.DataFrame(linhas); pd.set_option("display.width", 260); print(T.to_string(index=False), flush=True)
    print("\nOS 9 FP DO X = 6 (base atual, com PI): destino em cada configuração (sai | encurta | igual)", flush=True)
    fp6 = [(a, b) for a, b in eps if not any(a <= t and b >= t - JANELA for t in trips)]; tab = []
    for a, b in fp6:
        lin = dict(fp=loc(a), dur_h=round((b - a).total_seconds() / 3600, 1))
        for pi in ("com", "sem"):
            for X in XS:
                if (pi, X) == ("com", 6):
                    continue
                al = AL[("atual", pi, X)].loc[a:b]; h = float(al.sum()) * 2 / 60; d = (b - a).total_seconds() / 3600 + 2 / 60
                lin[f"{'semPI ' if pi == 'sem' else ''}{X}h"] = "sai" if h == 0 else ("encurta" if h < d - 0.05 else "igual")
        tab.append(lin)
    novos = {}
    for pi in ("com", "sem"):
        for X in XS:
            e2 = episodios_de_serie(AL[("atual", pi, X)].loc[P.INICIO:P.FIM])
            novos[f"{'semPI ' if pi == 'sem' else ''}{X}h"] = sum(1 for a, b in e2 if not any(a <= t and b >= t - JANELA for t in trips) and not any(a <= y and b >= x for x, y in fp6))
    F = pd.DataFrame(tab); print(F.to_string(index=False), flush=True); print("   FP novos (que não sobrepõem nenhum dos 9): " + str({k: v for k, v in novos.items() if k != "6h"}), flush=True)
    print("\nPLACEBO DO BLACKOUT (mesmas horas extras em blocos sorteados; p = fração com FP <= o real e detecção >= a real)", flush=True)
    PB = pd.DataFrame(GL["PB"], columns=["base", "pi", "X", "fp", "det"])
    for b in GL["BASES"]:
        for X in XS[1:]:
            r = M[(b, "com", X)]; s = PB[(PB.base == b) & (PB.X == X)]; p = float(((s.fp <= r["fp"]) & (s.det >= r["det"])).mean())
            print(f"   [{b:7s}] X={X:2d}: real FP {r['fp']} det {r['det']} | placebo FP mediano {s.fp.median():.0f}, det mediana {s.det.median():.0f} | p = {p:.3f}", flush=True)
    d_real, det_pi = GL["d_pi"]; pp = [(n, d, dd) for n, d, dd in GL["PPI"]]; p_pi = float(np.mean([(d <= d_real) and (dd >= det_pi) for _, d, dd in pp])) if pp else np.nan
    print(f"\nPLACEBO DA PI: tirar a PI_5134001 muda o FP em {d_real:+d} (det {det_pi}/8); tirar cada uma das outras: {sorted(pp, key=lambda x: x[1])} | p = {p_pi:.3f}", flush=True)
    print(f"\nREGRA (base atual): PASSO 1 sem PI: aceita no conjunto {GL['passo1']}; LOEO perde trip escondido em {sum(GL['loeo1'])} recortes -> PI {'SAI' if GL['pi_esc'] == 'sem' else 'FICA'}", flush=True)
    print(f"   PASSO 2 janela escolhida com os 8 trips: X = {GL['Xesc']} h; LOEO (trip escondido, janela escolhida sem ele, perdeu?): {[(f'{k:%d/%m}', X, p) for k, X, p in GL['loeo2']]}", flush=True)
    rb = {b: (sorted(f"{t:%d/%m}" for t in M[(b, GL['pi_esc'], 6)]['det_set'] - M[(b, GL['pi_esc'], GL['Xesc'])]['det_set']),
              M[(b, GL['pi_esc'], GL['Xesc'])]['fp'] - M[(b, 'com', 6)]['fp']) for b in GL["BASES"]}
    print(f"   -> {'ACEITA: X = ' + str(GL['Xesc']) + ' h' if GL['aceita'] else 'NENHUMA JANELA > 6 h ACEITA (fica 6 h)'} | PI {'sai' if GL['pi_esc'] == 'sem' else 'fica'} | robustez (trips perdidos, ΔFP) por base: {rb}", flush=True)
    if escreve:
        pd.DataFrame([dict(base=b, pi=pi, X=X, **{k: v for k, v in m.items() if k not in ("ic", "cmp", "det_set")}, ic_fp_mes=m["ic"]["fp_mes"], ic_det=m["ic"]["det"],
                           d_det=m["cmp"]["d_det"], ic_d_det=m["cmp"]["ic_d_det"], d_fp_mes=m["cmp"]["d_fp_mes"], ic_d_fp_mes=m["cmp"]["ic_d_fp_mes"], d_horas_mes=m["cmp"]["d_horas_mes"],
                           ic_d_horas_mes=m["cmp"]["ic_d_horas_mes"]) for (b, pi, X), m in M.items()]).to_csv(SAIDA / "h7_configuracoes.csv", index=False)
        T.to_csv(SAIDA / "h7_checagem_por_trip.csv", index=False); F.to_csv(SAIDA / "h7_fp_antes_depois.csv", index=False)
        pd.DataFrame([dict(config=k, fp_novos=v) for k, v in novos.items()]).to_csv(SAIDA / "h7_fp_novos.csv", index=False)
        pd.DataFrame([dict(decisao="pi_sai", valor=GL["pi_esc"] == "sem"), dict(decisao="janela", valor=GL["Xesc"]), dict(decisao="aceita", valor=GL["aceita"])]).to_csv(SAIDA / "h7_decisao.csv", index=False)


if __name__ == "__main__":
    main()
