#!/usr/bin/env python3
"""EXP 56 — ETAPA 4 DA REANÁLISE POR MODO: um sub-canal de SELAGEM dentro do voto da família de pressão.

PRÉ-REGISTRO (20/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp56.py`), branch exp/modos-de-falha.

POR QUÊ (medido no EXP 55). A selagem é o modo MAIS observável nas nossas tags (percentil da família 0,93-0,99, e sobe perto do trip), mas o detector a captura mal: no 29/04/2025
(trip de selagem, PDI_0302) o canal p acendeu por um sensor de ÓLEO, não de selagem. Causa estrutural conhecida: o p é um PCA de 12 pressões com 1 componente para 95% da variância,
dominado pelo PI_0319 (gás do motor de partida, 97,9% da variância escalada) — `canal-p-univariado-por-escala`; e no EXP 52 só duas tags do p importam, PI_0319 e PDIT_0305.
A hipótese: dar à selagem um sub-canal próprio (PCA das 4 tags de selagem), combinado ao resto das pressões pelo MÁXIMO, mantendo UM voto da família de pressão (o voto, os k,
o CUSUM, o refratário e o portão B[sp]|B[vb] ficam intactos), faz o p acender pela física certa nos trips de selagem sem custar FP.

O QUE SE FAZ. Harness fiel do EXP 52 (`walk_forward(com_sk(PCAfiel))`: RobustScaler, PCA(0,95) com k mensal, sens_p99 por sensor, recon_p99 do treino, retreino mensal causal),
n = 20.000 e 19.000. Cada sub-canal passa pelo EWMA de produção do p (1 h); o voto da família recebe o MÁXIMO dos sub-canais suavizados; substitui o p no `proc` do cache e a
decisão de produção (`ci.prever`) roda como caixa-preta. O pacote não é editado.
  FAMÍLIA SELAGEM (fixada no EXP 55 pela planilha): PDIT_0305, PDI_0302, PDI_0301, PI_0307. RESTO: as outras 8 tags do p do bundle.
BRAÇOS:
  C0  p das 12 tags (controle, reajustado pelo mesmo harness).
  S1  p := max(PCA selagem 4, PCA resto 8) — a SEPARAÇÃO (principal).
  S2  p := max(PCA das 12, PCA selagem 4) — o ACRÉSCIMO (o p atual mais um sub-canal de selagem).
  R0  p := PCA resto 8 — diagnóstico (o que a selagem contribui hoje), não candidato.
PLACEBOS DE DESENHO (a peça que decide): 6 subconjuntos de 4 tags do p sorteados (semente 20261022, excluído o conjunto exato da selagem). Para S1: max(PCA do sorteio, PCA do
complemento de 8). Para S2: max(PCA das 12, PCA do sorteio). Se qualquer partição ou acréscimo der o mesmo, o efeito é do "segundo sorteio no máximo", não da física da selagem.
MEDIDAS: detecção de 8, início, banda, FP, FP/mês, horas totais e fora, placebo de eventos (+45 a +225 d), detecção por evento; MECANISMO nos trips de selagem (27/02 e 29/04):
fração do tempo em que o p está aceso na janela de 48 h e, desse tempo, a fração em que o sub-canal de selagem é o máximo; duty do p (nível A) no período vigiado.
CRITÉRIOS (fixados antes; médias sobre os dois n):
  GANHO = [(det >= 7 e FP <= 8) ou (det >= 8 e FP <= 9)] E horas <= 1,10 x horas do C0 E melhor que a MEDIANA dos seus placebos de desenho (FP menor e detecção não menor).
  PIOR = det <= 5 OU FP >= 12. EQUIVALENTE no resto. META DO USUÁRIO (à parte) = FP <= 6 e det >= 6.
  MECANISMO (relatado, não é portão) = no 29/04 o p fica aceso na janela e a selagem é o máximo em >= 50% desse tempo.
EXPECTATIVAS: V1-V3 passam (~90%); S1 GANHO (~15%); S2 GANHO (~10%); MECANISMO em S1 (~55%); S1 ou S2 com FP >= 10, pela deriva de selagem/filtro depois de manutenção já
documentada (~45%); R0 com detecção <= 6 (~60%; o PDIT_0305 era essencial no EXP 52); mediana dos placebos a +-1 FP do controle (~60%); nenhum braço atinge a META (~85%).
INTERPRETAÇÃO PREVISTA. GANHO com MECANISMO: candidato à sombra (muda o canal p congelado; decisão do usuário). GANHO sem MECANISMO, ou empatado com os placebos: efeito de
partição, não de física — não se adota. Sem GANHO: a informação de selagem já entra pelo PDIT_0305 no p, e a etapa 4 da selagem fecha; o que resta da reanálise é o óleo (dado).
LIMITES: 8 eventos, 2 de selagem; o PCA de 4 tags é outro ajuste; o máximo de dois sub-canais aumenta o duty por construção (por isso os placebos de desenho); nada se adota sem a
sombra.
VERIFICAÇÕES: V1 o C0 reajustado reproduz `escores_sub34/p_A_{n}` (diferença <= 1e-6); V2 `prever` com o C0 dá 7/8 e FP 9 +- 1; V3 selagem + resto particionam exatamente as 12
tags do p do bundle.

RESULTADO (20/10/2026; worker do ClearML, tarefa 931e47d25a284159be2b1923b8573630, 15 s, 64 CPUs; pré-registro b344790, código enviado como commit e86c51e + diff do remoto;
a tarefa a85d4f69 falhou antes de rodar porque registrou o commit local, não publicado). V1 (dif. 0 e 2e-16), V2 (7/8, FP 9) e V3 PASSAM. SEM GANHO: a selagem já está no p.
    braço (média dos 2 n)        det   FP     horas   duty p   ||  placebos de desenho: det / FP medianos
    C0 controle (12)             7,0   9,0    1.456   0,34-0,40
    S1 separação                 7,0  10,0    1.482   0,33-0,35 ||  partição  7,0 / 11,8   -> EQUIVALENTE
    S2 acréscimo                 7,0   9,5    1.484   0,36-0,42 ||  acréscimo 7,0 / 10,2   -> EQUIVALENTE
    R0 sem selagem (diagnóstico) 6,0  12,5    1.516   0,23-0,27      perde o 27/02 (trip de SELAGEM) e o início cai de 6 para 4
  1) A INFORMAÇÃO DE SELAGEM JÁ ENTRA PELO p: tirar as 4 tags de selagem custa o trip de selagem 27/02, 2 de início e +3,5 FP. É o mesmo achado do EXP 52 (PDIT_0305 essencial),
     agora com o mecanismo nomeado: o p do v2 detecta o 27/02 PELA selagem (p aceso 100% da janela, selagem no máximo 92-100% do tempo em S1/S2).
  2) O MECANISMO DO 29/04 EXISTE MAS É PEQUENO: com o sub-canal, o p passa a acender em 4-5% da janela de 48 h do 29/04, e nesse tempo é a selagem que manda (no C0, 0%). Não muda a
     detecção (o 29/04 já era detectado pelo episódio de 27/04) nem o início.
  3) A FÍSICA DA SELAGEM IMPORTA EM RELAÇÃO AO ACASO DO DESENHO, MAS NÃO GANHA DO CONTROLE: a partição pela selagem (FP 10,0) é melhor que as partições sorteadas (mediana 11,8), e o
     acréscimo de selagem (9,5) melhor que os acréscimos sorteados (10,2) — qualquer segundo sub-canal no máximo custa FP, e o de selagem custa menos. Mas nenhum fica abaixo dos 9
     FP do controle; o critério de GANHO exigia as duas coisas.
PLACAR: ACERTEI: V1-V3 (~90%); S1 sem GANHO (dei ~15% para ganhar); S2 sem GANHO (~10%); MECANISMO em S1 (~55%; mas fraco, 4-5% da janela); S1 ou S2 com FP >= 10 (~45%; S1 10,0);
  R0 com detecção <= 6 (~60%; perdeu o 27/02); nenhum braço na META (~85%). ERREI: mediana dos placebos a +-1 FP do controle (~60%) — os placebos ficaram 1,2 a 2,8 FP PIORES: dividir
  ou acrescentar sub-canal no p custa FP por construção.
VEREDITO (como pré-registrado): sem GANHO, a etapa 4 da selagem FECHA. Fecho da reanálise por modo: mancal e selagem já são detectados pela física certa (5 de 7 detecções causais,
EXP 55; o p carrega a selagem); o que falta é o ÓLEO, sem precursor observável nas nossas tags, e o estado do vb que sustenta FP e acertos. O próximo ganho vem de dado (sistema de
óleo; as 47 tags) e do rótulo dos FP pela operação, não de reestruturar canal.

Uso local (encanamento, sem tabelas):  EXP56_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/56_subcanal_de_selagem.py
No worker:  PYTHONPATH=src python scripts/worker_exp56.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import JANELA
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward

TESTE = bool(os.environ.get("EXP56_TESTE"))
ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; CACHE = ROOT / ("escores_sel56_teste" if TESTE else "escores_sel56")
NS = (20000,) if TESTE else (20000, 19000); N_PLAC = 1 if TESTE else 6; SEMENTE = 20261022; GL = {}
P = "954005_624_"
SELAGEM = [P + "PDIT_0305", P + "PDI_0302", P + "PDI_0301", P + "PI_0307"]


def arquivo(nome, n):
    return CACHE / f"p_{nome}_{n}.parquet"


def pontua(a):
    nome, tags, n = a; f = arquivo(nome, n)
    if f.exists():
        return a
    t0 = time.time()
    s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), GL["g"], tags, GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f)
    print(f"      {nome:8s} n={n} ({len(tags)} tags): {time.time() - t0:.0f} s", flush=True); return a


def suave(nome, n):
    s = pd.read_parquet(arquivo(nome, n))["s"]
    return s.ewm(halflife=pd.Timedelta(GL["hl_p"]), times=s.index).mean().where(s.notna())


def maximo(*xs):
    return pd.concat(xs, axis=1).max(axis=1)


def mede(e):
    res = decide_v2(substitui(GL["proc"], "p", e), GL["det"])
    al = res["is_anomaly"].astype(bool).reindex(GL["al_c"].index).fillna(False)
    r = avalia(al, GL["vig_c"], GL["EV"]["real"], inicio_periodo=INICIO8)
    pl = [avalia(al, GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev) for k, ev in GL["EV"].items() if k != "real" and ev]
    vig = res["vigiado"].astype(bool).loc[INICIO8:]
    duty = float(res["a_p"].astype(bool).loc[INICIO8:][vig].mean())
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora,
                placebo=float(np.median(pl)) * 8 if pl else np.nan, duty_p=duty, ev={k[:10]: v["det"] for k, v in r.por_evento.items()}), res


def mecanismo(res, e_sel, e_outro):
    out = {}
    for T in GL["trips_selagem"]:
        w = slice(T - JANELA, T); aceso = (res["a_p"].astype(bool) | res["b_p"].astype(bool)).loc[w]
        dom = (e_sel.reindex(aceso.index) >= e_outro.reindex(aceso.index)).fillna(False)
        out[T.strftime("%d/%m")] = (float(aceso.mean()), float(dom[aceso].mean()) if aceso.any() else np.nan)
    return out


def main():
    procs = int(os.environ.get("EXP56_PROCS", 6)); ci = _ci(); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; tags_p = list(modelos[-1]["normalizacao"]["pressao"]["cols"])
    resto = [c for c in tags_p if c not in SELAGEM]
    ok = {"V3": sorted(SELAGEM + resto) == sorted(tags_p) and len(resto) == 8}
    print(f"V3 — selagem {[c.replace(P, '') for c in SELAGEM]} + resto {[c.replace(P, '') for c in resto]} = as {len(tags_p)} tags do p -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    if not ok["V3"]:
        return
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    if TESTE:
        fim = INICIO8 + pd.Timedelta(days=20)
    EV = {"real": [t for t in reg if t <= vig_c.index[-1]]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= vig_c.index[-1]]
    GL.update(g=g, est=est, fim=fim, ini=INICIO8 - pd.Timedelta(days=10), det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV, hl_p=det["halflife"]["p"],
              trips_selagem=[pd.Timestamp("2025-02-27 08:38", tz="UTC"), pd.Timestamp("2025-04-29 03:04", tz="UTC")])
    rng = np.random.default_rng(SEMENTE); sorteios = []
    while len(sorteios) < N_PLAC:
        s = sorted(rng.choice(tags_p, 4, replace=False).tolist())
        if s != sorted(SELAGEM) and s not in sorteios:
            sorteios.append(s)
    conj = {"p12": tags_p, "sel4": SELAGEM, "resto8": resto}
    for i, s in enumerate(sorteios):
        conj[f"plA{i + 1}"] = s; conj[f"plB{i + 1}"] = [c for c in tags_p if c not in s]
    print(f"EXP 56: {len(conj)} conjuntos x {len(NS)} n | PROCS {procs} | teste {TESTE}", flush=True)
    for i, s in enumerate(sorteios):
        print(f"   placebo {i + 1}: {[c.replace(P, '') for c in s]} (sobreposição com a selagem: {len(set(s) & set(SELAGEM))})", flush=True)
    with Pool(procs) as pool:
        list(pool.imap_unordered(pontua, [(nome, tags, n) for nome, tags in conj.items() for n in NS], chunksize=1))
    print(f"ESCORES em {time.time() - t0:.0f} s", flush=True)
    for n in NS:
        f34 = ROOT / "escores_sub34" / f"p_A_{n}.parquet"
        a = pd.read_parquet(f34)["s"]; b = pd.read_parquet(arquivo("p12", n))["s"]; j = a.index.intersection(b.index)
        d = float(np.nanmax(np.abs(a.loc[j].to_numpy() - b.loc[j].to_numpy()))); ok[f"V1_{n}"] = d <= 1e-6
        print(f"V1 [n={n}] C0 reajustado x escores_sub34: diferença máxima {d:.2e} -> {'PASSA' if ok[f'V1_{n}'] else 'FALHA'}", flush=True)

    BR = {}
    for n in NS:
        e = {k: suave(k, n) for k in conj}
        BR[("C0 controle (12)", n)] = (e["p12"], None, None)
        BR[("S1 separação", n)] = (maximo(e["sel4"], e["resto8"]), e["sel4"], e["resto8"])
        BR[("S2 acréscimo", n)] = (maximo(e["p12"], e["sel4"]), e["sel4"], e["p12"])
        BR[("R0 sem selagem", n)] = (e["resto8"], None, None)
        for i in range(N_PLAC):
            BR[(f"PS1-{i + 1} partição", n)] = (maximo(e[f"plA{i + 1}"], e[f"plB{i + 1}"]), None, None)
            BR[(f"PS2-{i + 1} acréscimo", n)] = (maximo(e["p12"], e[f"plA{i + 1}"]), None, None)
    R = {}
    for k, (e, es, eo) in BR.items():
        L, res = mede(e); L["mec"] = mecanismo(res, es, eo) if es is not None else None; R[k] = L
    c0 = R[("C0 controle (12)", NS[0])]; ok["V2"] = abs(c0["det"] - 7) == 0 and abs(c0["fp"] - 9) <= 1
    print(f"V2 — prever com o C0 reajustado: {c0['det']}/8, FP {c0['fp']} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(R, BR, det, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (janela curta; saída suprimida, nada gravado). V3 {ok['V3']}, V1 {ok['V1_20000']}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(R, BR, det, ok, t0, escreve=True)


def relata(R, BR, det, ok, t0, escreve):
    print("\nRESULTADOS (det/8 | início | banda | FP | FP/mês | horas | fora | placebo | duty do p no nível A)", flush=True)
    for (nome, n), L in R.items():
        print(f"   {nome:22s} n={n} | {L['det']}/8 | {L['inicio']} | {L['banda']} | FP {L['fp']:2d} | {L['fp_mes']:.3f} | {L['horas']:5.0f} h | {L['fora']:4.0f} h | {L['placebo']:.1f} | {L['duty_p']:.3f}", flush=True)
    print("\nDETECÇÃO POR EVENTO (n=20000)", flush=True)
    for nome in ("C0 controle (12)", "S1 separação", "S2 acréscimo", "R0 sem selagem"):
        print(f"   {nome:22s} " + " ".join(f"{k[8:10]}/{k[5:7]}:{int(v)}" for k, v in R[(nome, NS[0])]["ev"].items()), flush=True)
    print("\nMECANISMO nos trips de selagem (fração da janela de 48 h com o p aceso; desse tempo, fração com a selagem no máximo)", flush=True)
    for nome in ("S1 separação", "S2 acréscimo"):
        for n in NS:
            print(f"   {nome:22s} n={n}: " + " | ".join(f"{d}: p aceso {a:.2f}, selagem domina {b:.2f}" for d, (a, b) in R[(nome, n)]["mec"].items()), flush=True)
    for n in NS:
        print(f"   C0 n={n}: p aceso na janela " + " | ".join(f"{T.strftime('%d/%m')}: {float((decide_v2(substitui(GL['proc'], 'p', BR[('C0 controle (12)', n)][0]), det)[['a_p', 'b_p']].astype(bool).any(axis=1)).loc[T - JANELA:T].mean()):.2f}" for T in GL["trips_selagem"]), flush=True)

    med = lambda nome, ch: float(np.mean([R[(nome, n)][ch] for n in NS]))
    hc = med("C0 controle (12)", "horas")
    print("\nCLASSIFICAÇÃO (médias sobre os n)", flush=True)
    for cand, pref in (("S1 separação", "PS1-"), ("S2 acréscimo", "PS2-")):
        pls = [k for k in dict.fromkeys(nm for nm, _ in R) if k.startswith(pref)]
        mfp = float(np.median([med(k, "fp") for k in pls])); mdet = float(np.median([med(k, "det") for k in pls]))
        d, fp, h = med(cand, "det"), med(cand, "fp"), med(cand, "horas")
        melhor = fp < mfp and d >= mdet
        cl = "GANHO" if (((d >= 7 and fp <= 8) or (d >= 8 and fp <= 9)) and h <= 1.10 * hc and melhor) else "PIOR" if (d <= 5 or fp >= 12) else "EQUIVALENTE"
        a29 = [R[(cand, n)]["mec"]["29/04"] for n in NS]; mec = all(a > 0 and b >= 0.5 for a, b in a29)
        meta = " | META DO USUÁRIO" if (fp <= 6 and d >= 6) else ""
        print(f"   {cand:22s} det {d:.1f} | FP {fp:.1f} | horas {h:.0f} (C0 {hc:.0f}) | placebos: det mediana {mdet:.1f}, FP mediana {mfp:.1f} -> {cl} | MECANISMO {'sim' if mec else 'não'}{meta}", flush=True)
    print(f"   R0 sem selagem (diagnóstico): det {med('R0 sem selagem', 'det'):.1f} | FP {med('R0 sem selagem', 'fp'):.1f} | horas {med('R0 sem selagem', 'horas'):.0f}", flush=True)
    if escreve:
        pd.DataFrame([dict(braco=nm, n=n, **{k: v for k, v in L.items() if k not in ("ev", "mec")}, ev=str(L["ev"]), mec=str(L["mec"])) for (nm, n), L in R.items()]).to_csv(ROOT / "exp56_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
