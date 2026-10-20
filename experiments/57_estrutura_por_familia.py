#!/usr/bin/env python3
"""EXP 57 — ESTRUTURA POR FAMÍLIA FÍSICA: um canal por família (mancal, selagem, óleo, exaustão) mais a vibração, votando A CUSTO IGUAL ao controle.

PRÉ-REGISTRO (20/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp57.py`), branch exp/modos-de-falha. Pedido do usuário: "testar modelos nesse tipo
de estrutura agora, para cada grupo". Expectativa declarada antes: ~10% de ganho; o objetivo é FECHAR a questão da estrutura por grupos com evidência.

POR QUÊ. A reanálise por modo (EXP 55) mostrou que os canais atuais misturam físicas: o t junta mancais do compressor com a exaustão da turbina; o p junta selagem, óleo e o gás de
partida (PI_0319, que domina o PCA e não é causa de nenhum trip do alvo). A pergunta: reorganizar os canais pelas FAMÍLIAS FÍSICAS, um PCA por família, melhora a detecção ou o FP?
Precedentes contra: EXP 50 (um modelo por grupo nos 4 canais atuais, 0 ganho), EXP 56 (sub-canal de selagem empata; canal a mais no máximo custa FP), EXP 23 (quinto canal +2 a +4
FP) e a literatura de monitoramento por blocos (equivalente ao PCA único na detecção; o ganho é de localização).

DESENHO (fixado antes; nada é ajustado aos eventos):
  CANAIS F: mancal (TI_0301/0303/0305/0307), selagem (PDIT_0305, PDI_0302, PDI_0301, PI_0307), óleo (PI_0339, PI_0340, PI_0308, PDI_0338, TI_0325), exaustão (T5_AVG_A, TC382_01..06,
  TI_0315, TI_0317) — as famílias do EXP 55 — e o vb de produção, inalterado. O GÁS (PI_0315, PI_0319, PDI_0317, PI_5134001) FICA DE FORA a priori: não é modo de nenhum trip do
  alvo (definição do alvo, não os eventos). O sp sai: a informação dele (TI_0305 contra os irmãos) é quebra de correlação dentro do PCA do mancal.
  MODELO por família: o mesmo PCA fiel do v2 (`walk_forward(com_sk(PCAfiel))`, RobustScaler, PCA 0,95, k mensal, sens_p99, recon_p99, retreino mensal causal), n = 20.000 e 19.000;
  escore normalizado pelo p99 do treino, como o t e o p; EWMA de 1 h.
  LIMIARES: os do t (A 2,2 / B 3,4) para as famílias de temperatura (mancal, exaustão), os do p (A 2,4 / B 3,4) para as de pressão (selagem, óleo), os do vb para o vb; todos
  multiplicados por UM fator global f, escolhido SEM RÓTULOS para igualar as horas totais de alarme do controle (1.457 h, tolerância +-5%; bisseção em f).
  VOTO: nível A >= 3 canais; nível B >= 2 com portão mecânico B[mancal] | B[vb] (o análogo do B[sp] | B[vb]); escalada, refratário 72 h e duração mínima do pacote (reimplementação
  do EXP 51, provada idêntica ao `prever`).
  SECUNDÁRIO F' (informado pelo EXP 55, declarado): F sem o óleo (4 canais), porque o óleo não tem precursor observável.
  PLACEBOS DE PARTIÇÃO (6, semente 20261023): as mesmas 22 tags sorteadas em grupos de tamanhos 4, 4, 5 e 9, nas mesmas posições (o 1º grupo faz o papel do mancal no portão; o 1º e o
  4º usam os limiares do t), mesmo f igualado. Se a partição sorteada der o mesmo, a física da família não importa.
  LOEO não se aplica: nenhuma escolha usa os eventos (estrutura a priori, custo igualado sem rótulo). Robustez: os dois n.
MEDIDAS: f, detecção de 8, início, banda, FP, FP/mês, horas totais e fora, placebo de eventos (+45 a +225 d), detecção por evento, duty de cada canal no nível A.
CRITÉRIOS (fixados antes; médias sobre os n, a custo igual):
  GANHO = [(det >= 7 e FP <= 7) ou (det >= 8 e FP <= 9)] E melhor que a MEDIANA dos placebos de partição (FP menor e detecção não menor).
  PIOR = det <= 5 OU FP >= 12. EQUIVALENTE no resto. META DO USUÁRIO (à parte) = FP <= 6 e det >= 6.
EXPECTATIVAS: V1-V4 passam (~90%); F GANHO (~8%), EQUIVALENTE (~52%), PIOR (~40%); F' com menos FP que F (~55%); F melhor que a mediana dos placebos (~45%); F detecta os 2 trips de
selagem (~70%); o 04/11 continua perdido (~85%); nenhum braço na META (~90%).
INTERPRETAÇÃO PREVISTA: GANHO acima dos placebos -> a estrutura física importa; candidato à sombra (troca a arquitetura congelada, decisão do usuário). EQUIVALENTE -> confirma a
literatura (blocos não mudam a detecção); a estrutura por grupos FECHA e a família vira só camada de diagnóstico. PIOR -> o v2 atual, com o gás e o sp, carrega algo que a
organização física perde (provavelmente o estado do PI_0319 e o sp como votante próprio).
LIMITES: 8 eventos; o f iguala HORAS, não FP (as duas medidas vão juntas); a escolha das famílias usa a planilha e o P&ID, mas foi feita depois de ver o EXP 55; nada se adota sem a
sombra.
VERIFICAÇÕES: V1 a decisão genérica com os 4 canais do v2 e f = 1 reproduz `ci.prever` instante a instante; V2 as 4 famílias têm 22 tags disjuntas, todas na grade, sem o gás;
V3 o PCA das 12 tags do p reproduz `escores_sub34/p_A_{n}` (diferença <= 1e-6); V4 o f iguala as horas a +-5% em F e em >= 5 dos 6 placebos.

Uso local (encanamento, sem tabelas):  EXP57_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/57_estrutura_por_familia.py
No worker:  PYTHONPATH=src python scripts/worker_exp57.py --enqueue --queue default
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


E51 = _carrega("exp51", "51_estrutura_do_voto.py"); E46 = E51.E46
E55 = _carrega("exp55", "55_modos_de_falha.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.walk import walk_forward

TESTE = bool(os.environ.get("EXP57_TESTE"))
ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; CACHE = ROOT / ("escores_fam57_teste" if TESTE else "escores_fam57")
NS = (20000,) if TESTE else (20000, 19000); N_PLAC = 1 if TESTE else 6; SEMENTE = 20261023; GL = E51.GL
FAMS = ("mancal", "selagem", "oleo", "exaustao"); REF = {"mancal": "t", "selagem": "p", "oleo": "p", "exaustao": "t"}
TAMANHOS = (4, 4, 5, 9); TOL = 0.05


def arquivo(nome, n):
    return CACHE / f"{nome}_{n}.parquet"


def pontua(a):
    nome, tags, n = a; f = arquivo(nome, n)
    if f.exists():
        return a
    t0 = time.time()
    s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), GL["g"], tags, GL["est"], INICIO8, GL["fim_walk"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f)
    print(f"      {nome:10s} n={n} ({len(tags)} tags): {time.time() - t0:.0f} s", flush=True); return a


def suave(nome, n, hl):
    s = pd.read_parquet(arquivo(nome, n))["s"]
    return s.ewm(halflife=pd.Timedelta(hl), times=s.index).mean().where(s.notna())


def alarme_generico(px, canais, limA, limB, gate, f):
    """O voto do v2 com N canais: A >= 3, B >= 2 com portão B[gate0] | B[gate1]; pós-processamento do pacote (E51.pos_processa)."""
    ci = GL["ci"]; det = GL["det"]; mask = px["mask"].astype(bool); reset = px["reset"].to_numpy()
    A = {c: ci._acende(px[c], limA[c] * f, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in canais}
    B = {c: ci._acende(px[c], limB[c] * f, mask, reset, det["kappa"], det["h_cusum"], det["carga_reset"]) for c in canais}
    voto = ((sum(A[c].astype(int) for c in canais) >= det["voto_nivel_a"]) & mask) | \
           ((sum(B[c].astype(int) for c in canais) >= det["voto_nivel_b"]) & mask & (B[gate[0]] | B[gate[1]]))
    forca = pd.concat([px[c].where(mask) / (limB[c] * f) for c in canais], axis=1).max(axis=1)
    return E51.pos_processa(voto, forca, det), {c: float(A[c].loc[INICIO8:][mask.loc[INICIO8:]].mean()) for c in canais}


def mede(al):
    a = al.reindex(GL["al_c"].index).fillna(False)
    r = avalia(a, GL["vig_c"], GL["EV"]["real"], inicio_periodo=INICIO8)
    pl = [avalia(a, GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev) for k, ev in GL["EV"].items() if k != "real" and ev]
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora,
                placebo=float(np.median(pl)) * 8 if pl else np.nan, ev={k[:10]: int(v["det"]) for k, v in r.por_evento.items()})


def monta(grupos, n):
    """proc com um canal por grupo (PCA da família, EWMA 1 h) + o vb de produção; limiares por posição."""
    det = GL["det"]; px = GL["proc"][["vb", "mask", "reset"]].copy(); limA, limB = {}, {}
    for (nome, tags), ref in grupos:
        px[nome] = suave(nome, n, det["halflife"][ref]).reindex(px.index)
        limA[nome] = det["base"][ref] * det["k_nivel_a"][ref]; limB[nome] = det["base"][ref] * det["k_nivel_b"][ref]
    limA["vb"] = det["base"]["vb"] * det["k_nivel_a"]["vb"]; limB["vb"] = det["base"]["vb"] * det["k_nivel_b"]["vb"]
    return px, [nome for (nome, _), _ in grupos] + ["vb"], limA, limB


def iguala(a):
    """Escolhe f (sem rótulos) para igualar as horas totais de alarme ao controle; devolve as medidas nesse f."""
    rotulo, grupos, n = a; px, canais, limA, limB = monta(grupos, n); gate = (canais[0], "vb"); alvo = GL["horas_alvo"]
    cache = {}

    def avalia_f(f):
        if f not in cache:
            al, duty = alarme_generico(px, canais, limA, limB, gate, f); cache[f] = (mede(al), duty)
        return cache[f]
    grade = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0) if not TESTE else (0.8, 1.0, 1.5)
    for f in grade:
        avalia_f(f)
    fs = sorted(cache); hs = [cache[f][0]["horas"] for f in fs]
    lo = hi = None
    for i in range(len(fs) - 1):
        if (hs[i] - alvo) * (hs[i + 1] - alvo) <= 0:
            lo, hi = fs[i], fs[i + 1]; break
    if lo is not None:
        for _ in range(3 if TESTE else 8):
            m = (lo + hi) / 2; h = avalia_f(m)[0]["horas"]
            if (avalia_f(lo)[0]["horas"] - alvo) * (h - alvo) <= 0:
                hi = m
            else:
                lo = m
    f_best = min(cache, key=lambda f: abs(cache[f][0]["horas"] - alvo)); L, duty = cache[f_best]
    return rotulo, n, f_best, L, duty


def main():
    procs = int(os.environ.get("EXP57_PROCS", 6)); ci = _ci(); GL["ci"] = ci; t0 = time.time(); ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; tags_p = list(modelos[-1]["normalizacao"]["pressao"]["cols"])
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(g=g, est=est, fim_walk=(INICIO8 + pd.Timedelta(days=20)) if TESTE else fim, ini=INICIO8 - pd.Timedelta(days=10), det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV)

    with contextlib.redirect_stdout(io.StringIO()):
        ref = ci.prever({"detector": det}, proc)["is_anomaly"].astype(bool)
    lim0A = {c: det["base"][c] * det["k_nivel_a"][c] for c in det["sinais"]}; lim0B = {c: det["base"][c] * det["k_nivel_b"][c] for c in det["sinais"]}
    al0, _ = alarme_generico(proc, list(det["sinais"]), lim0A, lim0B, ("sp", "vb"), 1.0)
    d1 = int((al0 != ref).sum()); ok["V1"] = d1 == 0
    L0 = mede(al0); GL["horas_alvo"] = L0["horas"]
    print(f"V1 — decisão genérica (4 canais do v2, f = 1) x ci.prever: {d1} instantes diferentes -> {'PASSA' if ok['V1'] else 'FALHA'} | controle {L0['det']}/8, FP {L0['fp']}, {L0['horas']:.0f} h", flush=True)
    fam = {f: E55.FAMILIAS[f] for f in FAMS}; todas = [c for f in FAMS for c in fam[f]]
    ok["V2"] = len(todas) == 22 and len(set(todas)) == 22 and all(c in g.columns for c in todas) and not (set(todas) & set(E55.FAMILIAS["gas"]))
    print(f"V2 — famílias {[(f, len(fam[f])) for f in FAMS]} = {len(todas)} tags disjuntas, sem o gás -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)

    rng = np.random.default_rng(SEMENTE); parts = []
    while len(parts) < N_PLAC:
        perm = rng.permutation(todas).tolist(); grupos, i = [], 0
        for k in TAMANHOS:
            grupos.append(sorted(perm[i:i + k])); i += k
        if grupos != [sorted(fam[f]) for f in FAMS] and grupos not in parts:
            parts.append(grupos)
    conj = {"p12": tags_p} | {f"F_{f}": fam[f] for f in FAMS}
    for j, grupos in enumerate(parts):
        for k, gr in enumerate(grupos):
            conj[f"PL{j + 1}_{k + 1}"] = gr
    print(f"EXP 57: {len(conj)} conjuntos x {len(NS)} n | PROCS {procs} | teste {TESTE}", flush=True)
    for j, grupos in enumerate(parts):
        print(f"   placebo {j + 1}: " + " | ".join(str(len(gr)) + ":" + ",".join(c.replace('954005_624_', '') for c in gr) for gr in grupos), flush=True)
    with Pool(procs) as pool:
        list(pool.imap_unordered(pontua, [(nome, tags, n) for nome, tags in conj.items() for n in NS], chunksize=1))
    print(f"ESCORES em {time.time() - t0:.0f} s", flush=True)
    for n in NS:
        a = pd.read_parquet(ROOT / "escores_sub34" / f"p_A_{n}.parquet")["s"]; b = pd.read_parquet(arquivo("p12", n))["s"]; j = a.index.intersection(b.index)
        d = float(np.nanmax(np.abs(a.loc[j].to_numpy() - b.loc[j].to_numpy()))); ok[f"V3_{n}"] = d <= 1e-6
        print(f"V3 [n={n}] PCA das 12 tags do p x escores_sub34: diferença máxima {d:.2e} -> {'PASSA' if ok[f'V3_{n}'] else 'FALHA'}", flush=True)

    F = [((f"F_{f}", fam[f]), REF[f]) for f in FAMS]
    braços = {"F famílias": F, "F' sem óleo": [x for x in F if x[0][0] != "F_oleo"]}
    pos_ref = [REF[f] for f in FAMS]
    for j in range(N_PLAC):
        braços[f"PL{j + 1} partição"] = [((f"PL{j + 1}_{k + 1}", None), pos_ref[k]) for k in range(len(TAMANHOS))]
    tarefas = [(rot, grupos, n) for rot, grupos in braços.items() for n in NS]
    with Pool(min(procs, len(tarefas))) as pool:
        RES = {(rot, n): (f, L, duty) for rot, n, f, L, duty in pool.imap_unordered(iguala, tarefas, chunksize=1)}
    print(f"DECISÕES em {time.time() - t0:.0f} s", flush=True)
    alvo = GL["horas_alvo"]
    ok_h = {k: abs(L["horas"] - alvo) <= TOL * alvo for k, (f, L, _) in RES.items()}
    ok["V4"] = all(ok_h[("F famílias", n)] for n in NS) and sum(ok_h[(f"PL{j + 1} partição", n)] for j in range(N_PLAC) for n in NS) >= (5 if not TESTE else 0) * len(NS)
    print(f"V4 — horas igualadas a +-5% de {alvo:.0f} h: F {[ok_h[('F famílias', n)] for n in NS]}, placebos {sum(ok_h[(f'PL{j + 1} partição', n)] for j in range(N_PLAC) for n in NS)}/{N_PLAC * len(NS)} -> {'PASSA' if ok['V4'] else 'FALHA'}", flush=True)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(RES, L0, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (janela curta; saída suprimida, nada gravado). V1 {ok['V1']}, V2 {ok['V2']}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(RES, L0, ok, t0, escreve=True)


def relata(RES, L0, ok, t0, escreve):
    print("\nRESULTADOS A CUSTO IGUAL (f | det/8 | início | banda | FP | FP/mês | horas | fora | placebo de eventos | duty A por canal)", flush=True)
    print(f"   {'C0 controle (v2)':22s}      | {L0['det']}/8 | {L0['inicio']} | {L0['banda']} | FP {L0['fp']:2d} | {L0['fp_mes']:.3f} | {L0['horas']:5.0f} h | {L0['fora']:4.0f} h | {L0['placebo']:.1f}", flush=True)
    for (rot, n), (f, L, duty) in sorted(RES.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"   {rot:22s} n={n} f={f:.3f} | {L['det']}/8 | {L['inicio']} | {L['banda']} | FP {L['fp']:2d} | {L['fp_mes']:.3f} | {L['horas']:5.0f} h | {L['fora']:4.0f} h | {L['placebo']:.1f} | "
              + " ".join(f"{c.replace('F_', '')[:6]}:{v:.2f}" for c, v in duty.items()), flush=True)
    print("\nDETECÇÃO POR EVENTO", flush=True)
    print(f"   {'C0 controle (v2)':22s}        " + " ".join(f"{k[8:10]}/{k[5:7]}:{v}" for k, v in L0["ev"].items()), flush=True)
    for (rot, n), (f, L, _) in sorted(RES.items(), key=lambda x: (x[0][1], x[0][0])):
        if not rot.startswith("PL"):
            print(f"   {rot:22s} n={n} " + " ".join(f"{k[8:10]}/{k[5:7]}:{v}" for k, v in L["ev"].items()), flush=True)
    NSs = sorted({n for _, n in RES}); med = lambda rot, ch: float(np.mean([RES[(rot, n)][1][ch] for n in NSs]))
    pls = sorted({rot for rot, _ in RES if rot.startswith("PL")})
    mfp = float(np.median([med(p, "fp") for p in pls])); mdet = float(np.median([med(p, "det") for p in pls]))
    print(f"\nCLASSIFICAÇÃO (médias sobre os n; placebos de partição: det mediana {mdet:.1f}, FP mediana {mfp:.1f})", flush=True)
    for rot in ("F famílias", "F' sem óleo"):
        d, fp = med(rot, "det"), med(rot, "fp"); melhor = fp < mfp and d >= mdet
        cl = "GANHO" if (((d >= 7 and fp <= 7) or (d >= 8 and fp <= 9)) and melhor) else "PIOR" if (d <= 5 or fp >= 12) else "EQUIVALENTE"
        print(f"   {rot:22s} det {d:.1f} | FP {fp:.1f} | horas {med(rot, 'horas'):.0f} | {'melhor' if melhor else 'NÃO melhor'} que os placebos -> {cl}{' | META DO USUÁRIO' if (fp <= 6 and d >= 6) else ''}", flush=True)
    if escreve:
        pd.DataFrame([dict(braco=rot, n=n, f=f, **{k: v for k, v in L.items() if k != "ev"}, ev=str(L["ev"]), duty=str(duty)) for (rot, n), (f, L, duty) in RES.items()]).to_csv(ROOT / "exp57_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
