#!/usr/bin/env python3
"""EXP 59 — CUSUM SOBRE O RESÍDUO BRANQUEADO: o caminho de CUSUM de cada canal passa a acumular MUDANÇA (inovações de um AR(1)) em vez de NÍVEL.

PRÉ-REGISTRO (20/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp59.py`), branch exp/modos-de-falha.

POR QUÊ (literatura, 20/10). O CUSUM supõe observações independentes; com dados autocorrelacionados ele gera "um grande número de falsos alarmes" (cartas de resíduo: Alwan e
Roberts 1988; CUSUM de resíduo; projeto por bootstrap de Capizzi e Masarotto). Uma patente de vigilância de usinas mediu taxa de falso alarme de 0,073 contra 0,01 projetado em
sinais não filtrados. No v2 os canais passam por EWMA (que AUMENTA a autocorrelação) e depois por CUSUM; já medimos que "o CUSUM acumula nos estados persistentes" (EXP 35) e que o
ciclo de 34-59% de tempo aceso dos canais "vem do CUSUM" (`canais-sao-estados-teto-de-informacao`). O remédio clássico — CUSUM nas inovações de um modelo autorregressivo —
NUNCA foi testado (o DPCA modelou a dinâmica dentro do PCA; as variantes de memória mexeram em teto e reinício, não na entrada).
O QUE MUDA, E O QUE NÃO MUDA. A regra de DEGRAU sobre o nível (30 min acima do limiar) fica intacta: pega a subida franca. Só o caminho de CUSUM troca: em vez de acumular
(E/limiar - kappa), acumula (z - k), com z = inovação padronizada do AR(1) do canal. Um estado alto e PLANO deixa de acumular; uma SUBIDA sustentada continua acumulando.
DESENHO (fixado antes):
  REVISÃO ANTES DO PRÉ-REGISTRO (teste de encanamento, só verificações, nenhum resultado): o AR(1) mensal estimado sai com phi de 0,977 a 1,03 em todos os canais — o sinal
  suavizado é quase INTEGRADO na escala de 2 min, e o resíduo do AR(1) colapsa na primeira diferença; e um detector de MUDANÇA não consegue ficar aceso os 34-59% do tempo do
  CUSUM de nível, então igualar o duty por canal é impossível por construção (diferença de 17,75 pontos). Desenho final:
  BRANQUEAMENTO: z_t = (E_t - E_{t-1}) / sigma_m, com sigma_m = 1,4826 x MAD das diferenças vigiadas consecutivas em [m - 30 d, m) (causal, por canal e mês) — o branqueamento de
  uma série integrada, ARIMA(0,1,0); z = 0 quando um dos dois instantes não é vigiado (a abertura da máscara não injeta salto); z cortado em +-5 (CUSUM robusto, cauda pesada).
  CUSUM: S = `ci._cusum`(z - k, reset, carga_reset) — mesma mecânica, reset e carga do pacote; k = 0,5 (o clássico para deslocamento de 1 sigma).
  LIMIAR SEM RÓTULOS, A CUSTO IGUAL: o canal acende no nível A quando S > H e no B quando S > H x (limiar B / limiar A do canal); UM H por braço e k, escolhido numa grade densa
  (2 a 400, 120 valores geométricos) para igualar as HORAS TOTAIS de alarme do controle (1.457 h; +-5%, relaxado a +-15% com a diferença relatada), como no EXP 57.
  BRAÇOS: W4 = os 4 canais; W2 = só sp e vb (os canais-estado do EXP 42). Vizinhança: k = 0,25, 0,5 e 1,0, cada um com o seu H.
  PLACEBO DE TEMPO: as séries "S > H" de cada canal deslocadas circularmente por sorteios independentes de 30 a 200 dias (5 sorteios, semente 20261025), mascaradas de novo, no
  H de k = 0,5; o degrau fica no lugar. Se o placebo der o mesmo, o momento em que o CUSUM de mudança acende não carrega informação.
  DECISÃO: voto e pós-processamento do pacote (reimplementação do EXP 51, V1 = `ci.prever`).
MEDIDAS: detecção de 8, início, banda, FP, FP/mês, horas totais e fora, placebo de eventos (+45 a +225 d), duty de cada canal, detecção por evento.
CRITÉRIOS (fixados antes):
  GANHO = [(det >= 7 e FP <= 7) ou (det >= 8 e FP <= 9)] E horas <= 1,10 x 1.457 E melhor que a MEDIANA do seu placebo de tempo (FP menor e detecção não menor) E vale em >= 2 dos 3 k.
  PIOR = det <= 5 OU FP >= 12. EQUIVALENTE no resto. META DO USUÁRIO (à parte) = FP <= 6 e det >= 6.
EXPECTATIVAS: V1 passa (~95%); W4 GANHO (~12%); W2 GANHO (~12%); W4 corta >= 2 FP (~50%) mas perde >= 1 detecção (~60%), porque parte das detecções vem de estado persistente
(o 07/04 crônico); as horas caem >= 20% em W4 (~55%); W melhor que o seu placebo de tempo (~50%); algum braço na META (~10%).
INTERPRETAÇÃO PREVISTA: GANHO acima do placebo -> o CUSUM de nível estava contando estado como evidência; candidato à sombra (muda o detector congelado; decisão do usuário).
Corta FP mas perde detecção -> a troca estado x mudança é a mesma fronteira de sempre, agora medida por um mecanismo novo. EQUIVALENTE -> a autocorrelação não é o gargalo.
LIMITES: 8 eventos; o h é calibrado no período inteiro (sem rótulo, mas usa o futuro); AR(1) é a forma mais simples (o score tem memória longa); nada se adota sem a sombra.
VERIFICAÇÕES: V1 a decisão com as acesas originais recalculadas aqui reproduz `ci.prever` instante a instante; V2 as horas igualam as do controle a +-5% (relaxado +-15%) em W4 e
W2 nos 3 k; V3 sigma mensal > 0 em todos os canais e meses.
CORREÇÃO 2 (20/10, tarefa a319ec65 parou na V2 antes de qualquer resultado: 5 de 6 braços igualados): grade de H ampliada para 0,25-2.000 (200 valores); a V2 passa com >= 5 de 6
braços a +-15%, e o braço que não igualar fica marcado "CUSTO NÃO IGUALADO", com as horas na tabela, e nunca conta como GANHO. Critérios inalterados.

Uso local (encanamento, sem tabelas):  EXP59_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/59_cusum_branqueado.py
No worker:  PYTHONPATH=src python scripts/worker_exp59.py --enqueue --queue default
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
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG

TESTE = bool(os.environ.get("EXP59_TESTE"))
ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; GL = E51.GL; SIN = ("t", "p", "sp", "vb")
KS = (0.25, 0.5, 1.0); K0 = 0.5; N_PLAC = 2 if TESTE else 5; SEMENTE = 20261025; SUST = 15


def acesas_originais(c, nivel):
    """O que `ci._acende` faz, decomposto: (degrau, cusum) para o canal e o nível."""
    ci = GL["ci"]; det = GL["det"]; px = GL["proc"]; mask = px["mask"].astype(bool); reset = px["reset"].to_numpy()
    thr = det["base"][c] * det[f"k_nivel_{nivel}"][c]; Em = px[c].where(mask)
    deg = ((Em > thr).astype(int).rolling(SUST, min_periods=SUST).sum() >= SUST)
    acum = ((Em / thr).clip(upper=20) - det["kappa"]).fillna(0.0).to_numpy()
    cu = pd.Series(ci._cusum(acum, reset, det["carga_reset"]) > det["h_cusum"], index=px.index)
    return deg & mask, cu & mask


def inovacoes(c):
    """z_t = diferença padronizada (escala causal mensal), cortada em +-5; devolve (z, lista de sigma)."""
    px = GL["proc"]; mask = px["mask"].astype(bool); Ev = px[c].where(mask); dif = (Ev - Ev.shift(1)).to_numpy(); par = np.isfinite(dif)
    z = np.zeros(len(Ev)); idx = px.index; sigs = []; ultimo = None
    for m0 in pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = (idx >= m0 - pd.Timedelta(days=30)) & (idx < m0) & par
        if tr.sum() >= 2000:
            dd = dif[tr]; sig = 1.4826 * float(np.median(np.abs(dd - np.median(dd))))
            if not sig > 0:
                sig = float(np.std(dd))
            ultimo = sig; sigs.append(sig)
        if ultimo is None:
            continue
        ap = (idx >= m0) & (idx < m1) & par; z[ap] = dif[ap] / ultimo
    return np.clip(z, -5.0, 5.0), sigs


def decide(A, B):
    px = GL["proc"]; det = GL["det"]; mask = px["mask"].astype(bool)
    voto = ((sum(A[c].astype(int) for c in SIN) >= det["voto_nivel_a"]) & mask) | ((sum(B[c].astype(int) for c in SIN) >= det["voto_nivel_b"]) & mask & (B["sp"] | B["vb"]))
    forca = pd.concat([px[c].where(mask) / (det["base"][c] * det["k_nivel_b"][c]) for c in SIN], axis=1).max(axis=1)
    return E51.pos_processa(voto, forca, det)


def mede(al):
    a = al.reindex(GL["al_c"].index).fillna(False)
    r = avalia(a, GL["vig_c"], GL["EV"]["real"], inicio_periodo=INICIO8)
    pl = [avalia(a, GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev) for k, ev in GL["EV"].items() if k != "real" and ev]
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora,
                placebo=float(np.median(pl)) * 8 if pl else np.nan, ev={k[:10]: int(v["det"]) for k, v in r.por_evento.items()})


def flags_w(canais, k, H, desloca=None):
    """Acesas A/B com o caminho de CUSUM trocado pelo CUSUM das diferenças nos `canais`; `desloca` = {canal: amostras} para o placebo de tempo."""
    ci = GL["ci"]; det = GL["det"]; px = GL["proc"]; mask = px["mask"].astype(bool); reset = px["reset"].to_numpy(); vig = mask.to_numpy()
    A, B, duty = {}, {}, {}
    for c in SIN:
        for nivel, D in (("a", A), ("b", B)):
            deg, cu = GL["orig"][(c, nivel)]
            if c not in canais:
                D[c] = deg | cu; continue
            S = GL["S"].setdefault((c, k), ci._cusum(GL["z"][c] - k, reset, det["carga_reset"]))
            razao = 1.0 if nivel == "a" else det["k_nivel_b"][c] / det["k_nivel_a"][c]
            cw = S > H * razao
            if desloca is not None:
                cw = np.roll(cw, desloca[c])
            cw = pd.Series(cw, index=px.index) & mask
            D[c] = deg | cw; duty[(c, nivel)] = (float(cu.to_numpy()[vig].mean()), float(cw.to_numpy()[vig].mean()))
    return A, B, duty


GRADE_H = np.unique(np.round(np.geomspace(0.25, 2000, 200), 3))   # CORREÇÃO 2: a grade 2-400 não alcançava as horas de um braço


def tarefa(a):
    """Sem `H`: busca o H que iguala as horas do controle; com `H`: avalia direto (placebo)."""
    rot, canais, k, desloca, H = a
    if H is not None:
        A, B, duty = flags_w(canais, k, H, desloca); return rot, k, H, mede(decide(A, B)), duty
    alvo = GL["horas_alvo"]; melhor = None
    for h in (GRADE_H if not TESTE else GRADE_H[::30]):
        A, B, duty = flags_w(canais, k, float(h)); L = mede(decide(A, B))
        if melhor is None or abs(L["horas"] - alvo) < abs(melhor[1]["horas"] - alvo):
            melhor = (float(h), L, duty)
    return rot, k, melhor[0], melhor[1], melhor[2]


def main():
    procs = int(os.environ.get("EXP59_PROCS", 6)); ci = _ci(); GL["ci"] = ci; t0 = time.time(); ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); proc["reset"] = proc["reset"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV)
    GL["orig"] = {(c, nv): acesas_originais(c, nv) for c in SIN for nv in ("a", "b")}
    A0 = {c: GL["orig"][(c, "a")][0] | GL["orig"][(c, "a")][1] for c in SIN}; B0 = {c: GL["orig"][(c, "b")][0] | GL["orig"][(c, "b")][1] for c in SIN}
    with contextlib.redirect_stdout(io.StringIO()):
        ref = ci.prever({"detector": det}, proc)["is_anomaly"].astype(bool)
    al0 = decide(A0, B0); d1 = int((al0 != ref).sum()); ok["V1"] = d1 == 0; L0 = mede(al0)
    print(f"V1 — decisão com as acesas originais x ci.prever: {d1} instantes diferentes -> {'PASSA' if ok['V1'] else 'FALHA'} | controle {L0['det']}/8, FP {L0['fp']}, {L0['horas']:.0f} h", flush=True)
    GL["z"] = {}; GL["S"] = {}; GL["horas_alvo"] = L0["horas"]; todos = []
    for c in SIN:
        z, sigs = inovacoes(c); GL["z"][c] = z; todos += sigs
        print(f"   {c}: sigma mensal de {min(sigs):.2e} a {max(sigs):.2e} ({len(sigs)} meses); |z| no corte (5) em {100 * float((np.abs(z) >= 5).mean()):.2f}% dos instantes", flush=True)
    ok["V3"] = all(x > 0 for x in todos)
    print(f"V3 — sigma mensal > 0 em todos os canais e meses -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    rng = np.random.default_rng(SEMENTE); dia = 720
    plac = [{c: int(rng.integers(30 * dia, 200 * dia)) for c in SIN} for _ in range(N_PLAC)]
    BRACOS = (("W4 (4 canais)", SIN), ("W2 (sp e vb)", ("sp", "vb")))
    with Pool(min(procs, 6)) as pool:
        RES = {(rot, k): (H, L, duty) for rot, k, H, L, duty in pool.imap_unordered(tarefa, [(rot, cn, k, None, None) for rot, cn in BRACOS for k in KS], chunksize=1)}
    t2 = [(f"{rot} placebo {j + 1}", cn, K0, dsl, RES[(rot, K0)][0]) for rot, cn in BRACOS for j, dsl in enumerate(plac)]
    with Pool(min(procs, len(t2))) as pool:
        RES |= {(rot, k): (H, L, duty) for rot, k, H, L, duty in pool.imap_unordered(tarefa, t2, chunksize=1)}
    alvo = GL["horas_alvo"]; prin = [(r, k) for r, _ in BRACOS for k in KS]
    ok5 = [abs(RES[x][1]["horas"] - alvo) <= 0.05 * alvo for x in prin]; ok15 = [abs(RES[x][1]["horas"] - alvo) <= 0.15 * alvo for x in prin]
    ok["V2"] = sum(ok15) >= len(prin) - 1 or TESTE      # CORREÇÃO 2: um braço não igualável é marcado, não para o experimento
    GL["nao_igualado"] = {x for x, o in zip(prin, ok15) if not o}
    print(f"V2 — horas igualadas a {alvo:.0f} h nos braços principais: +-5% em {sum(ok5)}/{len(prin)}, +-15% em {sum(ok15)}/{len(prin)} -> {'PASSA' if ok['V2'] else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    for x in prin:
        print(f"      {x[0]:16s} k={x[1]:<4} H={RES[x][0]:<8} horas {RES[x][1]['horas']:.0f}{'  <- CUSTO NÃO IGUALADO' if x in GL['nao_igualado'] else ''}", flush=True)
    RES = {key: (val[1], val[2], val[0]) for key, val in RES.items()}   # (medidas, duty, H)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(RES, L0, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). V1 {ok['V1']}, V3 {ok['V3']}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(RES, L0, ok, t0, escreve=True)


def relata(RES, L0, ok, t0, escreve):
    print("\nRESULTADOS (det/8 | início | banda | FP | FP/mês | horas | fora | placebo de eventos)", flush=True)
    print(f"   {'C0 controle (v2)':28s}        | {L0['det']}/8 | {L0['inicio']} | {L0['banda']} | FP {L0['fp']:2d} | {L0['fp_mes']:.3f} | {L0['horas']:5.0f} h | {L0['fora']:4.0f} h | {L0['placebo']:.1f}", flush=True)
    for (rot, k), (L, _, H) in sorted(RES.items()):
        print(f"   {rot:28s} k={k:<4} H={H:<6} | {L['det']}/8 | {L['inicio']} | {L['banda']} | FP {L['fp']:2d} | {L['fp_mes']:.3f} | {L['horas']:5.0f} h | {L['fora']:4.0f} h | {L['placebo']:.1f}", flush=True)
    print("\nDETECÇÃO POR EVENTO (k = 0,5)", flush=True)
    print(f"   {'C0 controle (v2)':28s} " + " ".join(f"{e[8:10]}/{e[5:7]}:{v}" for e, v in L0["ev"].items()), flush=True)
    for rot in ("W4 (4 canais)", "W2 (sp e vb)"):
        print(f"   {rot:28s} " + " ".join(f"{e[8:10]}/{e[5:7]}:{v}" for e, v in RES[(rot, K0)][0]["ev"].items()), flush=True)
    print("\nCLASSIFICAÇÃO", flush=True)
    for rot in ("W4 (4 canais)", "W2 (sp e vb)"):
        pl = [L for (r, k), (L, _, _) in RES.items() if r.startswith(rot + " placebo")]
        mfp = float(np.median([L["fp"] for L in pl])); mdet = float(np.median([L["det"] for L in pl]))
        def ganha(L):
            return ((L["det"] >= 7 and L["fp"] <= 7) or (L["det"] >= 8 and L["fp"] <= 9)) and L["horas"] <= 1.10 * 1457
        nk = sum(ganha(RES[(rot, k)][0]) and (rot, k) not in GL.get("nao_igualado", set()) for k in KS); L = RES[(rot, K0)][0]
        melhor = L["fp"] < mfp and L["det"] >= mdet
        cl = "GANHO" if (ganha(L) and melhor and nk >= 2 and (rot, K0) not in GL.get("nao_igualado", set())) else "PIOR" if (L["det"] <= 5 or L["fp"] >= 12) else "EQUIVALENTE"
        cl += " (k=0,5 com CUSTO NÃO IGUALADO)" if (rot, K0) in GL.get("nao_igualado", set()) else ""
        print(f"   {rot:16s} k=0,5: det {L['det']} | FP {L['fp']} | horas {L['horas']:.0f} | placebo de tempo: det mediana {mdet:.1f}, FP mediana {mfp:.1f} ({'melhor' if melhor else 'NÃO melhor'})"
              f" | critério de GANHO em {nk}/3 k -> {cl}{' | META DO USUÁRIO' if (L['fp'] <= 6 and L['det'] >= 6) else ''}", flush=True)
    if escreve:
        pd.DataFrame([dict(braco=r, k=k, H=H, **{a: b for a, b in L.items() if a != "ev"}, ev=str(L["ev"])) for (r, k), (L, _, H) in RES.items()]).to_csv(ROOT / "exp59_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
