#!/usr/bin/env python3
"""EXP 50 — UM MODELO POR GRUPAMENTO, NOS QUATRO CANAIS (t, p, sp, vb), DENTRO DA DECISÃO DE PRODUÇÃO COMPLETA.

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Os EXP 44 a 49 compararam modelos só nos canais t e p, isolados, e não dentro do voto. A pergunta do usuário é se cada GRUPAMENTO (temperaturas, pressões, mancal, vibração) deveria ter o seu tipo de modelo, e se isso vale também para sp e vb, que são os canais que mais pesam (desligados um a um: sem t 3/8, sem p 5/8, sem sp 5/8, sem vb 2/8, EXP 36). Aqui o canal trocado entra no `prever` de produção (voto A 3 de 4, voto B 2 de 4 com portão sp|vb, CUSUM, escalada, refratário, duração mínima) sem mudar mais nada.
O QUE SE FAZ (8 eventos; escores causais do harness fiel, retreino mensal nas últimas n amostras estáveis, a partir de 01/02/2025; sinais de controle do cache `sinais_v2_8ev.parquet`).
  Grupos e tags: t = as 14 tags do bundle; p = as 12 do bundle; sp = TI_0305, TI_0301, TI_0303 e TI_0307 (os 4 mancais; o canal atual é o z robusto do spread de TI_0305); vb = as 10 sondas TV_351 a 355 (X e Y).
  Famílias por grupo (FIXADAS AQUI): PCA(0,95) (o tipo atual de t e p); K-Means (k=8); análise fatorial (95%); autoencoder LSTM (20 épocas); Ridge (cada sensor previsto pelos outros, máximo do erro) — cada uma com n = 20.000 e 19.000; e ZR = z robusto contra a referência rolante de 400 h de cada tag (a MESMA função do vb de produção, `_z_vibracao`, repetida em replay causal semanal com os trips já ocorridos, aplicada às tags do grupo); para sp, mais SPROLL = o spread de TI_0305 com a mesma referência rolante. Escores de t e p com K-Means, análise fatorial e LSTM vêm do EXP 44; o PCA vem do escores_sub34.
  Cada sinal alternativo passa pelo EWMA de produção (t e p 1 h; sp e vb 30 min), substitui UM canal e o `prever` de produção roda. Duas variantes de escala (as escalas diferem entre famílias e os limiares base x k foram calibrados para o canal atual): "cru" (sem mudar o sinal) e "esc" (a principal): o sinal é multiplicado por um fator tal que a fração do tempo vigiado acima do limiar A do canal seja a MESMA do canal de produção desde 01/02/2025 (sem rótulos). Vizinhança: o limiar do canal trocado multiplicado por 0,8, 0,9, 1,0, 1,1 e 1,25.
  Controles: (i) o `prever` sobre o cache (o controle oficial; 7/8, 9 FP, 0,807 FP/mês); (ii) o PCA de n=20.000 e de n=19.000 trocados no mesmo caminho (o ruído de n e a paridade do caminho de troca); (iii) o placebo (eventos deslocados de +45 a +225 d, o mesmo alarme).
VERIFICAÇÕES (todas precisam passar para o resto valer). V1: o `prever` sobre o cache reproduz o controle (7/8, FP 9 +-1, série de alarme coincidente em >= 97% dos instantes vigiados). V2: trocar o PCA (n=20.000) em t e em p reproduz o controle com diferença máxima de 1 em detecção e 2 em FP. V3: o ZR aplicado às sondas de vibração reproduz o vb do cache com diferença absoluta máxima <= 1e-6 (valida o replay causal que será reaproveitado nos outros grupos).
MEDIDAS: detecção de 8, FP, FP/mês, horas totais e fora das janelas, placebo, por sinal, variante e fator.
CRITÉRIOS (fixados antes; a detecção do controle é 7/8 e +2 é impossível, então o critério é o FP a detecção quase igual). GANHO (sinal, variante "esc") = FP <= FP do controle - 3 (<= 6, 0,54 FP/mês) E detecção >= a do controle - 1 (>= 6/8) E acima do placebo do próprio sinal E, para t e p, FP <= o menor FP dos PCA de n=20.000 e 19.000 trocados menos 1 E na vizinhança (5 fatores) ao menos 3 com detecção >= 6/8 e FP <= o do controle. PIOR = detecção <= a do controle - 2 OU FP >= a do controle + 3. EQUIVALENTE no resto. META DO USUÁRIO (relatada à parte): FP/mês <= 0,55 com detecção >= 5/8 e na vizinhança ao menos 3 fatores com FP <= 6 e detecção >= 5/8. Cerca de 40 sinais x 2 variantes: qualquer GANHO é HIPÓTESE e exige replicação com o outro n (quando houver) e a vizinhança. NÃO se combina trocas de grupos diferentes sem um GANHO individual de cada um (sem GANHO não há combinação).
EXPECTATIVAS: V1 passa ~90%, V2 ~80%, V3 ~80%; algum sinal com GANHO ~18% (GANHO confirmado em n e vizinhança ~7%); alguma troca atinge a meta do usuário ~22%; a troca do vb por qualquer alternativa é PIOR em >= 80% dos sinais; sp com SPROLL EQUIVALENTE ou PIOR ~85%; t e p: >= 50% dos sinais EQUIVALENTES; "esc" mais estável que "cru" (menor dispersão do FP entre sinais) ~70%; o PCA de n=19.000 difere do de 20.000 em >= 2 de FP ou >= 1 de detecção em ao menos um canal ~60%.
INTERPRETAÇÃO PREVISTA. Se nenhuma troca ganha, o tipo de modelo por grupo não é alavanca dentro do voto (como já não era isolado), e o limite continua sendo informação sobre o desfecho; se algum ganhar e replicar, vira HIPÓTESE para a sombra, com placebo e vizinhança próprios.
LIMITES. 8 eventos (+-2; o controle em FP oscila +-3 com a vizinhança); k e votos NÃO são recalibrados por modelo (só a escala, pela fração de tempo aceso, que usa o período inteiro, sem rótulos mas não causal); o ZR em t, p e sp usa tags de cada grupo contra o próprio passado (a deriva lenta vira "normal" depois de 400 h); sp e vb alternativos são multivariados sobre as tags do grupo e podem não representar o que o canal atual mede (o spread entre o mancal e os irmãos); só 2 valores de n; LSTM com uma semente.

Uso:  PYTHONPATH=src .venv/bin/python experiments/50_modelo_por_grupo.py
      variáveis: EXP50_PROCS (padrão 6), EXP50_TESTE=1 (encanamento: grupo sp, PCA e Ridge, n=20.000, 7 semanas de replay)
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

TESTE = bool(os.environ.get("EXP50_TESTE"))
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py"); E20 = E46.E20
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG, JANELA
from nv.modelos.familias import RidgeOutras
from nv.modelos.lstm_ae import walk_lstm
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; CACHE = ROOT / ("escores_grupo50_teste" if TESTE else "escores_grupo50"); SHIFTS = E46.SHIFTS; NS = (20000, 19000); FATORES = (0.8, 0.9, 1.0, 1.1, 1.25); GL = {}
SP_TAGS = ["954005_624_TI_0305", "954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]
FAMS = ("PCA", "KM", "FA", "LSTM", "RIDGE"); INI_REPLAY = pd.Timestamp("2025-01-08 07:34", tz="UTC")


def fabrica(fam):
    cls = {"PCA": lambda: E46.fabrica({"fam": "pca", "var": 0.95}), "KM": lambda: E46.fabrica({"fam": "km", "k": 8}), "FA": lambda: E46.fabrica({"fam": "fa", "var": 0.95}), "RIDGE": lambda: E20.com_sk(RidgeOutras)}[fam]()
    return lambda semente=0: cls(semente=semente)


def arquivo(grupo, fam, n):
    return CACHE / f"{grupo}_{fam}_{n}.parquet"


def replay_zr(tags, hl, com_spread):
    """ZR causal: a função de produção do vb sobre as tags do grupo, em replay semanal com os trips já ocorridos (o mesmo corte e a mesma janela do `sinais_causais`). Devolve o sinal JÁ com o EWMA de produção."""
    ci = _ci(); csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv)
    x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet"); raw = pd.concat([x.loc[x.index < raw.index[0], raw.columns], raw]).sort_index()
    df = ci.preparar_grade(raw)
    if com_spread:
        df["SPREAD"] = df[SP_TAGS[0]].astype("float64") - df[SP_TAGS[1:]].astype("float64").median(axis=1)
    trips = sorted(ci.carregar_trips(PKG / "registro_trips.csv")); real = df["RUNNING_A"].notna() & df[[c for c in df.columns if c.startswith("TV_")]].notna().any(axis=1); df = df.loc[:df.index[real][-1]]; fim = df.index[-1]
    if TESTE:
        fim = INI_REPLAY + pd.Timedelta(days=49)
    cortes = [INI_REPLAY]
    while cortes[-1] + pd.Timedelta(days=7) < fim:
        cortes.append(cortes[-1] + pd.Timedelta(days=7))
    cortes.append(fim); pedacos, anterior = [], None
    for c in cortes:
        sub = [f for f in trips if anterior is not None and anterior < f < c] + [c]
        for s in sub:
            conhecidos = trips if s == c else [t for t in trips if t < s]; w = df.loc[max(df.index[0], s - JANELA):s]
            with contextlib.redirect_stdout(io.StringIO()):
                g = ci._grade(w); mask, estavel, partida, op = ci._mascara(g); z = pd.Series(ci._z_vibracao(g, estavel, tags, [t for t in conhecidos if pd.Timestamp(t) <= g.index[-1]], 7.0), index=g.index)
            e = z.ewm(halflife=pd.Timedelta(hl), times=g.index).mean().where(z.notna()); novo = e[e.index > anterior] if anterior is not None else e; pedacos.append(novo); anterior = s
    res = pd.concat(pedacos); return res[~res.index.duplicated(keep="first")]


def pontua(a):
    grupo, fam, n = a; f = arquivo(grupo, fam, n)
    if f.exists():
        return a
    t0 = time.time(); GLB = GL; tags = GLB["tags"][grupo]; hl = GLB["hl"][grupo]
    if fam in ("ZR", "SPROLL"):
        s = replay_zr(["SPREAD"] if fam == "SPROLL" else tags, hl, fam == "SPROLL")
    else:
        import torch; torch.set_num_threads(1)
        if fam == "LSTM":
            s = walk_lstm(GLB["g"], tags, GLB["est"], INICIO8, GLB["fim"], semente=0, n_max=n, ep0=20)
        else:
            s = walk_forward(fabrica(fam), GLB["g"], tags, GLB["est"], INICIO8, GLB["fim"], n_treino=n, semente=0)
        s = s.loc[GLB["ini"]:]
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {grupo} {fam} n={n}: {time.time() - t0:.0f} s", flush=True); return a


def ewma_prod(s, hl):
    return s.ewm(halflife=pd.Timedelta(hl), times=s.index).mean().where(s.notna())


def carrega_sinal(grupo, fam, n):
    """O sinal alternativo JÁ no EWMA de produção."""
    hl = GL["hl"][grupo]
    if fam in ("ZR", "SPROLL"):
        return pd.read_parquet(arquivo(grupo, fam, 0))["s"]
    if grupo in ("t", "p") and fam in ("PCA", "KM", "FA", "LSTM"):
        f = ROOT / "escores_sub34" / f"{grupo}_A_{n}.parquet" if fam == "PCA" else ROOT / "escores_lit44_w" / f"{grupo}_{fam}_{n}.parquet"
        return ewma_prod(pd.read_parquet(f)["s"].loc[GL["ini"]:], hl)
    return ewma_prod(pd.read_parquet(arquivo(grupo, fam, n))["s"], hl)


def fator_escala(grupo, x):
    c = grupo; thr = GL["det"]["base"][c] * GL["det"]["k_nivel_a"][c]; p0 = GL["proc0"]; m = p0["mask"].astype(bool) & (p0.index >= INICIO8)
    idx = x.dropna().index.intersection(p0.index[m]); f_prod = float((p0.loc[idx, c] > thr).mean()); q = float(x.loc[idx].quantile(1 - f_prod)) if 0 < f_prod < 1 else np.nan
    return thr / q if q and np.isfinite(q) and q > 0 else 1.0


def roda(proc, ev):
    res = decide_v2(proc, GL["det"]); al = res["is_anomaly"].astype(bool); al_c = GL["al_c"]; al = al.reindex(al_c.index).fillna(False); out = {}
    for nome, evs in ev.items():
        r = avalia(al, GL["vig_c"], evs, inicio_periodo=INICIO8); out[nome] = r
    return out


def avalia_sinal(a):
    grupo, fam, n = a; x = carrega_sinal(grupo, fam, n); lam = fator_escala(grupo, x); linhas = []
    for variante in ("cru", "esc"):
        base = x * (lam if variante == "esc" else 1.0)
        for f in FATORES:
            proc = substitui(GL["proc0"], grupo, base / f); r = roda(proc, GL["EV"]); real = r["real"]
            pl = [r[k].det / r[k].n_eventos for k in r if k != "real" and r[k].n_eventos]
            linhas.append(dict(grupo=grupo, fam=fam, n=n, variante=variante, fator=f, det=real.det, fp=real.fp, fp_mes=real.fp_mes, horas=real.horas_alarme, fora=real.horas_alarme_fora, placebo=float(np.median(pl)) * 8 if pl else np.nan, lam=lam))
    return linhas


def classifica(L, ctrl, ruido):
    """L: linhas (variante 'esc') de um sinal; ctrl: (det, fp) do controle; ruido: FP mínimo dos PCA trocados (t e p) ou None."""
    u = [x for x in L if x["fator"] == 1.0][0]; viz = [x for x in L if x["det"] >= 6 and x["fp"] <= ctrl[1]]; viz_meta = [x for x in L if x["det"] >= 5 and x["fp"] <= 6]
    ganho = (u["fp"] <= ctrl[1] - 3 and u["det"] >= ctrl[0] - 1 and u["det"] > (u["placebo"] if np.isfinite(u["placebo"]) else 0.0) and (ruido is None or u["fp"] <= ruido - 1) and len(viz) >= 3)
    pior = u["det"] <= ctrl[0] - 2 or u["fp"] >= ctrl[1] + 3
    meta = u["fp_mes"] <= 0.55 and u["det"] >= 5 and len(viz_meta) >= 3
    return ("GANHO" if ganho else "PIOR" if pior else "EQUIVALENTE"), meta, u


def main():
    procs = int(os.environ.get("EXP50_PROCS", 6)); g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items() if c in ("t", "p")}; tags["sp"] = SP_TAGS; tags["vb"] = list(det["tags_vibracao"])
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    proc0 = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc0["mask"] = proc0["mask"].astype(bool)
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(g=g, est=est, fim=fim, ini=INICIO8 - pd.Timedelta(days=10), tags=tags, hl=det["halflife"], det=det, proc0=proc0, al_c=al_c, vig_c=vig_c, EV=EV)
    grupos = ("sp", "vb") if TESTE else ("t", "p", "sp", "vb"); nss = (20000,) if TESTE else NS; fams = ("PCA", "RIDGE") if TESTE else FAMS
    print(f"EXP 50: grupos {grupos} | famílias {fams} | n {nss} | PROCS {procs} | teste={TESTE} | eventos {({k: len(v) for k, v in EV.items()})}", flush=True)
    # V1: controle oficial
    print("\nV1 — O `prever` SOBRE O CACHE reproduz o controle", flush=True); r0 = roda(proc0, EV); c0 = r0["real"]
    res0 = decide_v2(proc0, det)["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False); v = vig_c.loc[INICIO8:]; conc = float((res0.loc[INICIO8:][v] == al_c.loc[INICIO8:][v]).mean())
    print(f"   controle: {c0.det}/{c0.n_eventos} | FP {c0.fp} | FP/mês {c0.fp_mes:.3f} | {c0.horas_alarme:.0f} h | {c0.horas_alarme_fora:.0f} h fora | coincidência com a série do ledger {100 * conc:.2f}%", flush=True)
    v1 = c0.det == 7 and abs(c0.fp - 9) <= 1 and conc >= 0.97; print(f"   V1 {'PASSA' if v1 else 'FALHA'}", flush=True)
    ctrl = (c0.det, c0.fp)
    # escores novos
    tarefas = []
    for gr in grupos:
        for fam in fams:
            for n in nss:
                if gr in ("t", "p") and fam in ("PCA", "KM", "FA", "LSTM"):
                    continue
                tarefas.append((gr, fam, n))
        tarefas.append((gr, "ZR", 0))
        if gr == "sp":
            tarefas.append((gr, "SPROLL", 0))
    t0 = time.time()
    with Pool(procs) as p:
        list(p.imap_unordered(pontua, tarefas, chunksize=1))
    print(f"ESCORES NOVOS ({len(tarefas)}) em {time.time() - t0:.0f} s", flush=True)
    # V3: ZR sobre as sondas == vb do cache
    if "vb" in grupos:
        zr = pd.read_parquet(arquivo("vb", "ZR", 0))["s"]; j = zr.index.intersection(proc0.index); j = j[j >= INICIO8]; d = (zr.loc[j] - proc0.loc[j, "vb"]).abs(); print(f"\nV3 — ZR sobre as sondas == vb do cache: diferença absoluta máxima {np.nanmax(d):.2e}, NaN diferentes {int((zr.loc[j].isna() != proc0.loc[j, 'vb'].isna()).sum())}", flush=True); v3 = np.nanmax(d) <= 1e-6; print(f"   V3 {'PASSA' if v3 else 'FALHA'}", flush=True)
    sinais = [(gr, fam, n) for gr in grupos for fam in fams for n in nss] + [(gr, "ZR", 0) for gr in grupos if gr != "vb"] + ([("sp", "SPROLL", 0)] if "sp" in grupos else [])
    t0 = time.time()
    with Pool(procs) as p:
        out = [l for ls in p.imap_unordered(avalia_sinal, sinais, chunksize=1) for l in ls]
    R = pd.DataFrame(out); R.to_csv(ROOT / ("exp50_resultados_teste.csv" if TESTE else "exp50_resultados.csv"), index=False); print(f"AVALIAÇÃO ({len(sinais)} sinais x 2 variantes x {len(FATORES)} fatores) em {time.time() - t0:.0f} s", flush=True)
    nomes = {"PCA": "PCA", "KM": "K-Means", "FA": "análise fatorial", "LSTM": "LSTM", "RIDGE": "Ridge", "ZR": "ZR (z rolante)", "SPROLL": "spread rolante"}
    # V2 e ruído de n
    ruido = {}
    for gr in ("t", "p"):
        if gr not in grupos:
            continue
        pc = [R[(R.grupo == gr) & (R.fam == "PCA") & (R.n == n) & (R.variante == "cru") & (R.fator == 1.0)].iloc[0] for n in nss]; ruido[gr] = min(int(x.fp) for x in pc)
        print(f"\nV2 [{gr}] PCA trocado (variante 'cru', fator 1,0): " + " | ".join(f"n={n}: {int(x.det)}/8, FP {int(x.fp)}" for n, x in zip(nss, pc)) + f" | controle {c0.det}/8, FP {c0.fp}", flush=True)
    print("\nRESULTADOS por grupo (variante 'esc', fator 1,0; detecção de 8 | FP | FP/mês | horas | placebo); 'cru' entre parênteses; vizinhança = fatores com detecção >= 6 e FP <= controle", flush=True)
    clas = {}
    for gr in grupos:
        print(f"\n[{gr}]", flush=True)
        for fam, n in [(f, n) for f in fams for n in nss] + [("ZR", 0)] * (gr != "vb") + [("SPROLL", 0)] * (gr == "sp"):
            if gr in ("t", "p") and fam not in ("PCA", "KM", "FA", "LSTM", "RIDGE", "ZR"):
                continue
            Ls = R[(R.grupo == gr) & (R.fam == fam) & (R.n == n)]
            if len(Ls) == 0:
                continue
            esc = [r.to_dict() for _, r in Ls[Ls.variante == "esc"].iterrows()]; cru = Ls[(Ls.variante == "cru") & (Ls.fator == 1.0)].iloc[0]
            cl, meta, u = classifica(esc, ctrl, ruido.get(gr)); clas[(gr, fam, n)] = (cl, meta, u)
            viz = sum(1 for x in esc if x["det"] >= 6 and x["fp"] <= ctrl[1])
            print(f"   {nomes[fam]:17s} n={n:5d} | esc: {u['det']}/8, FP {u['fp']}, {u['fp_mes']:.2f}/mês, {u['horas']:.0f} h, placebo {u['placebo']:.1f}, fator {u['lam']:.3g}, vizinhança {viz}/5 | (cru: {int(cru.det)}/8, FP {int(cru.fp)}) -> {cl}{' | META DO USUÁRIO' if meta else ''}", flush=True)
    print("\nRESUMO", flush=True)
    for gr in grupos:
        cs = [v[0] for k, v in clas.items() if k[0] == gr]; print(f"   [{gr}] GANHO {cs.count('GANHO')} | EQUIVALENTE {cs.count('EQUIVALENTE')} | PIOR {cs.count('PIOR')} | meta do usuário em {sum(1 for k, v in clas.items() if k[0] == gr and v[1])}", flush=True)


if __name__ == "__main__":
    main()
