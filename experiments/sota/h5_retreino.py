#!/usr/bin/env python3
"""SOTA H5 — RETREINO SOB DEMANDA: quantos retreinos um gatilho dispararia, e o que isso custa, contra o retreino mensal de hoje. Tag ClearML `sota-h5`.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. Pergunta do brief: com um gatilho objetivo, quantos retreinos seriam disparados em comparação ao calendário
mensal, e com que custo em detecção, FP e horas? Ajuste do usuário: além de ADWIN/Page-Hinkley sobre o escore cru, gatilho por proxy de desempenho no regime normal
(% de horas acima do limiar e percentil do escore em janela móvel), e 4 políticas: mensal, congelado, gatilho, automático com trava.
O QUE JÁ SE SABE (declarado): o `constroi_bundle.py` registra, num ponto de operação antigo, mensal 8/8 7,15 h/mês, trimestral 7/8 4,48, semestral 6/8 16,94, congelado
6/8 108,19; e que mudar o DIA do corte do baseline (1 -> 8 -> 15) muda muito o detector (drift_composicao.py). Por isso TODA política aqui só troca de bundle numa virada
de mês e só usa os bundles de dia 1 que existem: o gatilho decide SE troca, não QUANDO dentro do mês (exceto a trava, que troca no dia 8 depois de uma semana de sombra).
Só os canais t, p e sp dependem do bundle; o vb (referência rolante) é o mesmo em todas as políticas.
MONTAGEM: o escore cru de t, p e sp de cada mês é calculado com cada bundle candidato (aritmética do pacote, como na H6), montado conforme a política, suavizado pelo EWMA
do canal e decidido pelo `prever` com vb, máscara e resets do cache causal. Todas as políticas partem do bundle de fev/2025; 14 viradas (01/03/2025 a 01/04/2026).
POLÍTICAS (fixadas antes; os gatilhos só usam dado ANTERIOR à virada):
  P0 MENSAL — troca em toda virada (o de hoje).   P1 CONGELADO — nunca troca.
  P2a HORAS ACIMA DO LIMIAR — troca se, em algum canal (t, p, sp), a fração do tempo vigiado acima do limiar A nos últimos 14 d >= max(k x a dos primeiros 14 d de
      serviço do bundle vigente, 0,05); k = 2 (vizinhos 1,5 e 3). Com < 48 h vigiadas numa das janelas, não dispara.
  P2b PERCENTIL MÓVEL — troca se o p99 do escore CRU de t ou p (normalizado pelo recon_p99, ~1 no baseline) nos últimos 7 d vigiados > q; q = 2 (vizinhos 1,5 e 3).
      Com < 24 h vigiadas, não dispara.
  P2c ADWIN (river, parâmetros padrão) sobre a mediana horária do escore cru de t e de p em vigilância, desde a instalação do bundle; troca se detectou mudança no mês.
  P2d PAGE-HINKLEY (river, parâmetros padrão) sobre o p95 horário do escore cru de t e de p; idem.
  P3 AUTOMÁTICO COM TRAVA — em toda virada o bundle novo é candidato; durante 7 d ele roda em sombra e o vigente segue; no dia 8 entra se (i) o baseline dele é LIMPO
      (nenhuma janela [T − 48 h, T + 2 d] de trip já conhecido o intersecta e <= 10% dos seus pontos estavam em alarme no detector da política) E (ii) na semana de sombra
      ele não dá mais horas de alarme nem mais episódios que o vigente. Nunca deixa o vigente passar de 62 d de idade (troca forçada, contada).
MEDIDAS: número de retreinos (trocas), trocas forçadas, meses com bundle vencido (> 62 d: a inferência de produção recusaria rodar), todas as métricas do protocolo com IC,
e a comparação pareada de cada política contra a mensal; a decisão de cada política em cada virada.
CRITÉRIOS (fixados antes):
  SUSTENTA RETREINO SOB DEMANDA (P1, P2*) = <= 7 retreinos em 14 viradas E nenhuma piora declarada contra a mensal pela regra do protocolo E detecção >= mensal − 1 E horas
    <= 1,25 x mensal; para P2a e P2b, os dois vizinhos também sustentam.
  TRAVA BOA (P3) = detecção >= mensal, FP <= mensal, horas <= mensal (pontual) e nenhuma piora declarada.
EXPECTATIVAS: V1-V3 passam (~85%); P1 não sustenta (~90%); P2a dispara em >= 10 das 14 viradas (~50%); ADWIN dispara em >= 10 (~60%); alguma P2 sustenta (~25%);
P3 = mensal ou quase, indistinguível (~55%), TRAVA BOA (~35%).
INTERPRETAÇÃO PREVISTA: alguma P2 sustenta -> o argumento do cliente é "retreino sob demanda com o gatilho X, N retreinos em 14 meses", a validar na sombra. Nenhuma
sustenta -> o argumento é "retreino mensal AUTOMÁTICO (1,5 s, sem intervenção), com ou sem trava", e o sob demanda puro tem custo medido.
LIMITES: 8 eventos; 14 viradas; os gatilhos têm parâmetros a priori (não calibrados com eventos); a trava usa dado da semana de sombra, que pode conter precursor.
VERIFICAÇÕES: V1 a política mensal montada aqui reproduz o cache causal de t, p e sp (<= 1e-9, a partir de 15/01/2025) e o alarme do controle (7/8 FP 9, série idêntica);
V2 a política mensal passada pela mesma máquina de políticas reproduz o alarme da V1 (a versão anterior desta verificação, "P2a com k = 0 dispara em toda
virada", estava errada por causa do piso de 0,05; DECLARADO: ao depurá-la vi que, com o piso, o P2a com k = 0 dispara em 3 das 14 viradas — nenhum parâmetro mudou); V3 causalidade: a decisão de cada gatilho em cada virada não muda se o dado for cortado na virada.

RESULTADO (22/10/2026; worker do ClearML, tarefa 2ae653c1125145409740df89a3551c90; pré-registro a6ca5f5). V1-V3 PASSAM. NENHUMA POLÍTICA SUSTENTA RETREINO SOB DEMANDA.
    política            retreinos  vencidos  detecção  FP  horas   contra a mensal (pareado)
    P0 mensal              14         0        7/8      9   1.457
    P1 congelado            0        13        6/8     24   2.615   FP +1,36/mês [+0,76; +1,99] e horas +105/mês [+21; +206]: PIORA DECLARADA
    P2a k=1,5 / 2 / 3     6 / 2 / 2  4/10/10  6 / 4 / 4  12/15/15  2.921/1.689/1.689   perde 1 a 3 detecções
    P2b q=1,5 / 2 / 3    12 / 8 / 7  0 / 4 / 6   6/8      11/10/9  2.167/1.985/1.593   perde 1 detecção; com q=3 (7 retreinos) FP igual e +12 h/mês
    P2c ADWIN              13         0        7/8      9   1.466   igual à mensal
    P2d Page-Hinkley       12         0        7/8     10   1.468   igual à mensal (+1 FP)
    P3 trava           7 (6 forçados) 0        6/8     24   2.356   FP +1,33/mês [+0,54; +2,24]: PIORA DECLARADA
  LEITURA: (1) os detectores de drift objetivos, com parâmetros padrão e sem calibração, pedem retreino em 12 e 13 das 14 viradas — a "demanda" medida é quase mensal;
  (2) os gatilhos que economizam retreinos (horas acima do limiar, percentil com q alto) pagam com detecção (−1 a −3) e/ou horas; (3) congelar dobra o FP e as horas;
  (4) a trava rejeita a maioria dos bundles novos (baseline "sujo" ou semana de sombra pior), fica com bundle velho até a troca forçada aos 62 d e PIORA — coerente
  com o achado antigo de que o baseline precisa conter anormalidade.
PLACAR: ACERTEI: V1-V3 (~85%); P1 não sustenta (~90%); ADWIN dispara em >= 10 viradas (~60%; 13); nenhuma P2 sustenta (dei ~75% para isso). ERREI: P2a dispara em >= 10
  (~50%; 2 a 6); P3 = mensal ou quase (~55%; pior, com piora declarada de FP); TRAVA BOA (~35%; não).
VEREDITO (como pré-registrado): nenhuma sustenta -> o argumento para o cliente é retreino mensal AUTOMÁTICO (1,5 s, sem intervenção), com o ADWIN como evidência objetiva
  de que a demanda é mensal; o retreino sob demanda puro tem custo medido (tabela acima). A trava, como especificada, não deve ir para produção.

Uso local (encanamento):  SOTA_TESTE=1 PYTHONPATH=src:experiments .venv/bin/python -m sota.h5_retreino
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp h5_retreino --tags sota-h5 --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv.avalia.metricas import episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2
from sota import protocolo as P

TESTE = bool(os.environ.get("SOTA_TESTE"))
ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "h5_retreino"
INICIO_POL = pd.Timestamp("2025-02-01", tz="UTC"); VIRADAS = pd.date_range("2025-03-01", "2026-04-01", freq="MS", tz="UTC")
CB = ("t", "p", "sp"); D14, D7 = pd.Timedelta(days=14), pd.Timedelta(days=7)
GL = {}


def crus(g, mod):
    """Escore cru de t, p (máximo entre tags do erro / sens_p99 / recon_p99) e sp num trecho, com UM bundle — a aritmética de `preprocessar`."""
    out = {}
    for fam, ch in (("temperatura", "t"), ("pressao", "p")):
        cfg = mod["normalizacao"][fam]; X = g[cfg["cols"]]; ok = X.notna().all(axis=1).to_numpy(); v = np.full(len(X), np.nan)
        if ok.any():
            tr = mod[f"{fam}_transformacao"]
            Xs = (X[ok].astype("float64").to_numpy() - np.asarray(tr["center"])) / np.asarray(tr["scale"])
            Z = Xs @ np.asarray(tr["components"]).T - np.asarray(tr["mean_proj"]); rec = Z @ np.asarray(tr["components"]) + np.asarray(tr["mean"])
            v[ok] = np.max(((Xs - rec) ** 2) / np.asarray(cfg["sens_p99"], dtype="float64"), axis=1)
        out[ch] = v / cfg["recon_p99"]
    sm = mod["spread_mancal"]; sp = g[sm["tag_alvo"]].astype("float64") - g[sm["tags_irmaos"]].astype("float64").median(axis=1)
    out["sp"] = np.abs((sp - sm["mediana"]) / sm["mad_robusto"]).to_numpy()
    return pd.DataFrame(out, index=g.index)


def prepara(ci, modelos):
    """Grade completa (CSV + complemento, como o replay) e o escore cru de cada bundle em cada mês em que ele pode estar vigente."""
    csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv")); extra = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet")
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv); raw = pd.concat([extra.loc[extra.index < raw.index[0], raw.columns], raw]).sort_index(); df = ci.preparar_grade(raw)
        real = df["RUNNING_A"].notna() & df[[c for c in df.columns if c.startswith("TV_")]].notna().any(axis=1); g = ci._grade(df.loc[:df.index[real][-1]])
    B = {m["modelo"]["mes_servido"]: m for m in modelos}; meses = pd.period_range(g.index[0].tz_localize(None).to_period("M"), g.index[-1].tz_localize(None).to_period("M"), freq="M")
    mes_de = g.index.tz_localize(None).to_period("M"); R = {}
    for m in meses:
        sel = mes_de == m
        for b in B:
            if pd.Period(b, "M") <= max(m, pd.Period("2025-01", "M")):
                R[(b, str(m))] = crus(g.loc[sel], B[b])
    return g, B, R


def monta(segs, idx, det):
    """segs: lista ordenada de (início, bundle); cada um vale até o próximo. Devolve o EWMA de t, p, sp na grade inteira."""
    R = GL["R"]; mes_de = idx.tz_localize(None).to_period("M"); partes = []
    for i, (a, b) in enumerate(segs):
        fim = segs[i + 1][0] if i + 1 < len(segs) else idx[-1] + pd.Timedelta(minutes=2)
        for m in sorted(set(mes_de[(idx >= a) & (idx < fim)])):
            x = R[(b, str(m))]; partes.append(x.loc[(x.index >= a) & (x.index < fim)])
    cru = pd.concat(partes).reindex(idx)
    ew = pd.DataFrame({c: cru[c].ewm(halflife=pd.Timedelta(det["halflife"][c]), times=idx).mean().where(cru[c].notna()) for c in CB}, index=idx)
    return cru, ew


def decide(ew, det):
    proc = GL["proc"].copy()
    for c in CB:
        proc[c] = ew[c].reindex(proc.index)
    with contextlib.redirect_stdout(io.StringIO()):
        return decide_v2(proc, det)


def segs_iniciais():
    return [(GL["idx"][0], "2025-01"), (INICIO_POL, "2025-02")]


def gatilho(tipo, par, segs, B_, det):
    """Decisão causal na virada B_: só dado anterior a B_."""
    idx = GL["idx"]; cru, ew = monta(segs, idx, det); vig = GL["vig"].reindex(idx).fillna(False)
    D = segs[-1][0]; antes = idx < B_
    if tipo == "P2a":
        for c in CB:
            thr = det["base"][c] * det["k_nivel_a"][c]
            def duty(lo, hi):
                s = vig & (idx >= lo) & (idx < hi) & antes; h = s.sum() * 2 / 60
                return (float((ew[c][s] > thr).mean()), h) if h >= 48 else (np.nan, h)
            fr, _ = duty(D, D + D14); ul, _ = duty(B_ - D14, B_)
            if np.isfinite(fr) and np.isfinite(ul) and ul >= max(par * fr, 0.05):
                return True
        return False
    if tipo == "P2b":
        s = vig & (idx >= B_ - D7) & antes
        if s.sum() * 2 / 60 < 24:
            return False
        return any(float(np.nanpercentile(cru[c][s], 99)) > par for c in ("t", "p"))
    from river import drift
    s = vig & (idx >= D) & antes; ini_mes = B_ - pd.DateOffset(months=1)
    for c in ("t", "p"):
        x = cru[c][s].dropna(); h = x.groupby(x.index.floor("h")).median() if tipo == "P2c" else x.groupby(x.index.floor("h")).quantile(0.95)
        det_ = drift.ADWIN() if tipo == "P2c" else drift.PageHinkley()
        for t, v in h.items():
            det_.update(float(v))
            if det_.drift_detected and t >= ini_mes:
                return True
    return False


def politica(nome, tipo=None, par=None):
    det = GL["det"]; segs = segs_iniciais(); decs = []; forc = 0
    for B_ in VIRADAS:
        mes = f"{B_:%Y-%m}"
        if tipo == "P0":
            troca = True
        elif tipo == "P1":
            troca = False
        elif tipo == "P3":
            troca = trava(segs, B_, mes, det)
            if troca == "forcada":
                forc += 1
            if troca:
                segs.append((B_ + D7, mes))
            decs.append((mes, bool(troca))); continue
        else:
            troca = gatilho(tipo, par, segs, B_, det)
        if troca:
            segs.append((B_, mes))
        decs.append((mes, troca))
    _, ew = monta(segs, GL["idx"], det); al = decide(ew, det)["is_anomaly"].astype(bool)
    idade = [(B_ - pd.Timestamp(GL["B"][next(b for a, b in reversed(segs) if a <= B_ - pd.Timedelta(minutes=2))]["modelo"]["baseline_fim"])).days for B_ in list(VIRADAS[1:]) + [P.FIM]]
    return dict(nome=nome, segs=segs, decs=decs, trocas=sum(d for _, d in decs), forcadas=forc, vencidos=sum(x > 62 for x in idade), alarme=al)


def trava(segs, B_, mes, det):
    nb = GL["B"][mes]["modelo"]; ini, fim = pd.Timestamp(nb["baseline_inicio"]), pd.Timestamp(nb["baseline_fim"])
    vig_atual = GL["B"][segs[-1][1]]["modelo"]
    if (B_ + D7 - pd.Timestamp(vig_atual["baseline_fim"])).days > 62:
        return "forcada"
    conhecidos = [t for t in GL["todos_trips"] if t < B_]
    if any(ini <= t + pd.Timedelta(days=2) and fim >= t - pd.Timedelta(hours=48) for t in conhecidos):
        return False
    _, ew0 = monta(segs, GL["idx"], det); al0 = decide(ew0, det)["is_anomaly"].astype(bool)
    est = GL["estavel"].reindex(GL["idx"]).fillna(False); base = est & (GL["idx"] >= ini) & (GL["idx"] <= fim)
    if float(al0[base].mean()) > 0.10:
        return False
    _, ew1 = monta(segs + [(B_, mes)], GL["idx"], det); al1 = decide(ew1, det)["is_anomaly"].astype(bool)
    w = (GL["idx"] >= B_) & (GL["idx"] < B_ + D7)
    h0, h1 = al0[w].sum(), al1[w].sum(); e0, e1 = len(episodios_de_serie(al0[w])), len(episodios_de_serie(al1[w]))
    return bool(h1 <= h0 and e1 <= e0)


def main():
    t0 = time.time(); ci = _ci(); SAIDA.mkdir(parents=True, exist_ok=True); ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; GL.update(ci=ci, det=det)
    g, Bd, R = prepara(ci, modelos); GL.update(B=Bd, R=R)
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    idx = g.index.intersection(proc.index); GL.update(idx=idx, proc=proc.loc[idx], vig=proc["mask"].loc[idx])
    GL["estavel"] = (g["RUNNING_A"] > 0.5).fillna(False) & (g["T5_AVG_A"] > 300)
    GL["todos_trips"] = sorted(carrega_registro()); trips = P.trips_alvo(GL["todos_trips"])
    from nv.avalia import carrega_controle
    al_c, vig_c = carrega_controle("atual")
    print(f"preparo: {len(R)} combinações bundle x mês ({time.time() - t0:.0f} s)", flush=True)
    seg_m = segs_iniciais() + [(B_, f"{B_:%Y-%m}") for B_ in VIRADAS]; cru, ew = monta(seg_m, idx, det)
    j = idx[idx >= pd.Timestamp("2025-01-15", tz="UTC")]
    dif = {c: float(np.nanmax(np.abs(ew.loc[j, c].to_numpy() - S.loc[j, c].to_numpy()))) for c in CB}; nan = {c: int((ew.loc[j, c].isna() != S.loc[j, c].isna()).sum()) for c in CB}
    al_m = decide(ew, det)["is_anomaly"].astype(bool); mm = P.mede(al_m, vig_c, trips); co = float((al_m.reindex(al_c.index).fillna(False).loc[P.INICIO:P.FIM] == al_c.loc[P.INICIO:P.FIM]).mean())
    ok["V1"] = all(v <= 1e-9 for v in dif.values()) and all(v == 0 for v in nan.values()) and (mm["det"], mm["fp"]) == (7, 9) and co == 1.0
    print(f"V1 — mensal montada x cache: {({k: f'{v:.1e}' for k, v in dif.items()})}, NaN {nan}; {mm['det']}/8 FP {mm['fp']}; alarme idêntico ao controle em {100 * co:.3f}% -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)
    if not ok["V1"]:
        print("verificação falhou — parando.", flush=True); return
    v2 = politica("P0 mensal", "P0"); ok["V2"] = v2["trocas"] == 14 and bool((v2["alarme"] == al_m).all())
    print(f"V2 — mensal pela máquina de políticas: {v2['trocas']} trocas, alarme igual ao da V1: {bool((v2['alarme'] == al_m).all())} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    seg0 = segs_iniciais(); v3 = []
    for tipo, par in (("P2a", 2.0), ("P2b", 2.0), ("P2c", None), ("P2d", None)):
        for B_ in VIRADAS[:3 if TESTE else 6]:
            d1 = gatilho(tipo, par, seg0, B_, det); GL["_idx"] = GL["idx"]; GL["idx"] = idx[idx < B_]
            GL["vig"] = proc["mask"].loc[GL["idx"]]; d2 = gatilho(tipo, par, seg0, B_, det); GL["idx"] = GL["_idx"]; GL["vig"] = proc["mask"].loc[idx]
            v3.append(d1 == d2)
    ok["V3"] = all(v3); print(f"V3 — decisão do gatilho igual com o dado cortado na virada em {sum(v3)}/{len(v3)} -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    POL = [("P0 mensal", "P0", None), ("P1 congelado", "P1", None)]
    POL += [(f"P2a k={k}", "P2a", k) for k in (1.5, 2.0, 3.0)] + [(f"P2b q={q}", "P2b", q) for q in (1.5, 2.0, 3.0)]
    POL += [("P2c ADWIN", "P2c", None), ("P2d Page-Hinkley", "P2d", None), ("P3 trava", "P3", None)]
    if TESTE:
        POL = POL[:3] + [POL[-1]]
    RES = {}
    for nome, tipo, par in POL:
        r = politica(nome, tipo, par); r["m"] = P.mede(r["alarme"], vig_c, trips); r["ic"] = P.ic(r["alarme"], vig_c, trips, n=400 if TESTE else P.N_BOOT)
        r["cmp"] = P.compara(r["alarme"], al_m, vig_c, trips, n=400 if TESTE else P.N_BOOT); RES[nome] = r
        print(f"   {nome:18s} pronta ({time.time() - t0:.0f} s)", flush=True)
    GL["RES"] = RES
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(escreve=False)
        print(f"TESTE: encanamento ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). {ok}", flush=True); return
    relata(escreve=True); print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


def relata(escreve):
    RES = GL["RES"]; m0 = RES["P0 mensal"]["m"]
    print("\nPOLÍTICAS DE RETREINO (14 viradas, 01/03/2025 a 01/04/2026; protocolo único; comparação pareada contra a mensal)", flush=True)
    for nome, r in RES.items():
        m, c, d = r["m"], r["ic"], r["cmp"]
        print(f" {nome:18s} retreinos {r['trocas']:2d} (forçados {r['forcadas']}) | meses com bundle vencido {r['vencidos']:2d} | {m['det']}/8 [{c['det'][0]:.0f}-{c['det'][1]:.0f}] "
              f"início {m['inicio']} banda {m['banda']} | FP {m['fp']:2d} ({m['fp_mes']:.3f}/mês [{c['fp_mes'][0]:.2f}-{c['fp_mes'][1]:.2f}]) | {m['horas']:5.0f} h ({m['horas_mes']:.1f}/mês) | "
              f"Δdet {d['d_det']:+d} ΔFP/mês {d['d_fp_mes']:+.2f} [{d['ic_d_fp_mes'][0]:+.2f}, {d['ic_d_fp_mes'][1]:+.2f}] Δh/mês {d['d_horas_mes']:+.1f} [{d['ic_d_horas_mes'][0]:+.1f}, {d['ic_d_horas_mes'][1]:+.1f}]", flush=True)
    print("\nDECISÃO EM CADA VIRADA (1 = retreina)", flush=True)
    print("   " + " ".join(f"{m[2:]:>5s}" for m, _ in RES["P0 mensal"]["decs"]), flush=True)
    for nome, r in RES.items():
        print(f"   " + " ".join(f"{int(bool(x)):>5d}" for _, x in r["decs"]) + f"   {nome}", flush=True)
    piora = lambda d: (d["difere_det"] and d["d_det"] < 0) or (d["difere_fp"] and d["d_fp_mes"] > 0) or (d["difere_horas"] and d["d_horas_mes"] > 0)
    sust = lambda r: r["trocas"] <= 7 and not piora(r["cmp"]) and r["m"]["det"] >= m0["det"] - 1 and r["m"]["horas"] <= 1.25 * m0["horas"]
    print("\nCLASSIFICAÇÃO", flush=True)
    for fam, nomes in (("P1", ["P1 congelado"]), ("P2a", ["P2a k=1.5", "P2a k=2.0", "P2a k=3.0"]), ("P2b", ["P2b q=1.5", "P2b q=2.0", "P2b q=3.0"]), ("P2c", ["P2c ADWIN"]), ("P2d", ["P2d Page-Hinkley"])):
        nomes = [n for n in nomes if n in RES]
        if not nomes:
            continue
        principal = nomes[len(nomes) // 2] if len(nomes) == 3 else nomes[0]
        s = sust(RES[principal]) and all(sust(RES[n]) for n in nomes)
        print(f"   {fam}: {'SUSTENTA retreino sob demanda' if s else 'NÃO sustenta'} (principal {principal}: {RES[principal]['trocas']} retreinos, sustenta {sust(RES[principal])}; vizinhos {[sust(RES[n]) for n in nomes if n != principal]})", flush=True)
    if "P3 trava" in RES:
        r = RES["P3 trava"]; boa = r["m"]["det"] >= m0["det"] and r["m"]["fp"] <= m0["fp"] and r["m"]["horas"] <= m0["horas"] and not piora(r["cmp"])
        print(f"   P3: {'TRAVA BOA' if boa else 'trava não melhora'} ({r['trocas']} retreinos, {r['forcadas']} forçados)", flush=True)
    if escreve:
        pd.DataFrame([dict(politica=n, retreinos=r["trocas"], forcados=r["forcadas"], meses_vencidos=r["vencidos"], **r["m"], **{f"cmp_{k}": v for k, v in r["cmp"].items()}) for n, r in RES.items()]).to_csv(SAIDA / "politicas.csv", index=False)
        pd.DataFrame({n: [int(bool(x)) for _, x in r["decs"]] for n, r in RES.items()}, index=[m for m, _ in RES["P0 mensal"]["decs"]]).to_csv(SAIDA / "decisoes_por_virada.csv")


if __name__ == "__main__":
    main()
