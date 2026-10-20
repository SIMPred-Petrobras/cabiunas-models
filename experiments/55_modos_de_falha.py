#!/usr/bin/env python3
"""EXP 55 — REANÁLISE POR MODO DE FALHA (etapas 2 e 3): cada modo é visível nas nossas tags? e cada episódio do detector vem de qual família física?

PRÉ-REGISTRO (20/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp55.py`), branch exp/modos-de-falha.

POR QUÊ. Pelo diário de alarmes da planta, os 8 trips têm TRÊS causas físicas: mancal radial LNA do compressor (TI_0305: 17/03, 07/04, 11/04, 09/12), selagem (PDIT_0305 em 27/02,
PDI_0302 em 29/04) e pressão de óleo lubrificante (PALL_6240309 / PI_0339/0340: 04/11, 26/02/2026). Até aqui o detector foi medido contra os 8 como um fenômeno só — o que explica
"os oito eventos não têm forma comum" e "não existe direção compartilhada". Esta reanálise troca a pergunta: em vez de "qual detector?", "o que é observável, de qual modo, com
quanta antecedência, e por qual família o detector acerta e erra?". Não muda o detector; produz decisões (onde intervir, o que pedir de dado, como atribuir o alarme).
JÁ VISTO HOJE (exploratório, declarado): com 131 janelas normais, a família do próprio modo foi a mais desviada em 6 de 8 trips no horizonte de 48 h; selagem forte (percentil 0,93 e
0,98), mancal moderado (0,85-0,95), óleo fraco (0,76 e 0,85). O horizonte de 48 h da etapa 2 é, portanto, em parte REPLICAÇÃO; o novo são os horizontes de 6 h e 1 h, o nulo maior,
a atribuição dos episódios (inclusive os 9 FP) e a decomposição das detecções.

DEFINIÇÕES FIXADAS ANTES (pela planilha de limites e pelo P&ID, não pelos eventos):
  FAMÍLIAS (36 tags, disjuntas): mancal = TI_0301/0303/0305/0307 (mancais do compressor); selagem = PDIT_0305, PDI_0302, PDI_0301, PI_0307 (gás de selagem e balanceamento);
  óleo = PI_0339, PI_0340, PI_0308, PDI_0338, TI_0325 (óleo lubrificante); vibração = TV_351..355 X/Y (turbina); gás = PI_0315, PI_0319, PDI_0317, PI_5134001 (combustível,
  partida, ar de barreira); exaustão = T5_AVG_A, TC382_01..06, TI_0315, TI_0317.
  MODO DE CADA TRIP: a primeira ativação (ACT) no diário em [T - 20 min, T + 5 min] de uma tag de trip; TI_03xx -> mancal; PDIT_0305/PDI_0302/PDI_0301 -> selagem;
  PALL_6240309/PI_0339/PI_0340/PI_0308 -> óleo.
  ESTATÍSTICA DA FAMÍLIA numa janela: para cada tag, z robusto contra a referência CAUSAL [fim - 28 d, fim - 7 d] em operação estável (RUNNING_A > 0,5 e T5 > 300); |z| sustentado
  30 min (mediana móvel de 15 amostras); máximo sobre as tags da família e o tempo. Só instantes estáveis. Escala = 1,4826 x MAD (desvio padrão se MAD = 0; tag fora se ambos 0 ou
  < 100 amostras na referência).
  HORIZONTES (etapa 2): janela [T - H, T - 10 min] com H = 48 h, 6 h e 1 h (os 10 min finais excluídos: é a cascata do próprio trip).
  NULO: 400 instantes de operação estável entre 01/02/2025 e 20/04/2026, a mais de 7 d de qualquer trip do registro, semente 20261021; mesma construção. Percentil = fração do nulo
  abaixo do valor do evento.
  OBSERVÁVEL (por evento e horizonte): percentil da família do próprio modo >= 0,95; FRACO 0,80-0,95; INVISÍVEL < 0,80.
  EPISÓDIOS (etapa 3): os 18 do detector publicado desde 01/02/2025 (`ci.prever` sobre o cache causal; TP = intersecta [T - 48 h, T]).
    A1 (independente do v2): família de maior percentil na janela de NOTIFICAÇÃO [nascimento, nascimento + 2 h], nulo com a mesma janela de 2 h.
    A2 (de dentro do v2): famílias dos canais acesos na notificação; sp -> mancal, vb -> vibração, t e p -> família do sensor que mais pesa no erro de reconstrução (argmax de
    erro/sens_p99 do bundle em vigor, o mais frequente nos instantes em que o canal está aceso).
  DECOMPOSIÇÃO DAS DETECÇÕES: um trip detectado é CAUSAL se a família do seu modo aparece em A2 calculado na interseção episódio x [T - 48 h, T] E o percentil da família do modo a
  48 h é >= 0,80; senão INDIRETA.

MEDIDAS E CRITÉRIOS (fixados antes):
  M1 observabilidade: tabela 8 eventos x 6 famílias x 3 horizontes; resumo por modo.
  M2 acurácia da atribuição A1 nos 7 episódios de trip detectados, entre os 3 modos (acaso 1/3) e entre as 6 famílias (acaso 1/6), com p binomial unilateral.
     ATRIBUIÇÃO ÚTIL = >= 5 de 7 entre os 3 modos (p < 0,05 contra 1/3).
  M3 FP por família (A1 e A2): quantos dos 9 FP caem em famílias que NÃO são causa de nenhum trip (vibração, gás, exaustão).
  M4 detecções causais x indiretas.
EXPECTATIVAS: os 8 trips recebem modo 4/2/2 (~99%, já visto); a 48 h, família do modo >= 0,80 em 4/4 de mancal (~70%), 2/2 de selagem (~80%) e <= 1/2 de óleo (~70%); a 1 h,
mancal OBSERVÁVEL (>= 0,95) em >= 3/4 (~60%); óleo INVISÍVEL nos três horizontes em pelo menos 1 dos 2 (~65%); ATRIBUIÇÃO ÚTIL (~45%); >= 5 dos 9 FP em famílias sem trip, por A1
(~55%); >= 5 das 7 detecções CAUSAIS (~50%).
INTERPRETAÇÃO PREVISTA: óleo invisível -> argumento quantificado para pedir as tags do sistema de óleo, e teto realista por modo; FP concentrados em famílias sem trip -> candidata
(etapa 4, pré-registro próprio) a exigir apoio da família de um modo para notificar o nível B; selagem visível mas não capturada pelo `p` (29/04) -> sub-canal de selagem (etapa 4);
ATRIBUIÇÃO ÚTIL -> camada de diagnóstico "onde olhar" para a apresentação (sem efeito no FP).
LIMITES: 4/2/2 eventos por modo — tudo aqui é descritivo, com nulo, sem poder estatístico por modo; a 48 h é em parte replicação de uma olhada exploratória; as janelas nulas
excluem +-7 d dos trips; nada se adota sem a sombra.
VERIFICAÇÕES: V1 `ci.prever` sobre o cache dá 7/8 e FP 9; V2 famílias com 36 tags disjuntas, todas na grade; V3 os 8 trips recebem modo; V4 >= 300 valores válidos no nulo por
família a 48 h; V5 o argmax por sensor reproduz `ci._recon_pca` (diferença <= 1e-9 no máximo normalizado) num mês de amostra.

RESULTADO (20/10/2026; worker do ClearML, tarefa 7399ed7ebe1b46b28e84dbfd648b0d9c, 39 s, 64 CPUs; pré-registro e86c51e, correção do worker 48c85aa; a 1ª tarefa,
f042bca0, falhou antes do experimento: o --insumos não chegava ao worker). V1-V5 PASSAM (nulo com 390 janelas válidas por família).
  ETAPA 2 — OBSERVABILIDADE (percentil da família do próprio modo; O >= 0,95, F 0,80-0,95, I < 0,80):
                     48 h          6 h           1 h
    SELAGEM 27/02    0,97 O        0,99 O        0,99 O      o modo MAIS visível, e fica mais visível perto do trip
            29/04    0,93 F        0,97 O        0,97 O
    MANCAL  17/03    0,96 O        0,99 O        0,99 O
            07/04    0,94 F        0,98 O        0,97 O
            11/04    0,92 F        0,91 F        0,89 F
            09/12    0,87 F        0,63 I        0,55 I      súbito: a subida final cabe nos 10 min excluídos
    ÓLEO    04/11    0,87 F        0,66 I        0,69 I      fraco em todos os horizontes; nenhum O
            26/02    0,77 I        0,89 F        0,91 F
  ETAPA 3 — ATRIBUIÇÃO: A1 acerta o modo em 5 de 7 trips detectados (acaso 2,3; p 0,045 entre os 3 modos; p 0,002 entre as 6 famílias) -> ATRIBUIÇÃO ÚTIL pelo critério. Erra o 11/04
  (mancal; A1 aponta exaustão, e óleo entre os modos) e o 26/02/2026 (óleo; A1 aponta selagem — a selagem está a 0,95 a 48 h antes desse trip de óleo).
  FP por família (A1): vibração 2, gás 2, selagem 2, exaustão 1, mancal 1, óleo 1 -> 5/9 em famílias sem trip, MAS o acaso é 4,5 (3 de 6 famílias): o critério que escrevi não
  comparava com o acaso e foi cumprido trivialmente. LEITURA CORRETA: os FP NÃO se concentram numa família. Pelo A2, a vibração está em 8 de 9 FP e em todos os TP (o vb é estado).
  DETECÇÕES: 5 de 7 CAUSAIS (27/02, 17/03, 07/04, 11/04, 09/12); INDIRETAS o 29/04 (selagem visível a 0,93-0,97, mas o canal p do v2 acendeu por um sensor de ÓLEO, mais a
  vibração) e o 26/02/2026 (óleo fraco, 0,77; detectado pelo episódio travado de 647 h, com todas as famílias). O 04/11 (óleo) segue perdido.
  PÓS-HOC, NÃO PRÉ-REGISTRADO: a força da família física na notificação (máximo de A1) é maior nos TP (mediana 0,979) que nos FP (0,828): AUC 0,746, p 0,056. É da mesma família
  das regras de força que já morreram no teste (força no nascimento AUC 0,70 nominal, piso de força, portão P1b): no máximo hipótese, e qualquer limiar sairia destes eventos.
PLACAR: ACERTEI: modos 4/2/2 (~99%); família do modo >= 0,80 a 48 h em 4/4 de mancal (~70%) e 2/2 de selagem (~80%); óleo <= 1/2 a 48 h (~70%); ATRIBUIÇÃO ÚTIL (~45%); >= 5 de 7
  causais (~50%); >= 5 de 9 FP em famílias sem trip (~55%), mas trivial pelo acaso. ERREI: mancal OBSERVÁVEL a 1 h em >= 3/4 (~60%; foram 2 — o 09/12 é súbito e o 11/04 fica em
  F); óleo INVISÍVEL nos três horizontes em pelo menos 1 dos 2 (~65%; é FRACO, não invisível: cada um tem um horizonte em F).
DECISÕES (como pré-registrado): (1) ÓLEO: sem nenhum horizonte observável nas nossas tags, 2 de 8 trips (25%) — argumento quantificado para pedir o sistema de óleo; (2) SELAGEM:
visível e a mais forte perto do trip, mas o 29/04 entrou no v2 por um sensor de óleo -> o sub-canal de selagem (etapa 4) tem motivação medida; (3) a camada "onde olhar" passa
no critério (5/7), ainda com 7 eventos; (4) FP não têm família dominante: não há veto por família a propor.

Uso local (teste de encanamento, sem tabelas):  EXP55_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/55_modos_de_falha.py
No worker:  PYTHONPATH=src python scripts/worker_exp55.py --enqueue --queue default --insumos ID
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import episodios_de_serie, JANELA
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

ROOT = HERE.parent / "data_local"
ALARMES = Path(os.environ.get("ALARMES_CSV", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv"))
TESTE = bool(os.environ.get("EXP55_TESTE"))
INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); FIM_NULO = pd.Timestamp("2026-04-20", tz="UTC")
P = "954005_624_"
FAMILIAS = {
    "mancal": [P + "TI_0301", P + "TI_0303", P + "TI_0305", P + "TI_0307"],
    "selagem": [P + "PDIT_0305", P + "PDI_0302", P + "PDI_0301", P + "PI_0307"],
    "oleo": [P + "PI_0339", P + "PI_0340", P + "PI_0308", P + "PDI_0338", P + "TI_0325"],
    "vibracao": [f"TV_35{i}{e}_A" for i in range(1, 6) for e in "XY"],
    "gas": [P + "PI_0315", P + "PI_0319", P + "PDI_0317", "PI_5134001"],
    "exaustao": ["T5_AVG_A"] + [f"TC382_0{i}_A" for i in range(1, 7)] + [P + "TI_0315", P + "TI_0317"],
}
COLS = [c for v in FAMILIAS.values() for c in v]
FAM_DE = {c: f for f, v in FAMILIAS.items() for c in v}
MODOS = ("mancal", "selagem", "oleo"); SEM_TRIP = ("vibracao", "gas", "exaustao")
TAG_MODO = {"TI_0301": "mancal", "TI_0303": "mancal", "TI_0305": "mancal", "TI_0307": "mancal", "PDIT_0305": "selagem", "PDI_0302": "selagem", "PDI_0301": "selagem",
            "PALL_6240309": "oleo", "PI_0339": "oleo", "PI_0340": "oleo", "PI_0308": "oleo"}
HORIZ = {"48h": 48.0, "6h": 6.0, "1h": 1.0}
N_NULO = 20 if TESTE else 400; SEMENTE = 20261021
REF_INI, REF_FIM = pd.Timedelta(days=28), pd.Timedelta(days=7)
CORTE = pd.Timedelta(minutes=10); NOTIF = pd.Timedelta(hours=2); SUST = 15


def modos_dos_trips(trips):
    a = pd.read_csv(ALARMES)
    a["t"] = pd.to_datetime(a["Data da Ocorrência"]).dt.tz_localize("UTC")
    a = a[a["Status"].astype(str).str.startswith("ACT")].sort_values("t")
    out = {}
    for T in trips:
        w = a[(a.t >= T - pd.Timedelta(minutes=20)) & (a.t <= T + pd.Timedelta(minutes=5))]
        out[T] = next((TAG_MODO[k] for k in w["Tag Alarme"].astype(str).str.replace(P, "", regex=False) if k in TAG_MODO), None)
    return out


def refstats(g, est, ref_ate):
    r = g.loc[ref_ate - REF_INI:ref_ate - REF_FIM, COLS]; r = r[est.reindex(r.index).fillna(False).to_numpy()]
    if len(r) < 500:
        return None
    med = r.median(); mad = 1.4826 * (r - med).abs().median(); sd = r.std(); n = r.notna().sum()
    esc = mad.where(mad > 0, sd); esc = esc.where((esc > 0) & (n >= 100))
    return med, esc


def stat_janela(g, est, ini, fim, rs):
    if rs is None:
        return {f: np.nan for f in FAMILIAS}
    med, esc = rs
    w = g.loc[ini:fim, COLS]; w = w[est.reindex(w.index).fillna(False).to_numpy()]
    if len(w) < SUST:
        return {f: np.nan for f in FAMILIAS}
    z = ((w - med) / esc).abs().rolling(SUST, min_periods=SUST).median().max()
    return {f: (float(z[t].max()) if z[t].notna().any() else np.nan) for f, t in FAMILIAS.items()}


def janelas_trip(g, est, T):
    rs = refstats(g, est, T)
    return {h: stat_janela(g, est, T - pd.Timedelta(hours=H), T - CORTE, rs) for h, H in HORIZ.items()}


def janela_notif(g, est, a):
    return stat_janela(g, est, a, a + NOTIF, refstats(g, est, a + NOTIF))


def pct(v, nulo):
    x = np.asarray([u for u in nulo if np.isfinite(u)])
    return float((x < v).mean()) if np.isfinite(v) and len(x) else np.nan


def erro_por_sensor(ci, modelos, fam, X):
    """err/sens_p99 por sensor, com o bundle em vigor no mês de X; e o máximo normalizado como em `ci._recon_pca`."""
    m0 = pd.Timestamp(X.index[0].year, X.index[0].month, 1, tz="UTC")
    mod = ci.vigencia(modelos, max(m0, X.index[0])); cfg = mod["normalizacao"][fam]; cols = cfg["cols"]; sens = np.asarray(cfg["sens_p99"], dtype="float64")
    tr = mod[f"{fam}_transformacao"]; Xc = X[cols]; ok = Xc.notna().all(axis=1).to_numpy()
    E = np.full((len(Xc), len(cols)), np.nan)
    if ok.any():
        Xs = (Xc[ok].astype("float64").to_numpy() - tr["center"]) / tr["scale"]
        Z = Xs @ tr["components"].T - tr["mean_proj"]; rec = Z @ tr["components"] + tr["mean"]
        E[ok] = (Xs - rec) ** 2 / sens
    return pd.DataFrame(E, index=X.index, columns=cols), cfg["recon_p99"]


def familias_v2(ci, modelos, g, res, a, b):
    """A2: famílias dos canais acesos em [a, b]."""
    w = res.loc[a:b]; fams = set()
    for c in ("t", "p", "sp", "vb"):
        aceso = (w[f"a_{c}"].astype(bool) | w[f"b_{c}"].astype(bool))
        if not aceso.any():
            continue
        if c == "sp":
            fams.add("mancal")
        elif c == "vb":
            fams.add("vibracao")
        else:
            idx = aceso[aceso].index
            E, _ = erro_por_sensor(ci, modelos, "temperatura" if c == "t" else "pressao", g.loc[idx])
            top = E.dropna(how="all").idxmax(axis=1).value_counts()
            if len(top):
                fams.add(FAM_DE.get(top.index[0], "outra"))
    return fams


def main():
    t0 = time.time(); ok = {}
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    reg = carrega_registro(); trips = [t for t in reg if t >= INICIO8]
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]

    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        res = ci.prever({"detector": det}, proc)
    al = res["is_anomaly"].astype(bool)
    r = avalia(al.reindex(al_c.index).fillna(False), vig_c, trips, inicio_periodo=INICIO8)
    ok["V1"] = r.det == 7 and r.fp == 9
    print(f"V1 — prever sobre o cache: {r.det}/{r.n_eventos}, FP {r.fp} -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)
    ok["V2"] = len(COLS) == 36 and len(set(COLS)) == 36 and all(c in g.columns for c in COLS)
    print(f"V2 — famílias: {len(COLS)} tags, {len(set(COLS))} distintas, ausentes da grade: {[c for c in COLS if c not in g.columns]} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    modo = modos_dos_trips(trips)
    ok["V3"] = all(v is not None for v in modo.values()) and len(trips) == 8
    print(f"V3 — modo de cada trip: {[(t.strftime('%d/%m/%Y'), m) for t, m in modo.items()]} -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    m0 = pd.Timestamp("2025-06-01", tz="UTC"); X = g.loc[m0:m0 + pd.Timedelta(days=3)]
    v5 = 0.0
    for fam in ("temperatura", "pressao"):
        E, p99 = erro_por_sensor(ci, modelos, fam, X); mine = (E.max(axis=1) / p99).to_numpy()
        ref = ci._recon_pca(ci.vigencia(modelos, m0), fam, X); m = np.isfinite(mine) & np.isfinite(ref)
        v5 = max(v5, float(np.max(np.abs(mine[m] - ref[m]))) if m.any() else np.inf)
    ok["V5"] = v5 <= 1e-9
    print(f"V5 — argmax por sensor x ci._recon_pca: diferença máxima {v5:.1e} -> {'PASSA' if ok['V5'] else 'FALHA'}", flush=True)

    rng = np.random.default_rng(SEMENTE)
    cand = est[est].index; cand = cand[(cand >= INICIO8) & (cand <= FIM_NULO)]
    longe = np.ones(len(cand), bool)
    for t in reg:
        longe &= np.abs((cand - t).total_seconds().to_numpy()) > 7 * 86400
    sel = cand[longe]; ends = sel[np.sort(rng.choice(len(sel), N_NULO, replace=False))]
    NUL = {h: {f: [] for f in FAMILIAS} for h in list(HORIZ) + ["notif"]}
    for E_ in ends:
        jt = janelas_trip(g, est, E_)
        for h in HORIZ:
            for f in FAMILIAS:
                NUL[h][f].append(jt[h][f])
        jn = stat_janela(g, est, E_ - NOTIF, E_, refstats(g, est, E_))
        for f in FAMILIAS:
            NUL["notif"][f].append(jn[f])
    nval = {f: int(np.isfinite(NUL["48h"][f]).sum()) for f in FAMILIAS}
    ok["V4"] = TESTE or min(nval.values()) >= 300
    print(f"V4 — valores válidos no nulo a 48 h por família: {nval} -> {'PASSA' if ok['V4'] else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    if not all(ok.values()):
        print("verificação falhou — parando.", flush=True); return
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            etapas(ci, modelos, g, est, trips, modo, res, al, fim, NUL, ok, t0, escreve=False)
        print(f"TESTE: etapas 2 e 3 executaram sem erro em {time.time() - t0:.0f} s (saída suprimida, nada gravado).", flush=True); return
    etapas(ci, modelos, g, est, trips, modo, res, al, fim, NUL, ok, t0, escreve=True)


def etapas(ci, modelos, g, est, trips, modo, res, al, fim, NUL, ok, t0, escreve):
    # ── ETAPA 2: observabilidade ──
    L2 = []
    print("\nETAPA 2 — OBSERVABILIDADE (percentil de cada família contra o nulo; * = família do modo; O >= 0,95, F 0,80-0,95, I < 0,80)", flush=True)
    for T in trips:
        jt = janelas_trip(g, est, T)
        for h in HORIZ:
            pc = {f: pct(jt[h][f], NUL[h][f]) for f in FAMILIAS}; pm = pc[modo[T]]
            cl = "O" if pm >= 0.95 else "F" if pm >= 0.80 else "I"
            L2.append(dict(evento=T, modo=modo[T], horizonte=h, classe=cl, **{f"pct_{f}": pc[f] for f in FAMILIAS}, **{f"z_{f}": jt[h][f] for f in FAMILIAS}))
            print(f"   {T.strftime('%d/%m/%Y')} {modo[T]:7s} {h:3s} [{cl}] | " + " | ".join(f"{f}{'*' if f == modo[T] else ' '} {pc[f]:.2f}" for f in FAMILIAS), flush=True)
    d2 = pd.DataFrame(L2)
    print("\n   resumo por modo (nº de eventos O/F/I por horizonte):", flush=True)
    for m in MODOS:
        for h in HORIZ:
            s = d2[(d2.modo == m) & (d2.horizonte == h)].classe.value_counts().reindex(["O", "F", "I"], fill_value=0)
            print(f"   {m:7s} {h:3s}: O {s['O']} | F {s['F']} | I {s['I']}", flush=True)

    # ── ETAPA 3: atribuição ──
    eps = [(a, b) for a, b in episodios_de_serie(al.loc[INICIO8:fim])]
    L3 = []
    print("\nETAPA 3 — ATRIBUIÇÃO DOS EPISÓDIOS (A1 = família de maior percentil na notificação; A2 = famílias dos canais do v2)", flush=True)
    for a, b in eps:
        ts = [T for T in trips if a <= T and b >= T - JANELA]; cls = "TP" if ts else "FP"
        jn = janela_notif(g, est, a); pc = {f: pct(jn[f], NUL["notif"][f]) for f in FAMILIAS}
        a1 = max((f for f in FAMILIAS if np.isfinite(pc[f])), key=lambda f: pc[f], default=None)
        a1m = max((f for f in MODOS if np.isfinite(pc[f])), key=lambda f: pc[f], default=None)
        a2 = familias_v2(ci, modelos, g, res, a, min(b, a + NOTIF))
        m_ = modo[ts[-1]] if ts else None
        L3.append(dict(inicio=a, fim=b, horas=(b - a).total_seconds() / 3600, classe=cls, trip=ts[-1] if ts else None, modo=m_, A1=a1, A1_modos=a1m, A2=",".join(sorted(a2)),
                       **{f"pct_{f}": pc[f] for f in FAMILIAS}))
        print(f"   {a.strftime('%d/%m/%Y %H:%M')} {(b - a).total_seconds() / 3600:6.1f} h {cls} {('(' + m_ + ')') if m_ else '':10s} | A1 {a1} ({pc.get(a1, np.nan):.2f}); entre modos {a1m} | A2 {sorted(a2)}", flush=True)
    d3 = pd.DataFrame(L3)
    tp = d3[d3.classe == "TP"].groupby("trip").tail(1)       # o último episódio antes de cada trip detectado
    acerto3 = int((tp.A1_modos == tp.modo).sum()); acerto6 = int((tp.A1 == tp.modo).sum()); n = len(tp)
    p3 = float(stats.binom.sf(acerto3 - 1, n, 1 / 3)); p6 = float(stats.binom.sf(acerto6 - 1, n, 1 / 6))
    print(f"\n   M2 atribuição A1 nos {n} trips detectados: entre os 3 modos {acerto3}/{n} (acaso {n / 3:.1f}, p {p3:.3f}); entre as 6 famílias {acerto6}/{n} (acaso {n / 6:.1f}, p {p6:.3f})"
          f" -> {'ATRIBUIÇÃO ÚTIL' if acerto3 >= 5 else 'não útil'}", flush=True)
    fp = d3[d3.classe == "FP"]
    print(f"   M3 FP por família A1: {fp.A1.value_counts().to_dict()} | em famílias sem trip: {int(fp.A1.isin(SEM_TRIP).sum())}/{len(fp)}", flush=True)
    print(f"      FP por A2 (contagem de famílias implicadas): {pd.Series([f for s in fp.A2 for f in s.split(',') if f]).value_counts().to_dict()}", flush=True)
    print(f"      TP por A1: {d3[d3.classe == 'TP'].A1.value_counts().to_dict()}", flush=True)

    L4 = []
    print("\n   M4 decomposição das detecções:", flush=True)
    for T in trips:
        inter = [(a, b) for a, b in eps if a <= T and b >= T - JANELA]
        if not inter:
            print(f"   {T.strftime('%d/%m/%Y')} {modo[T]:7s}: não detectado", flush=True); continue
        fams = set()
        for a, b in inter:
            fams |= familias_v2(ci, modelos, g, res, max(a, T - JANELA), min(b, T))
        p48 = float(d2[(d2.evento == T) & (d2.horizonte == "48h")][f"pct_{modo[T]}"].iloc[0])
        causal = modo[T] in fams and p48 >= 0.80
        L4.append(dict(evento=T, modo=modo[T], familias_v2=",".join(sorted(fams)), pct48=p48, causal=causal))
        print(f"   {T.strftime('%d/%m/%Y')} {modo[T]:7s}: famílias do v2 {sorted(fams)} | percentil 48 h do modo {p48:.2f} -> {'CAUSAL' if causal else 'INDIRETA'}", flush=True)
    print(f"   causais: {sum(x['causal'] for x in L4)}/{len(L4)}", flush=True)
    if escreve:
        d2.to_csv(ROOT / "exp55_observabilidade.csv", index=False); d3.to_csv(ROOT / "exp55_episodios.csv", index=False); pd.DataFrame(L4).to_csv(ROOT / "exp55_deteccoes.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
