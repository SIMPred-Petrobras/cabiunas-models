#!/usr/bin/env python3
"""EXP 54 — A META DO USUÁRIO MEDIDA EM DETECÇÕES ACIMA DO ACASO: as alavancas de FP já conhecidas perdem sinal ou perdem acaso?

PRÉ-REGISTRO (20/10/2026), COMMITADO ANTES DE RODAR.

POR QUÊ. A meta do usuário é "FP abaixo de 0,5/mês ou próximo, com uma diferença de 1 ou 2 na detecção". Três alavancas já medidas em outras réguas chegam perto do FP pedido e foram
reprovadas por perder detecção: o blackout de 12 h (M29/M34: FP 0,861 -> 0,517, detecção mediana -1,5), o EWMA vigiado (`ewma_vigiado.py`: FP 0,861 -> 0,517, 6 de 8 composições perdem
detecção) e o veto do modo comum k=4 (EXP 51b: FP 9 -> 5, detecção intacta, NÃO adotado pelo risco). Mas os canais sp e vb ficam acesos ~60% do tempo, e já se mediu que só ~3,8 das
~6,5 detecções típicas estão acima do acaso (`ewma_come_a_partida`; no EWMA vigiado a perda acima do acaso foi só 0,27). Se a detecção que essas alavancas perdem for a que um detector sem
informação também teria, a perda REAL é perto de zero e a meta pode estar ao alcance com o que já existe. Este experimento mede isso no harness causal atual (7/8, 9 FP, 1.457 h).

CONFIGURAÇÕES (lista FECHADA; todas já medidas antes em outra régua, nenhuma nova):
  C0  controle: o detector publicado sobre o cache causal `sinais_v2_8ev.parquet`.
  C1  blackout de 12 h: só a máscara muda (o EWMA é calculado sobre o sinal cru inteiro e o vb usa `estavel`, não a máscara — lido em `preprocessar`); reset = ~máscara | partida.
  C2  blackout de 12 h + refratário de 96 h (a combinação do M34).
  C3  EWMA vigiado nos 4 canais: o EWMA recebe só instantes vigiados (`cru.where(mask)`), como em `regua_fp.py`; exige refazer o replay causal semanal com uma cópia LITERAL de
      `preprocessar` que difere em uma linha (o pacote não se edita; a cópia é provada idêntica com a chave desligada).
  C4  veto k=4 depois do refratário (EXP 51b), relatado para completar a fronteira; o veredito de NÃO adotar continua de pé seja qual for o resultado (o risco é o limite superior de
      24-33% de falhas nascendo com k=4, que nada aqui mede).

TRÊS NULOS (o que um detector com a MESMA cobertura acertaria sem informação):
  N1  UNIFORME — instantes de operação estável (RUNNING_A e T5 > 300) sorteados ao acaso; acaso esperado = 8 x cobertura, onde cobertura = fração desses instantes com alarme nas 48 h
      anteriores. É a expectativa exata do `avalia.permuta` da pesquisa, a régua que deu os "~3,8 acima do acaso".
  N2  CONDICIONADO AO RELÓGIO DE PARTIDA — o mesmo, mas cada trip só pode ser substituído por instantes com a mesma idade de campanha (horas desde o último religamento), em faixas
      [0,12), [12,24), [24,48), [48,168), >=168 h; acaso esperado = soma das coberturas da faixa de cada trip. Controla o confundidor conhecido: 5 de 8 trips vêm até 31 h após uma
      partida, e metade dos episódios nasce na borda do blackout. Sob N2, uma detecção que um alarme "X h após cada partida" também faria conta como acaso.
  N3  CIRCULAR — o alarme inteiro deslocado no tempo (30 dias até o fim menos 30, de dia em dia) contra os trips fixos (`nulo_circular.py`). Sensibilidade, não decide.

MEDIDAS: detecção de 8, início, banda, FP, FP/mês (denominador = horas vigiadas do CONTROLE, igual para todas, para que FP/mês só reflita a contagem), horas totais e fora, placebo de
eventos (+45 a +225 d); acaso esperado e S = det - acaso sob N1, N2, N3 com p; dS = S(config) - S(C0) com intervalo de 90% por bootstrap dos 8 eventos (10.000, semente 20261020) e a
faixa leave-one-event-out; tabela por evento (detectado, idade de campanha, probabilidade de acaso).

CRITÉRIOS (fixados antes):
  META-S (meta do usuário em sinal) = FP <= 6 (<= 0,54/mês) E dS >= -1,0 sob N1 E sob N2 (estimativa pontual).
  META-S PARCIAL = FP <= 6 e dS >= -1,0 sob UM só dos dois nulos.
  FORA = o resto.
EXPECTATIVAS: V1-V5 passam (~90%); C1 leva o FP a <= 6 (~55%) e perde >= 1 detecção bruta (~70%); C2 leva o FP a <= 6 (~65%); C3 leva o FP a <= 6 (~45%) e perde o 11/04 pelo
mecanismo refratário -> duração já documentado (~50%); C4 tem dS >= 0 sob N1 e N2 (~85%); o acaso do controle é MAIOR sob N2 que sob N1 (~70%); N3 fica entre N1 e N2 para o controle
(~50%); alguma de C1-C3 atinge META-S (~30%), e só META-S PARCIAL sob N2 (~20% a mais).
INTERPRETAÇÃO PREVISTA. META-S em C1, C2 ou C3: a meta está ao alcance com uma alavanca conhecida cuja perda é de acaso; vira candidata à sombra (muda a configuração congelada,
decisão do usuário), ainda HIPÓTESE com 8 eventos. Só PARCIAL sob N2: as detecções perdidas são do relógio de partida — se valem algo é pergunta operacional, vai ao usuário. FORA em
todas: a meta não se alcança sem perder detecção acima do acaso; confirma o teto de informação, e o caminho é dado novo (as 47 tags, a frota B/C).
LIMITES. 8 eventos; as 4 configurações foram escolhidas depois de vistas em outras réguas (seleção entre pontos conhecidos, não busca); as faixas do N2 foram fixadas agora, mas
com a idade de campanha dos trips já conhecida pelo atlas (13/22/31/15/9,4/150/211/677 h); o bootstrap de 8 eventos dá intervalos largos; nada se adota sem a sombra.

VERIFICAÇÕES:
  V1  C0 reimplementado (`E51B.alarme`) dá 7/8 e FP 9.
  V2  a máscara recalculada com blackout de 6 h sobre a grade completa coincide com a do cache em todos os instantes (máscara e reset).
      CORREÇÃO (20/10, 1ª execução, antes de qualquer resultado): falhou com 180 instantes = as 6 h iniciais do cache (25/10/2024 07:34-13:32), onde a 1ª janela do replay causal
      começa com a máquina ligada e o `preprocessar` conta a 1ª linha como partida. É artefato do replay, 3 meses antes do período avaliado; a V2 passa a excluir só essas 6 h.
  V3  a cópia de `preprocessar` com a chave desligada reproduz `ci.preprocessar` numa janela de 75 d (diferença máxima 0 em t, p, sp, vb; máscara igual).
  V4  todos os 8 trips têm idade de campanha definida e caem numa faixa.
  V5  para C0-C3, o alarme reimplementado coincide com `ci.prever` (a decisão de produção como caixa-preta) em todos os instantes.

Uso:  PYTHONPATH=src .venv/bin/python experiments/54_meta_acima_do_acaso.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E51B = _carrega("exp51b", "51b_concordancia_total.py")
E51 = E51B.E51
from nv.causal import carrega_registro
from nv.avalia import carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG, JANELA as JANELA_REPLAY

ROOT = E51.ROOT; INICIO8 = E51.INICIO8; GL = E51B.GL; SIN = E51.SIN
CACHE_EV = ROOT / "sinais_v2_8ev_ewmavig.parquet"
FAIXAS = (0.0, 12.0, 24.0, 48.0, 168.0, np.inf)
JAN = pd.Timedelta(hours=48); POR_DIA = 720; SEMENTE = 20261020; N_BOOT = 10_000


def preprocessar_vigiado(modelos, df: pd.DataFrame, trips=None) -> pd.DataFrame:
    """Cópia LITERAL de `ci.preprocessar`. Única diferença, sob GL['ewma_vigiado']: o EWMA recebe `cru[c].where(mask)`."""
    ci = GL["ci"]
    if isinstance(modelos, dict):
        modelos = [modelos]
    g = ci._grade(df)
    mask, estavel, partida, op = ci._mascara(g)
    det = modelos[-1]["detector"]
    t = np.full(len(g), np.nan); p = np.full(len(g), np.nan); sp = np.full(len(g), np.nan)
    meses = pd.date_range(g.index[0].normalize().replace(day=1), g.index[-1], freq="MS", tz="UTC")
    for i_m, m0 in enumerate(meses):
        m1 = meses[i_m + 1] if i_m + 1 < len(meses) else g.index[-1] + pd.Timedelta(ci.GRID)
        fatia = (g.index >= max(m0, g.index[0])) & (g.index < m1)
        if not fatia.any():
            continue
        mod = ci.vigencia(modelos, max(m0, g.index[0]))
        w = g.loc[fatia]
        t[fatia] = ci._recon_pca(mod, "temperatura", w)
        p[fatia] = ci._recon_pca(mod, "pressao", w)
        sm = mod["spread_mancal"]
        spread = (w[sm["tag_alvo"]].astype("float64") - w[sm["tags_irmaos"]].astype("float64").median(axis=1))
        sp[fatia] = np.abs((spread - sm["mediana"]) / sm["mad_robusto"]).to_numpy()
    cru = pd.DataFrame({
        "t": t, "p": p, "sp": sp,
        "vb": ci._z_vibracao(g, estavel, det["tags_vibracao"],
                             [x for x in (trips if trips is not None else det.get("trips_conhecidos", [])) if pd.Timestamp(x) <= g.index[-1]],
                             det.get("vb_exclusao_dias", 7.0)),
    }, index=g.index)
    out = pd.DataFrame(index=g.index)
    for c, hl in det["halflife"].items():
        ent = cru[c].where(mask) if GL.get("ewma_vigiado") else cru[c]
        out[c] = ent.ewm(halflife=pd.Timedelta(hl), times=g.index).mean().where(cru[c].notna())
    out["mask"] = mask
    out["reset"] = (~mask) | partida
    out["operando"] = op
    return out


def grade_completa(x):
    """O dado de entrada do replay (CSV bruto + 30 s anteriores) e a grade do detector sobre o período inteiro."""
    ci = GL["ci"]; csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv)
        raw = pd.concat([x.loc[x.index < raw.index[0], raw.columns], raw]).sort_index()
        df = ci.preparar_grade(raw)
        g = ci._grade(df)
    return df, g


def mascara(g, horas):
    ci = GL["ci"]; velho = ci.BLACKOUT_H; ci.BLACKOUT_H = horas
    try:
        return ci._mascara(g)
    finally:
        ci.BLACKOUT_H = velho


def sinais_ewma_vigiado(x):
    ci = GL["ci"]; orig = ci.preprocessar
    GL["ewma_vigiado"] = True; ci.preprocessar = preprocessar_vigiado
    try:
        return sinais_causais(pd.Timestamp("2025-01-08 07:34", tz="UTC"), cache=CACHE_EV, extra=x)
    finally:
        ci.preprocessar = orig; GL["ewma_vigiado"] = False


def cobertura_previa(a: np.ndarray) -> np.ndarray:
    """Para cada instante, houve alarme em [t - 48 h, t]? (a mesma janela de `avalia`)."""
    w = 48 * 30; cs = np.concatenate(([0], np.cumsum(a.astype(np.int64)))); i = np.arange(len(a))
    return (cs[i + 1] - cs[np.maximum(0, i - w)]) > 0


def cauda_poisson_binomial(ps, k):
    dist = np.array([1.0])
    for q in ps:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[k:].sum())


def nulos(al: pd.Series, elig: np.ndarray, faixa: np.ndarray, trips, faixa_trip):
    """Acaso esperado sob N1, N2, N3 e a probabilidade de acaso de cada trip (N1, N2)."""
    a = al.to_numpy(bool); cov = cobertura_previa(a); idx = al.index
    c1 = float(cov[elig].mean())
    cb = {b: float(cov[elig & (faixa == b)].mean()) if (elig & (faixa == b)).any() else np.nan for b in range(len(FAIXAS) - 1)}
    c2 = np.array([cb[b] for b in faixa_trip])
    fim_i = idx.searchsorted(trips, side="right") - 1; ini_i = idx.searchsorted([t - JAN for t in trips])
    obs = int(sum(a[i:j + 1].any() for i, j in zip(ini_i, fim_i)))
    n = len(a); desl = np.arange(30 * POR_DIA, n - 30 * POR_DIA, POR_DIA); d3 = []
    for s in desl:
        cs = np.concatenate(([0], np.cumsum(np.roll(a, s).astype(np.int64)))); d3.append(int(((cs[fim_i + 1] - cs[ini_i]) > 0).sum()))
    d3 = np.array(d3); nev = len(trips)
    return dict(obs=obs, e1=nev * c1, p1=float(stats.binom.sf(obs - 1, nev, c1)), e2=float(c2.sum()), p2=cauda_poisson_binomial(c2, obs),
                e3=float(d3.mean()), p3=float((d3 >= obs).mean()), c1=c1, c2=c2, cob_faixa=cb)


def main():
    ci = _ci(); GL["ci"] = ci
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc0 = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc0["mask"] = proc0["mask"].astype(bool); proc0["reset"] = proc0["reset"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in E51.SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(det=det, proc=proc0, al_c=al_c, vig_c=vig_c, EV=EV)
    ok = {}

    al0, _ = E51B.alarme(det); L0 = E51.linha(al0)
    ok["V1"] = L0["det"] == 7 and L0["fp"] == 9
    print(f"V1 — controle reimplementado: {L0['det']}/8, FP {L0['fp']} -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)

    x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet"); df, g = grade_completa(x)
    m6, est, part, op = mascara(g, 6.0); idx = proc0.index
    # CORREÇÃO DA V2 (20/10, antes de ver resultado): a 1ª janela do replay começa com a máquina ligada e o `preprocessar` conta a 1ª linha como partida -> 6 h de blackout
    # artificial em 25/10/2024, 3 meses antes do período avaliado. A comparação exclui só essas 6 h iniciais do cache.
    v2 = idx >= idx[0] + pd.Timedelta(hours=6)
    dm = int((m6.reindex(idx).fillna(False).astype(bool) != proc0["mask"])[v2].sum()); dm0 = int((m6.reindex(idx).fillna(False).astype(bool) != proc0["mask"])[~v2].sum())
    dr = int(((~m6 | part).reindex(idx).fillna(True).astype(bool) != proc0["reset"])[v2].sum())
    ok["V2"] = dm == 0 and dr == 0
    print(f"V2 — máscara 6 h recalculada x cache (fora das 6 h iniciais do replay): {dm} instantes diferentes na máscara, {dr} no reset; nas 6 h iniciais {dm0} (artefato da 1ª janela) -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)

    s = pd.Timestamp("2025-06-01", tz="UTC"); w = df.loc[max(df.index[0], s - JANELA_REPLAY):s]; conhecidos = [t for t in reg if t < s] + [t for t in carrega_registro() if t < pd.Timestamp("2025-02-27", tz="UTC")]
    conhecidos = sorted(set(conhecidos))
    with contextlib.redirect_stdout(io.StringIO()):
        a_ = ci.preprocessar(modelos, w, trips=conhecidos); GL["ewma_vigiado"] = False; b_ = preprocessar_vigiado(modelos, w, trips=conhecidos)
    dif = max(float(np.nanmax((a_[c] - b_[c]).abs().to_numpy())) if a_[c].notna().any() else 0.0 for c in SIN)
    nan_dif = int(sum((a_[c].isna() != b_[c].isna()).sum() for c in SIN)); mdif = int((a_["mask"] != b_["mask"]).sum())
    ok["V3"] = dif == 0.0 and nan_dif == 0 and mdif == 0
    print(f"V3 — cópia de preprocessar (chave desligada) x pacote, janela até {s.date()}: dif. máx {dif:.1e}, NaN diferentes {nan_dif}, máscara {mdif} -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    if not (ok["V1"] and ok["V2"] and ok["V3"]):
        print("verificação falhou — parando.", flush=True); return

    m12, _, _, _ = mascara(g, 12.0)
    proc12 = proc0.copy(); proc12["mask"] = m12.reindex(idx).fillna(False).astype(bool); proc12["reset"] = (~proc12["mask"]) | part.reindex(idx).fillna(False).astype(bool)
    print("\nreplay causal com EWMA vigiado (C3)...", flush=True)
    Sv = sinais_ewma_vigiado(x)
    procv = Sv[["t", "p", "sp", "vb", "mask", "reset"]].copy(); procv["mask"] = procv["mask"].astype(bool); procv["reset"] = procv["reset"].astype(bool)
    print(f"   {len(procv)} instantes; máscara igual à do controle em {100 * float((procv['mask'].reindex(idx) == proc0['mask']).mean()):.2f}% dos instantes", flush=True)
    det96 = {**det, "refratario_h": 96.0}
    CFG = {"C0 controle": (proc0, det, {}), "C1 blackout 12 h": (proc12, det, {}), "C2 blackout 12 h + refr. 96 h": (proc12, det96, {}),
           "C3 EWMA vigiado": (procv, det, {}), "C4 veto k=4 (depois)": (proc0, det, dict(modo="veto", kalvo=4, posicao="depois"))}
    AL = {}
    for nome, (pr, d_, kw) in CFG.items():
        GL["proc"] = pr; AL[nome] = E51B.alarme(d_, **kw)[0]
    GL["proc"] = proc0
    v5 = []
    for nome in list(CFG)[:4]:
        pr, d_, _ = CFG[nome]
        with contextlib.redirect_stdout(io.StringIO()):
            pv = ci.prever({"detector": d_}, pr)["is_anomaly"].astype(bool)
        v5.append(int((pv != AL[nome].reindex(pv.index).fillna(False)).sum()))
    ok["V5"] = all(v == 0 for v in v5)
    print(f"V5 — reimplementado x ci.prever, instantes diferentes em C0-C3: {v5} -> {'PASSA' if ok['V5'] else 'FALHA'}", flush=True)

    # ── nulos sobre o período avaliado, na grade do controle ──
    per = al_c.loc[INICIO8:fim].index
    elig = est.reindex(per).fillna(False).to_numpy(bool)
    pt = pd.Series(np.where(part.to_numpy(bool), g.index.asi8, np.nan), index=g.index).ffill()
    idade = ((pd.Series(per.asi8, index=per) - pt.reindex(per)) / 3.6e12).fillna(np.inf).to_numpy()
    faixa = np.minimum(np.searchsorted(FAIXAS, idade, side="right") - 1, len(FAIXAS) - 2)
    trips = [t for t in EV["real"] if t >= INICIO8]
    idade_trip = [float((t.value - pt.loc[:t].dropna().iloc[-1]) / 3.6e12) if pt.loc[:t].notna().any() else np.inf for t in trips]
    faixa_trip = [int(min(np.searchsorted(FAIXAS, h, side="right") - 1, len(FAIXAS) - 2)) for h in idade_trip]
    ok["V4"] = len(trips) == 8 and all(0 <= b < len(FAIXAS) - 1 for b in faixa_trip)
    print(f"V4 — idade de campanha dos trips (h): {[round(h, 1) for h in idade_trip]}; faixas {faixa_trip} -> {'PASSA' if ok['V4'] else 'FALHA'}", flush=True)
    print(f"   instantes elegíveis (operação estável) no período: {elig.sum()} de {len(per)}; por faixa: {[int((elig & (faixa == b)).sum()) for b in range(len(FAIXAS) - 1)]}", flush=True)

    R = {}
    print("\nRESULTADOS (det/8 | início | banda | FP | FP/mês | horas | fora | placebo || acaso e S = det - acaso sob N1 / N2 / N3, com p)", flush=True)
    for nome, al in AL.items():
        a = al.reindex(al_c.index).fillna(False)
        L = E51.linha(al); r = E51.mede(al); nl = nulos(a.loc[per], elig, faixa, trips, faixa_trip)
        d_ev = np.array([int(r.por_evento[str(t)]["det"]) for t in trips])
        R[nome] = dict(L=L, r=r, nl=nl, d=d_ev)
        print(f"   {nome:30s} | {L['det']}/8 | {L['inicio']} | {r.banda} | FP {L['fp']:2d} | {L['fp_mes']:.3f} | {L['horas']:5.0f} h | {L['fora']:4.0f} h | {L['placebo']:.1f} || "
              f"N1 {nl['e1']:.2f} S {L['det'] - nl['e1']:+.2f} (p {nl['p1']:.3f}) | N2 {nl['e2']:.2f} S {L['det'] - nl['e2']:+.2f} (p {nl['p2']:.3f}) | N3 {nl['e3']:.2f} S {L['det'] - nl['e3']:+.2f} (p {nl['p3']:.3f})", flush=True)
        print(f"   {'':30s}   cobertura por faixa de idade {[f'{v:.2f}' for v in nl['cob_faixa'].values()]}", flush=True)

    print("\nPOR EVENTO (idade de campanha; detectado por configuração; probabilidade de acaso N2 por configuração)", flush=True)
    for i, t in enumerate(trips):
        print(f"   {t.strftime('%d/%m/%Y')} idade {idade_trip[i]:6.1f} h faixa {faixa_trip[i]} | " + " | ".join(f"{n[:2]} {R[n]['d'][i]} ({R[n]['nl']['c2'][i]:.2f})" for n in R), flush=True)

    rng = np.random.default_rng(SEMENTE); boot = rng.integers(0, len(trips), size=(N_BOOT, len(trips)))
    b0 = R["C0 controle"]; linhas = []
    print("\nDIFERENÇA CONTRA O CONTROLE: dS = S(config) - S(C0); [IC 90% bootstrap dos eventos]; faixa LOEO", flush=True)
    for nome in list(AL)[1:]:
        x_ = R[nome]; Lx = x_["L"]; dd = x_["d"] - b0["d"]
        res = {}
        for nn, cc in (("N1", "c1"), ("N2", "c2")):
            cx = np.full(len(trips), x_["nl"]["c1"]) if cc == "c1" else x_["nl"]["c2"]; c0 = np.full(len(trips), b0["nl"]["c1"]) if cc == "c1" else b0["nl"]["c2"]
            delta = dd - (cx - c0); ds = float(delta.sum()); bt = delta[boot].sum(axis=1); lo, hi = np.percentile(bt, [5, 95])
            loeo = [float(delta.sum() - delta[i]) for i in range(len(trips))]
            res[nn] = ds
            print(f"   {nome:30s} {nn}: dS {ds:+.2f} [{lo:+.2f}; {hi:+.2f}] | LOEO {min(loeo):+.2f} a {max(loeo):+.2f} | det {Lx['det'] - b0['L']['det']:+d}, acaso {float((cx - c0).sum()):+.2f}", flush=True)
            linhas.append(dict(config=nome, nulo=nn, dS=ds, ic5=lo, ic95=hi, loeo_min=min(loeo), loeo_max=max(loeo)))
        ds3 = (Lx["det"] - x_["nl"]["e3"]) - (b0["L"]["det"] - b0["nl"]["e3"])
        fp_ok = Lx["fp"] <= 6
        cl = "META-S" if fp_ok and res["N1"] >= -1.0 and res["N2"] >= -1.0 else "META-S PARCIAL (" + ("N1" if res["N1"] >= -1.0 else "N2") + ")" if fp_ok and (res["N1"] >= -1.0 or res["N2"] >= -1.0) else "FORA"
        print(f"   {'':30s} N3 (sensibilidade): dS {ds3:+.2f} | FP {Lx['fp']} {'<= 6' if fp_ok else '> 6'} -> {cl}", flush=True)
        x_["classe"] = cl
    out = pd.DataFrame([dict(config=n, det=R[n]["L"]["det"], inicio=R[n]["L"]["inicio"], banda=R[n]["r"].banda, fp=R[n]["L"]["fp"], fp_mes=R[n]["L"]["fp_mes"], horas=R[n]["L"]["horas"],
                             fora=R[n]["L"]["fora"], placebo=R[n]["L"]["placebo"], e1=R[n]["nl"]["e1"], e2=R[n]["nl"]["e2"], e3=R[n]["nl"]["e3"], p1=R[n]["nl"]["p1"], p2=R[n]["nl"]["p2"],
                             p3=R[n]["nl"]["p3"], classe=R[n].get("classe", "controle")) for n in R])
    out.to_csv(ROOT / "exp54_resultados.csv", index=False); pd.DataFrame(linhas).to_csv(ROOT / "exp54_delta.csv", index=False)
    print(f"\nverificações: {ok}", flush=True)


if __name__ == "__main__":
    main()
