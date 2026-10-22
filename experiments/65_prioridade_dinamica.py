#!/usr/bin/env python3
"""EXP 65 — PRIORIDADE DINÂMICA PELA CARGA: o alarme nascido em carga baixa começa como atenção e SOBE para alarme se a carga subir enquanto ele está aceso.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp65.py`), branch exp/modos-de-falha.

POR QUÊ (EXP 61-62 e a checagem carga x fantasma de 22/10). A carga na notificação (L = percentil do T5 mediano em [nascimento, +2 h] contra os 30 d anteriores de
operação estável) é o primeiro discriminante FP x acerto que replica; a regra estática "L < 0,5 vira atenção" dá 7/8 FP 4 / 7/8 FP 6 / 7/8 FP 6 / 5/8 FP 6 nas quatro bases,
mas é FRÁGIL (vizinhança 5/8) e, nas bases G168 e B800, perde o 17/03/2025 — o episódio nasce em carga 0,24 e dura 197 h. A carga é uma condição que MUDA durante o
episódio; a regra estática a lê uma vez só. Se o trip vem quando a máquina está carregada (7/8 trips acima da mediana do T5), um precursor nascido em carga baixa deveria
ganhar prioridade quando a carga sobe — e um FP de carga baixa que nunca vê carga alta continua atenção.
ATENÇÃO, DECLARADO: a hipótese foi motivada por UM evento (o 17/03 nas bases G168/B800). Não olhei a trajetória de carga de nenhum episódio antes deste pré-registro.

REGRA (fixada antes; pós-processa a série de alarme já produzida — o detector não muda, e o episódio rebaixado continua armando o refratário, como no EXP 62):
  Para cada episódio (a, b) de `ci._episodios`: L0 = L na notificação (EXP 62). Se L0 >= θ (ou indefinido): alarme desde a, como hoje. Se L0 < θ: ATENÇÃO desde a; a cada
  instante t >= a + 2 h, L(t) = percentil da mediana do T5 estável em (t − 2 h, t] contra a MESMA referência (os 30 d antes de a). No primeiro t com L(t) >= θ o episódio
  é PROMOVIDO: alarme de t até b (o início do alarme é t, não a). Sem promoção: atenção o episódio inteiro (= regra estática). Uma vez promovido, não volta.
  Principal: θ = 0,5, T5, referência 30 d, janela 2 h (os parâmetros do EXP 62, nenhum novo). Vizinhança: os mesmos 8 vizinhos do EXP 62.
BASES: atual, G168, B800 (ledgers do controle) e atual com retreino n = 19.000 — as do EXP 62.
MEDIDAS: por base, original x estática x dinâmica (detecção, início, banda, FP, FP/mês, horas, fora); episódios promovidos (L0, atraso até a promoção, TP/FP); vizinhança;
placebo INFORMATIVO: o mesmo número de episódios de baixa prioridade sorteado e promovido em instante uniforme (500 sorteios); com k episódios de baixa prioridade por base o p
mínimo atingível é ~1/k, por isso o placebo não entra no critério.
CRITÉRIOS (fixados antes, nesta ordem):
  IGUAL = em todas as bases detecção e FP iguais aos da estática.
  MELHOR = (i) G168 e B800 voltam a 8/8 (recuperam o 17/03); E (ii) em todas as bases FP <= estática + 1, detecção, início e banda >= os da estática; E (iii) vizinhança: >= 6
           dos 8 vizinhos com perda de detecção <= 1 contra o original em todas as bases e FP médio <= 7 (o critério (b) do EXP 62, que a estática não passou: 5/8).
  PIOR = em alguma base FP > estática + 2 ou detecção < estática.
  PARCIAL no resto.
EXPECTATIVAS: V1-V4 passam (~90%); o 17/03 é promovido em G168 e B800 (~65%); FP <= estática + 1 nas 4 bases (~35%: os FP longos de carga baixa — 52, 34, 20, 17 e 14 h no
controle — também podem ver carga alta); vizinhança >= 6/8 (~30%); classificação: MELHOR ~15%, PARCIAL ~35%, PIOR ~40%, IGUAL ~10%.
INTERPRETAÇÃO PREVISTA: MELHOR -> a prioridade dinâmica substitui a estática como hipótese principal na sombra. PARCIAL -> as duas vão para a sombra, nenhuma é adotada.
PIOR -> a carga durante o episódio não discrimina; fica só a carga na notificação, e o 17/03 nas variantes é o preço conhecido dela. IGUAL -> sem efeito.
LIMITES: 8 eventos; as 4 bases dividem o mesmo dado; motivada por um evento; θ herdado do EXP 62 (não reescolhido).
VERIFICAÇÕES: V1 sem regra, cada base reproduz a referência (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12; n=19.000 5/8 FP 11); V2 sem promoção, a regra reproduz a estática do
EXP 62 (FP 4, 6, 6, 6); V3 com θ = 0 nada é rebaixado e a série volta a ser a original; V4 causalidade: para todo episódio promovido, recalcular a promoção com o T5 cortado no
instante da promoção devolve o mesmo instante.

RESULTADO (22/10/2026; worker do ClearML, tarefa f911227990fc475c8c9243dfeeb00c15, commit d77de7d sem diff; pré-registro a08a342; a 1ª tarefa, 4b57b959, foi abortada
por terceiro aos 73 s). DECLARADO: antes desta corrida vi números da dinâmica no histórico pelo `prioridade_sombra.py --verifica` (o pré-registro já estava commitado).
V1-V4 PASSAM. PARCIAL.
    base        original                  estática                  dinâmica
    atual       7/8 · FP  9 · 1.457 h     7/8 · FP 4 (0,359)        7/8 · FP 5 (0,448) · 1.336 h
    G168        8/8 · FP 14 · 1.995 h     7/8 · FP 6 (0,538)        8/8 · FP 8 (0,717) · 1.406 h
    B800        8/8 · FP 12 · 1.712 h     7/8 · FP 6 (0,538)        8/8 · FP 7 (0,628) · 1.356 h
    n=19000     5/8 · FP 11 · 1.505 h     5/8 · FP 6 (0,538)        5/8 · FP 7 (0,628) · 1.377 h
  (i) PASSA: o 17/03 é promovido em G168 e B800 (episódio de 09/03/2025, 205 h, promovido 3,2 h depois de nascer). (ii) FALHA: G168 FP 8 contra 6 da estática (+2); nas
  outras +1. O FP que volta em todas as bases é o 09/01/2026 (34-78 h, promovido 17 h depois); no G168 também o 02/01/2026 (164 h, promovido 159 h depois).
  (iii) vizinhança 5/8 — o MESMO número da estática, mas a falha mudou de natureza: na estática os vizinhos que falham PERDEM DETECÇÃO (θ 0,6, PDI_0317, ref 60 d); na dinâmica
  os que falham SOBEM O FP (θ 0,4: FP médio 8,2; ref 60 d: 8,8; janela 1 h: 7,2), e nenhum vizinho perde mais de 1 detecção.
  Placebo informativo: p 1,000 / 0,316 / 0,288 / 1,000 — em atual e n=19.000 a única promoção é um FP; o ganho da dinâmica é só o 17/03 nas variantes.
PLACAR: ACERTEI: V1-V4 (~90%); o 17/03 é promovido em G168 e B800 (~65%); FP <= estática + 1 nas 4 bases NÃO (dei ~35%); vizinhança >= 6/8 NÃO (dei ~30%); PARCIAL (~35%).
VEREDITO (como pré-registrado): PARCIAL -> as duas vão para a sombra, nenhuma é adotada (já registradas como H-CARGA e H-CARGA-DIN em `scripts/prioridade_sombra.py`, d77de7d).
  Leitura, sem mudar o veredito: a dinâmica troca o modo de falha da regra — de "perder um trip" para "um FP a mais" —, o que para a operação é o lado mais seguro do erro.

Uso local (encanamento, sem tabelas):  EXP65_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/65_prioridade_dinamica.py
No worker:  PYTHONPATH=src python scripts/worker_exp65.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E60 = _carrega("exp60", "60_borda_adiada.py"); E62 = _carrega("exp62", "62_regra_de_carga.py")
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

TESTE = bool(os.environ.get("EXP65_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); P = "954005_624_"
PRINC = E62.PRINC; VIZ = E62.VIZ; N_PERM = 40 if TESTE else 500; SEMENTE = 20261022; DOIS_MIN = pd.Timedelta(minutes=2)
JAN = pd.Timedelta(hours=48)


def promocao(serie, a, b, cfg):
    """Primeiro instante t >= a + janela com L(t) >= θ, ou None. L(t) usa só (t − janela, t] e a referência antes de a."""
    ref = np.sort(serie.loc[a - pd.Timedelta(days=cfg["ref_d"]):a - DOIS_MIN].dropna().to_numpy())
    if len(ref) < 30:
        return None
    jan = pd.Timedelta(hours=cfg["notif_h"]); s = serie.loc[a:b]
    med = s.rolling(jan, min_periods=1).median().loc[a + jan:]
    v = med.to_numpy(); fin = np.isfinite(v)
    L = np.searchsorted(ref, np.where(fin, v, 0.0), side="left") / len(ref)
    k = np.flatnonzero(fin & (L >= cfg["theta"]))
    return med.index[k[0]] if len(k) else None


def regra(al, eps, Ls, serie, cfg, dinamica=True):
    a_ = al.copy(); prom = {}
    for (ini, fim), L in zip(eps, Ls):
        if np.isfinite(L) and L < cfg["theta"]:
            t = promocao(serie, ini, fim, cfg) if dinamica else None
            if t is None:
                a_.loc[ini:fim] = False
            else:
                a_.loc[ini:t - DOIS_MIN] = False; prom[(ini, fim)] = t
    return a_, prom


def rotulo(a, b, trips):
    return "TP" if any(a <= t and b >= t - JAN for t in trips) else "FP"


def main():
    t0 = time.time(); ci = _ci(); E60.GL["ci"] = ci; ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    todos = sorted(carrega_registro()); E60.GL["trips"] = [t for t in todos if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    IND = {"T5": g["T5_AVG_A"].where(est), "TC382": g[[f"TC382_0{i}_A" for i in range(1, 7)]].mean(axis=1).where(est), "PDI_0317": g[P + "PDI_0317"].where(est)}
    B = {v: carrega_controle(v) for v in ("atual", "G168", "B800")}
    E60.GL["idx"] = B["atual"][0].index
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    B["atual n=19000"] = (E60.base_n(ci, det, proc, 19000).reindex(E60.GL["idx"]).fillna(False), B["atual"][1])
    base0 = {v: E60.mede(*B[v]) for v in B}
    for v, (dr, fr) in dict(E60.REF, **{"atual n=19000": (5, 11)}).items():
        ok[f"V1_{v}"] = (base0[v]["det"], base0[v]["fp"]) == (dr, fr)
    print(f"V1 — bases sem regra: {[(v, base0[v]['det'], base0[v]['fp']) for v in B]} -> {'PASSA' if all(ok[k] for k in ok if k.startswith('V1')) else 'FALHA'}", flush=True)
    EPS = {v: [(a, b) for a, b in ci._episodios(al.loc[INICIO8:])] for v, (al, vig) in B.items()}

    def roda(cfg, dinamica=True):
        out = {}
        for v, (al, vig) in B.items():
            Ls = [E62.L_de(IND[cfg["ind"]], a, cfg["ref_d"], cfg["notif_h"])[0] for a, b in EPS[v]]
            a_, prom = regra(al, EPS[v], Ls, IND[cfg["ind"]], cfg, dinamica)
            out[v] = (E60.mede(a_, vig), prom, Ls)
        return out
    EST = roda(PRINC, dinamica=False); DIN = roda(PRINC)
    fps = [EST[v][0]["fp"] for v in B]; ok["V2"] = fps == [4, 6, 6, 6]
    print(f"V2 — sem promoção reproduz a estática do EXP 62 (FP 4, 6, 6, 6): {fps} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    Z = roda(dict(PRINC, theta=0.0)); ok["V3"] = all((Z[v][0]["det"], Z[v][0]["fp"], round(Z[v][0]["horas"], 3)) == (base0[v]["det"], base0[v]["fp"], round(base0[v]["horas"], 3)) for v in B)
    print(f"V3 — θ = 0 devolve a original -> {'PASSA' if ok['V3'] else 'FALHA'}", flush=True)
    s = IND["T5"]; v4 = [promocao(s.loc[:t], a, b, PRINC) == t for v in B for (a, b), t in DIN[v][1].items()]
    ok["V4"] = all(v4); print(f"V4 — promoção causal (T5 cortado na promoção dá o mesmo instante) em {sum(v4)}/{len(v4)} -> {'PASSA' if ok['V4'] else 'FALHA'}", flush=True)
    RV = [(cfg, roda(cfg)) for cfg in VIZ]
    rng = np.random.default_rng(SEMENTE); PERM = {}
    for v, (al, vig) in B.items():
        m_r, prom, Ls = DIN[v]; k = len(prom)
        baixos = [(a, b) for (a, b), L in zip(EPS[v], Ls) if np.isfinite(L) and L < PRINC["theta"] and (b - a) > pd.Timedelta(hours=PRINC["notif_h"])]
        if k == 0 or not baixos:
            PERM[v] = (np.nan, len(baixos)); continue
        a_est, _ = regra(al, EPS[v], Ls, s, PRINC, dinamica=False); cont = 0
        for _ in range(N_PERM):
            ap = a_est.copy()
            for j in rng.choice(len(baixos), size=min(k, len(baixos)), replace=False):
                a, b = baixos[j]; lo = a + pd.Timedelta(hours=PRINC["notif_h"]); t = lo + (b - lo) * rng.random(); t = t.floor("2min")
                ap.loc[t:b] = al.loc[t:b]
            mp = E60.mede(ap, vig); cont += (mp["det"] >= m_r["det"]) and (mp["fp"] <= m_r["fp"])
        PERM[v] = (cont / N_PERM, len(baixos))
    GL = dict(B=B, base0=base0, EST=EST, DIN=DIN, RV=RV, PERM=PERM, todos=todos)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(GL, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(GL, ok, t0, escreve=True)


def relata(GL, ok, t0, escreve):
    B, base0, EST, DIN, RV, PERM, todos = GL["B"], GL["base0"], GL["EST"], GL["DIN"], GL["RV"], GL["PERM"], GL["todos"]
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    f = lambda m: f"{m['det']}/8 ini {m['inicio']} ban {m['banda']} FP {m['fp']:2d} ({m['fp_mes']:.3f}) {m['horas']:5.0f} h"
    print("\nPRINCIPAL (θ 0,5 | T5 | ref 30 d | janela 2 h)          original | estática | dinâmica", flush=True)
    for v in B:
        print(f"   {v:14s} {f(base0[v])} | {f(EST[v][0])} | {f(DIN[v][0])}", flush=True)
        for (a, b), t in sorted(DIN[v][1].items()):
            print(f"      promovido: {loc(a)} ({(b - a).total_seconds() / 3600:.0f} h, {rotulo(a, b, todos)}) -> alarme desde {loc(t)} (+{(t - a).total_seconds() / 3600:.1f} h)", flush=True)
        p, kb = PERM[v]
        print(f"      placebo informativo: p = {p:.3f} ({kb} episódios de baixa prioridade com mais de 2 h)" if np.isfinite(p) else f"      placebo: sem promoção ({kb} de baixa prioridade)", flush=True)
    nome = lambda c: f"θ {c['theta']} | {c['ind']} | ref {c['ref_d']} d | janela {c['notif_h']} h"
    print("\nVIZINHANÇA (det/8 e FP: " + " | ".join(B) + ")", flush=True)
    viz_ok = 0
    for cfg, r in RV:
        perdas = [base0[v]["det"] - r[v][0]["det"] for v in B]; fpm = float(np.mean([r[v][0]["fp"] for v in B])); bom = max(perdas) <= 1 and fpm <= 7; viz_ok += bom
        print(f"   {nome(cfg):40s} | " + " | ".join(f"{r[v][0]['det']}/8 FP {r[v][0]['fp']:2d}" for v in B) + f" | FP médio {fpm:.1f} -> {'ok' if bom else 'não'}", flush=True)
    igual = all((DIN[v][0]["det"], DIN[v][0]["fp"]) == (EST[v][0]["det"], EST[v][0]["fp"]) for v in B)
    i_ = DIN["G168"][0]["det"] == 8 and DIN["B800"][0]["det"] == 8
    ii = all(DIN[v][0]["fp"] <= EST[v][0]["fp"] + 1 and all(DIN[v][0][k] >= EST[v][0][k] for k in ("det", "inicio", "banda")) for v in B)
    iii = viz_ok >= 6
    pior = any(DIN[v][0]["fp"] > EST[v][0]["fp"] + 2 or DIN[v][0]["det"] < EST[v][0]["det"] for v in B)
    cl = "IGUAL" if igual else "MELHOR" if (i_ and ii and iii) else "PIOR" if pior else "PARCIAL"
    print(f"\nCLASSIFICAÇÃO: (i) 17/03 recuperado em G168 e B800: {i_} | (ii) FP <= estática + 1 e nada cai: {ii} | (iii) vizinhança {viz_ok}/8 | -> {cl}", flush=True)
    if escreve:
        linhas = [dict(cfg="principal", regra=r, base=v, **{k: m[k] for k in ("det", "inicio", "banda", "fp", "fp_mes", "horas", "fora")})
                  for v in B for r, m in (("original", base0[v]), ("estatica", EST[v][0]), ("dinamica", DIN[v][0]))]
        linhas += [dict(cfg=nome(c), regra="dinamica", base=v, **{k: r[v][0][k] for k in ("det", "inicio", "banda", "fp", "fp_mes", "horas", "fora")}) for c, r in RV for v in B]
        pd.DataFrame(linhas).to_csv(ROOT / "exp65_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
