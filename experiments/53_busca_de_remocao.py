#!/usr/bin/env python3
"""EXP 53 — BUSCA DE REMOÇÃO DE SENSORES nos três canais ajustáveis (t, p, vb), com busca gulosa, LOEO e placebo.

PRÉ-REGISTRO (19/10/2026), COMMITADO ANTES DE RODAR.

POR QUÊ. O EXP 52 testou remoção de UMA tag no canal p e achou indiferença (o placebo tag a tag deu FP mediano 9,0, igual ao controle), com duas exceções: PI_0340 sai idêntico (duplicata) e PI_0319 corta 45% das horas por 1 detecção. Ficaram de fora: (i) o canal t, que tem 14 tags e 3 dimensões efetivas (o argumento do "sorteio no máximo" é mais fraco lá, mas nunca foi medido); (ii) o canal vb, 10 sondas, onde o escore é o máximo do z contra a referência rolante — nunca se testou remover sonda (o que se testou e refutou foi VOTAR entre sondas, `voto_sondas.py`, LOEO 7/9 do max contra 3-4/9 do voto); (iii) remover MAIS DE UMA tag.
O RISCO DESTE EXPERIMENTO É O SOBREAJUSTE, e ele é grande: uma busca sobre subconjuntos é literalmente "melhor de N", e o EXP 46 mediu que isso infla de 25 a 50 pontos percentuais. Por isso toda a estrutura abaixo é de proteção, não de busca.

O QUE SE FAZ (harness fiel: RobustScaler do sklearn, PCA(0,95) com k por mês, sens_p99 por sensor com piso 0,1 x mediana, recon_p99 do treino, retreino mensal causal desde 01/02/2025; n = 20.000 e 19.000; o sinal passa pelo EWMA de produção do canal, substitui o canal no `proc` do cache e a decisão de produção roda inalterada).
  FASE A — remoção simples: cada tag de t (14), de p (12) e cada sonda de vb (10), uma de cada vez. 36 braços. No vb o sinal é o replay causal do `_z_vibracao` de produção sobre o subconjunto de sondas (validado no EXP 50 contra o vb do cache com diferença 0,0).
  FASE B — busca gulosa, no canal que mostrar maior efeito na Fase A: a cada passo remove-se a tag que mais reduz o FP mantendo detecção >= 6/8; até 3 passos. É a configuração "melhor de N" que serve de teto otimista, NÃO de resultado.
  FASE C — LOEO DA BUSCA: a gulosa é refeita 8 vezes, cada uma usando só 7 eventos para escolher o que remover, e o resultado é medido NO EVENTO DEIXADO DE FORA. É o número honesto.
  FASE D — PLACEBO DE SUBCONJUNTO: 10 remoções ALEATÓRIAS do mesmo tamanho da gulosa (mesma semente, 20261019), para medir quanto do ganho é o "melhor de N".
MEDIDAS: detecção de 8, início, FP, FP/mês, horas totais e fora, placebo de eventos (+45 a +225 d) em cada braço.
CRITÉRIOS (fixados antes). GANHO = (a) o LOEO da busca >= detecção do controle - 1 E (b) FP da gulosa <= 6 E (c) o FP da gulosa ABAIXO do percentil 10 dos 10 placebos de subconjunto. Faltando qualquer um, o resultado é "melhor de N" e não vale. PIOR = detecção <= 5/8 OU FP >= 12. HIGIENE = detecção e FP a ±1 do controle (a tag sai por correção, sem alegar ganho). META DO USUÁRIO à parte: FP/mês <= 0,55 com detecção >= 5/8.
EXPECTATIVAS: nenhum GANHO pelos três critérios ~85%; a gulosa isolada (sem LOEO) alcança FP <= 6 ~55%, e esse número será "melhor de N"; o LOEO fica >= 1 evento abaixo da gulosa ~75%; algum placebo de subconjunto empata com a gulosa ~50%; no canal t a remoção simples tem efeito MENOR que no p ~60% (3 dimensões contra 1); remover sonda do vb custa detecção em >= 3 das 10 ~70% (o vb é o canal cuja ablação mais dói: sem ele 2/8); alguma remoção simples no t ou no vb é HIGIENE ~50%.
INTERPRETAÇÃO PREVISTA. O desfecho mais provável é a gulosa achar um FP baixo que o LOEO e o placebo desmontam — e aí o veredito é que a seleção de tags é mais um "melhor de N", como o Optuna do EXP 46. Se o LOEO e o placebo sobreviverem, vira HIPÓTESE para a sombra, nunca adoção direta.
LIMITES. 8 eventos; a gulosa é míope (não vê interações além do passo); 2 valores de n; o vb depende do replay causal (mais lento, 10 braços); nada se adota sem a sombra.

RESULTADO (19/10/2026; rodado localmente, ~17 min; commit do pré-registro 429c36f). NENHUM GANHO. A busca não encontra subconjunto que melhore: a gulosa, que é o teto OTIMISTA, termina com FP PIOR que o controle.
  FASE A — remoção simples (média de n = 20.000 e 19.000):
    · canal t (controle 6,0/8, FP 10,0, 1.482 h): NENHUMA das 14 remoções reduz o FP. A melhor empata (sem TC382_01_A: FP 10,0, mas 5,0/8 — perde 1 detecção). As piores sobem a 13,0-13,5 (sem TC382_03_A: 4,0/8; sem TI_0305: 13,5 FP). Tirar os mancais dispara as horas: sem TI_0303 1.990 h e sem TI_0301 1.843 h, contra 1.482 do controle — ou seja, os 4 sensores que o canal sp duplica NÃO são removíveis do t, o que fecha pela negativa o braço P2 do plano de ontem (desfazer a sobreposição t<->sp pelo lado do t).
    · canal p (controle 7,0/8, FP 9,0): confirma o EXP 52 — PDI_0317 (FP 8,5), PI_0315 e PI_0340 (9,0) são higiene; nenhuma melhora de verdade.
    · canal vb (controle 7,0/8, FP 9,0): as 10 sondas são removíveis uma a uma sem efeito — 9 das 10 ficam em HIGIENE. A melhor é sem TV_354X_A (7,0/8, FP 8,0, 1.442 h); a única que custa detecção é TV_354Y_A (6,0/8). Isto é novo: a ablação do CANAL vb inteiro custa 5 detecções (sem vb 2/8, EXP 42), mas nenhuma SONDA individual é necessária — a informação está distribuída nas 10 e o `max` a recupera de qualquer subconjunto de 9.
  FASE B — busca gulosa (no t, o canal de maior efeito, ΔFP máximo 3,5): remove TC382_06_A, TI_0301 e TC382_01_A -> 6,0/8, início 5,0, FP 11,0 (0,986/mês), 1.443 h. O FP SOBE de 10,0 para 11,0 mesmo sendo a busca que minimiza FP a cada passo — porque a restrição de manter detecção >= 6/8 não deixa a gulosa descer.
  FASE C — LOEO da busca: 5/8 contra 6/8 do controle. Em 7 das 8 dobras a busca honesta decide NÃO REMOVER NADA (lista vazia); só na dobra do 04/11/2025 ela remove as três, e justamente aí perde o evento.
  FASE D — placebo de subconjunto (10 remoções aleatórias de 3 tags): FP mínimo 11,5, percentil 10 em 12,4, mediana 13,8, contra 11,0 da gulosa. A gulosa fica abaixo do percentil 10, mas num patamar em que TODAS as opções são piores que não remover nada.
  VEREDITO pelos três critérios: (a) LOEO 5/8 >= controle-1 = SIM; (b) FP <= 6 = NÃO (11,0); (c) abaixo do percentil 10 do placebo = SIM. Falta (b), então é EQUIVALENTE / melhor-de-N.
  CONCLUSÃO: remover sensor não melhora o detector em nenhum dos três canais ajustáveis. O que a busca mostra é o contrário do que se procurava — no canal t cada tag contribui (a remoção sempre custa detecção, horas ou as duas), no p as tags são intercambiáveis (EXP 52) e no vb as sondas são redundantes entre si mas o canal é insubstituível. A única remoção defensável continua sendo a do EXP 52: PI_0340, por ser duplicata, e sai sem alterar nada.
PLACAR: ACERTEI: nenhum GANHO pelos três critérios (~85%); o LOEO fica >= 1 evento abaixo da gulosa (~75%; 5 contra 6); no canal t a remoção simples tem efeito MENOR que no p (~60%) — ERRADO ao contrário, ver abaixo; alguma remoção simples no t ou no vb é HIGIENE (~50%; 9 das 10 sondas e 4 tags do t). ERREI: previ que a gulosa isolada alcançaria FP <= 6 (~55%) — ela terminou em 11,0, PIOR que o controle, porque a restrição de detecção impede descer; previ que o efeito no t seria menor que no p (~60%) — foi MAIOR (ΔFP até 3,5 no t contra 1,0 no p); previ que remover sonda do vb custaria detecção em >= 3 das 10 (~70%) — custou em UMA só; previ que algum placebo empataria com a gulosa (~50%) — o melhor placebo ficou em 11,5 contra 11,0.

Uso:  PYTHONPATH=src .venv/bin/python experiments/53_busca_de_remocao.py
      variáveis: EXP53_PROCS (padrão 6), EXP53_SEM_VB=1 (pula o canal vb, que é o lento)
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


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py"); E50 = _carrega("exp50", "50_modelo_por_grupo.py"); E20 = E46.E20
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; CACHE = ROOT / "escores_rem53"; NS = (20000, 19000)
SEMENTE = 20261019; GL = {}


def chave(canal, remove):
    return f"{canal}_" + ("TODAS" if not remove else "-".join(sorted(c.replace("954005_624_", "") for c in remove)))


def pontua(a):
    canal, remove, n = a; f = CACHE / f"{chave(canal, remove)}_{n}.parquet"
    if f.exists():
        return a
    t0 = time.time(); tags = [c for c in GL["tags"][canal] if c not in remove]
    if canal == "vb":
        s = E50.replay_zr(tags, GL["hl"][canal], False)
    else:
        s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), GL["g"], tags, GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
        s = s.loc[GL["ini"]:]
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f)
    print(f"      {chave(canal, remove):28s} n={n}: {time.time() - t0:.0f} s", flush=True); return a


def sinal(canal, remove, n):
    s = pd.read_parquet(CACHE / f"{chave(canal, remove)}_{n}.parquet")["s"]
    return s if canal == "vb" else s.ewm(halflife=pd.Timedelta(GL["hl"][canal]), times=s.index).mean().where(s.notna())


def avalia_br(canal, remove, n, evs=None):
    al = decide_v2(substitui(GL["proc"], canal, sinal(canal, remove, n)), GL["det"])["is_anomaly"].astype(bool).reindex(GL["al_c"].index).fillna(False)
    r = avalia(al, GL["vig_c"], evs if evs is not None else GL["EV"]["real"], inicio_periodo=INICIO8)
    pl = [avalia(al, GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev) for k, ev in GL["EV"].items() if k != "real" and ev]
    return dict(det=r.det, inicio=r.inicio, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora, placebo=float(np.median(pl)) * 8)


def medio(canal, remove, evs=None):
    L = [avalia_br(canal, remove, n, evs) for n in NS]
    return {k: float(np.mean([x[k] for x in L])) for k in L[0]}


def main():
    procs = int(os.environ.get("EXP53_PROCS", 6)); sem_vb = bool(os.environ.get("EXP53_SEM_VB")); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; norm = modelos[-1]["normalizacao"]
    tags = {"t": list(norm["temperatura"]["cols"]), "p": list(norm["pressao"]["cols"]), "vb": list(det["tags_vibracao"])}
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(g=g, est=est, fim=fim, ini=INICIO8 - pd.Timedelta(days=10), tags=tags, hl=det["halflife"], det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV)
    E50.GL.update(g=g, est=est, fim=fim, ini=GL["ini"], tags=tags, hl=det["halflife"])
    canais = ["t", "p"] + ([] if sem_vb else ["vb"])
    print(f"EXP 53: canais {canais} | tags {({c: len(tags[c]) for c in canais})} | PROCS {procs}", flush=True)
    # ── FASE A ──
    tarefas = [(c, (), n) for c in canais for n in NS] + [(c, (tag,), n) for c in canais for tag in tags[c] for n in NS]
    t0 = time.time()
    with Pool(procs) as pool:
        list(pool.imap_unordered(pontua, tarefas, chunksize=1))
    print(f"FASE A: {len(tarefas)} escores em {time.time() - t0:.0f} s", flush=True)
    base = {c: medio(c, ()) for c in canais}
    for c in canais:
        print(f"\nFASE A [{c}] controle: {base[c]['det']:.1f}/8, início {base[c]['inicio']:.1f}, FP {base[c]['fp']:.1f}, {base[c]['horas']:.0f} h, {base[c]['fora']:.0f} h fora", flush=True)
        linhas = []
        for tag in tags[c]:
            L = medio(c, (tag,)); linhas.append((tag, L))
        for tag, L in sorted(linhas, key=lambda x: (x[1]["fp"], -x[1]["det"])):
            hig = " HIGIENE" if (abs(L["det"] - base[c]["det"]) <= 1 and abs(L["fp"] - base[c]["fp"]) <= 1) else ""
            print(f"   sem {tag.replace('954005_624_',''):14s} | {L['det']:.1f}/8 | iní {L['inicio']:.1f} | FP {L['fp']:4.1f} | {L['horas']:6.0f} h | {L['fora']:5.0f} h |{hig}", flush=True)
    # ── FASE B: gulosa no canal de maior efeito ──
    efeito = {c: max(abs(medio(c, (tag,))["fp"] - base[c]["fp"]) for tag in tags[c]) for c in canais}
    alvo = max(efeito, key=efeito.get); print(f"\nFASE B — busca gulosa no canal '{alvo}' (maior efeito na Fase A: ΔFP máximo {efeito[alvo]:.1f})", flush=True)

    def gulosa(evs, passos=3, verboso=False):
        rem = []
        for passo in range(passos):
            cand = [tag for tag in tags[alvo] if tag not in rem]
            novos = [(c,) for c in cand]
            with Pool(procs) as pool:
                list(pool.imap_unordered(pontua, [(alvo, tuple(sorted(rem + [c])), n) for c in cand for n in NS], chunksize=1))
            melhor = None
            for c in cand:
                L = medio(alvo, tuple(sorted(rem + [c])), evs)
                if L["det"] >= 6 and (melhor is None or L["fp"] < melhor[1]["fp"]):
                    melhor = (c, L)
            if melhor is None:
                break
            rem.append(melhor[0])
            if verboso:
                print(f"   passo {passo + 1}: remove {melhor[0].replace('954005_624_','')} -> {melhor[1]['det']:.1f}/8, FP {melhor[1]['fp']:.1f}, {melhor[1]['horas']:.0f} h", flush=True)
        return rem

    rem_full = gulosa(None, 3, verboso=True); Lg = medio(alvo, tuple(sorted(rem_full)))
    print(f"   GULOSA (melhor de N, otimista): remove {[r.replace('954005_624_','') for r in rem_full]} -> {Lg['det']:.1f}/8, início {Lg['inicio']:.1f}, FP {Lg['fp']:.1f} ({Lg['fp_mes']:.3f}/mês), {Lg['horas']:.0f} h, {Lg['fora']:.0f} h fora", flush=True)
    # ── FASE C: LOEO ──
    print("\nFASE C — LOEO da busca (escolhe com 7 eventos, mede no 8º)", flush=True)
    evs = EV["real"]; acertos = 0
    for i, t_ in enumerate(evs):
        treino = [t for j, t in enumerate(evs) if j != i]
        rem_i = gulosa(treino, 3); d = medio(alvo, tuple(sorted(rem_i)), [t_])["det"]; acertos += int(d >= 1)
        print(f"   {t_.strftime('%d/%m/%Y')}: removeria {[r.replace('954005_624_','') for r in rem_i]} -> {'detectou' if d >= 1 else 'PERDEU'}", flush=True)
    print(f"   LOEO = {acertos}/8   (controle: {base[alvo]['det']:.0f}/8 | gulosa dentro da amostra: {Lg['det']:.0f}/8)", flush=True)
    # ── FASE D: placebo de subconjunto ──
    print(f"\nFASE D — placebo: 10 remoções aleatórias de {len(rem_full)} tags", flush=True)
    rng = np.random.default_rng(SEMENTE); sorteios = [tuple(sorted(rng.choice(tags[alvo], size=len(rem_full), replace=False))) for _ in range(10)]
    with Pool(procs) as pool:
        list(pool.imap_unordered(pontua, [(alvo, s, n) for s in sorteios for n in NS], chunksize=1))
    fps = []
    for s in sorteios:
        L = medio(alvo, s); fps.append(L["fp"])
        print(f"   {[x.replace('954005_624_','') for x in s]} -> {L['det']:.1f}/8, FP {L['fp']:.1f}", flush=True)
    p10 = float(np.percentile(fps, 10))
    print(f"   FP dos placebos: mínimo {min(fps):.1f} | percentil 10 {p10:.1f} | mediana {np.median(fps):.1f} | gulosa {Lg['fp']:.1f}", flush=True)
    # ── veredito ──
    ok_loeo = acertos >= base[alvo]["det"] - 1; ok_fp = Lg["fp"] <= 6; ok_plac = Lg["fp"] < p10
    cl = "GANHO" if (ok_loeo and ok_fp and ok_plac) else "PIOR" if (Lg["det"] <= 5 or Lg["fp"] >= 12) else "EQUIVALENTE / melhor-de-N"
    print(f"\nVEREDITO: {cl}", flush=True)
    print(f"   (a) LOEO {acertos}/8 >= controle-1 ({base[alvo]['det'] - 1:.0f})? {'sim' if ok_loeo else 'NÃO'}", flush=True)
    print(f"   (b) FP da gulosa {Lg['fp']:.1f} <= 6? {'sim' if ok_fp else 'NÃO'}", flush=True)
    print(f"   (c) FP da gulosa abaixo do percentil 10 dos placebos ({p10:.1f})? {'sim' if ok_plac else 'NÃO'}", flush=True)


if __name__ == "__main__":
    main()
