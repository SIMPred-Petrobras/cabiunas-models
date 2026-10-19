#!/usr/bin/env python3
"""EXP 52 — REMOÇÃO DE SENSORES: higiene (duplicata e tag de outra unidade) contra placebo de remoção.

PRÉ-REGISTRO (19/10/2026), COMMITADO ANTES DE RODAR.

POR QUÊ. A pergunta do usuário ("retirar alguns sensores da análise") tem dois precedentes que NÃO a respondem:
  · F1 (`f1_exclusao.py`, 01/10) excluiu sensor SINALIZADO PELO MONITOR, de forma dinâmica: Δcarga −31,3 [−61,1; −8,9], FP −0,11, mas a composição 4 perdeu o 09/12/2025 → REPROVADO. É exclusão dinâmica, não fixa.
  · C3 (`escala_teto_ar.py`, 02/10) removeu o PI_5134001: detecção e banda idênticas nas 8 composições, carga −0,4 → "seguro, não ganho". Testou UMA tag, no harness das 8 composições.
  · A memória do F1 deixou registrada uma hipótese nunca testada: "tirar o PI_0319 do canal p de vez é hipótese nova — só com pré-registro próprio".
MEDIDO AGORA (instantes estáveis desde 01/02/2025, 252.916 pontos), o que justifica a remoção A PRIORI, sem olhar os eventos:
  · PI_0340 é DUPLICATA do PI_0339: r = +0,99878 e |diferença| MÁXIMA de 0,056 em 100% do tempo (mediana 0,0104). Duas tags, uma informação.
  · PI_5134001 pertence à U-5134, OUTRA UNIDADE — não é da turbina (já era sabido; aqui entra junto).
  · T5_AVG_A foi cogitada e DESCARTADA pelos dados: r = 0,999949 com a média dos 6 TC382, mas |diferença| máxima de 48,4 °C (p99 1,28) — diverge justamente no transiente, onde pode carregar informação. Fica.
  · DIMENSIONALIDADE EFETIVA (RobustScaler + PCA): canal t = 3 componentes para 95% de 14 tags (1º = 84%); canal p = 1 COMPONENTE para 95% de 12 tags (1º = 99%). No regime univariado do p o escore é o MÁXIMO de desvios por tag, e cada tag a mais é um sorteio extra no máximo — mais chance de acender sem acrescentar informação multivariada. É por isso que a remoção pode valer mais no p do que no t.
O QUE SE FAZ. Canal p reajustado SEM as tags removidas, pelo harness fiel (`walk_forward(com_sk(PCAfiel), …)`: RobustScaler do sklearn, PCA(0,95) com k variável por mês, sens_p99 por sensor com piso 0,1 × mediana, recon_p99 do treino, retreino mensal causal a partir de 01/02/2025), n = 20.000 e 19.000. O sinal passa pelo EWMA de produção (1 h), substitui o canal p no `proc` do cache e a decisão de produção (`ci.prever` via `decide_v2`) roda inalterada — mesmos k, mesmo voto, mesmo CUSUM, mesmo refratário. O pacote de produção NÃO é editado.
  BRAÇOS (fixados aqui): H1 sem PI_0340 (duplicata) · H2 sem PI_5134001 (outra unidade) · H3 sem as duas · H4 sem PI_0319 (a hipótese registrada do F1) · H5 sem PI_0340 e PI_0319.
  PLACEBO DE REMOÇÃO (a peça que faltou nos anteriores): as 5 tags restantes do canal p removidas UMA A UMA, como braços próprios. Se remover QUALQUER tag melhorar na mesma medida, o efeito não é higiene — é "menos sorteios no máximo", ou seja, mexer no limiar por outro caminho, e aí nenhuma remoção se justifica como correção.
  CONTROLE: o canal p com as 12 tags, reajustado pelo MESMO harness (não o cache), para que a comparação seja entre iguais.
MEDIDAS: detecção de 8, início, FP, FP/mês, horas totais e fora das janelas, placebo de eventos (+45 a +225 d), e o k do PCA por mês (para ver se a remoção muda a dimensionalidade do ajuste).
VERIFICAÇÃO. V1 — o canal p com as 12 tags reajustado reproduz o `escores_sub34/p_A_{n}` com diferença máxima <= 1e-6 (o mesmo ajuste já validado no EXP 19). V2 — o `prever` com esse p reajustado devolve o controle (7/8, FP 9 ± 1).
CRITÉRIOS (fixados antes). GANHO = FP <= 6 E detecção >= 6/8 E horas <= 1.457 E o braço de higiene (H1, H2 ou H3) melhor que a MEDIANA dos 5 placebos de remoção em FP. PIOR = detecção <= 5/8 OU FP >= 12. EQUIVALENTE no resto. HIGIENE (veredito próprio, sem precisar de ganho) = detecção e FP dentro de ±1 do controle: a tag sai por correção, não por desempenho.
EXPECTATIVAS: nenhum GANHO ~75%; H1 (duplicata) EQUIVALENTE/HIGIENE ~70%; H3 melhor que a mediana dos placebos ~40%; algum placebo de remoção com FP <= o de H3 ~55% (seria o sinal de que o efeito é do número de tags, não da tag); H4 (PI_0319) muda mais que os outros, porque concentra 97,9% da variância escalada ~65%; a remoção muda o k mensal do PCA em >= 1 em algum mês ~80%.
INTERPRETAÇÃO PREVISTA. Se H1/H3 ficarem em HIGIENE e os placebos custarem igual, as duas tags saem por correção e o resultado não muda — é o desfecho mais provável e ainda assim vale (o pacote fica correto). Se algum placebo empatar com a higiene, a conclusão é que o canal p é sensível ao NÚMERO de tags, o que seria um achado sobre a fragilidade do agregador `max`, não sobre as tags.
LIMITES. 8 eventos; só o canal p (o t tem 3 dimensões reais e o argumento do sorteio é fraco lá); 2 valores de n; o PCA reajustado sobre 11 tags é outro ajuste, não um recorte do atual; nada se adota sem a sombra.

RESULTADO (19/10/2026; rodado localmente em ~2,5 min; commit do pré-registro 3399e43). NENHUM GANHO. O PLACEBO DE REMOÇÃO EXPLICA TUDO: remover QUALQUER tag do canal p dá o mesmo resultado, com duas exceções que importam.
  V1 PASSA com diferença 0,00e+00 nos dois n (o controle de 12 tags reajustado reproduz o escores_sub34 exatamente).
  CONTROLE (12 tags): 7/8 · início 6 · FP 9 · 0,807/mês · 1.457 h · 407 h fora.
  HIGIENE:  H1 sem PI_0340 (a duplicata) -> 7/8 · início 6 · FP 9 · 1.457 h · 408 h fora, nos DOIS n. IDÊNTICO ao controle a menos de 1 h. É a confirmação direta de que o PI_0340 não carrega informação: sai de graça.
            H3 sem PI_0340 e PI_5134001 -> 7/8 · FP 9 e 8 · 1.454-1.458 h. Também higiene.
            H2 sem PI_5134001 (outra unidade) -> 7/8 · FP 9-10, mas 1.590-1.593 h (+135 h de alarme) e 541-544 h fora (+135). Remover a tag que NÃO é da turbina PIORA a carga. Discrepância registrada com o C3 (`escala_teto_ar.py`, 02/10), que mediu "carga −0,4, neutro" nas 8 composições do harness antigo; aqui, no harness fiel com 8 eventos, o sinal é o oposto. Não sei qual das duas medidas representa produção; o que muda entre elas é o conjunto de composições e a janela.
  PLACEBO DE REMOÇÃO (a peça que faltou no F1 e no C3): das 5 tags restantes removidas uma a uma, QUATRO dão exatamente o mesmo que a higiene — R1 sem PI_0315 7/8 FP 9 · R2 sem PI_0339 7/8 FP 9 · R3 sem PDI_0317 7/8 FP 8-9 · R4 sem PDI_0302 7/8 FP 9-10. FP MEDIANO DOS PLACEBOS = 9,0, idêntico ao do controle e ao da higiene. Portanto o canal p é INDIFERENTE a qual tag sai: não há efeito "menos sorteios no máximo" nem efeito da tag específica.
  AS DUAS TAGS QUE IMPORTAM, e são as duas que NÃO se deveria remover:
    · PI_0319 (H4/H5): a detecção cai de 7/8 para 6/8 e o início de 6 para 4-5, MAS as horas de alarme caem de 1.457 para 804-847 h (−45%). É a maior alavanca de carga já medida nesta frente, e custa 1 detecção e 2 de início. Confirma `canal-p-univariado-por-escala` (o PI_0319 concentra 97,9% da variância escalada) e responde a hipótese que o F1 deixou aberta: tirar o PI_0319 do canal p NÃO é higiene, é uma troca na mesma fronteira de sempre.
    · PDIT_0305 (R5): remover dá 6/8, início 4, FP 12, 1.568 h — PIOR em tudo. Era um placebo e saiu como a tag mais necessária do canal.
  LEITURA DE FUNDO: de 12 tags do canal p, 10 são intercambiáveis e 2 (PI_0319 e PDIT_0305) carregam o comportamento. Isso é coerente com a dimensionalidade efetiva de 1 componente para 95% da variância, medida no pré-registro. "Retirar sensores" não é alavanca aqui — é higiene quando a tag é duplicata (PI_0340, e sai sem custo) e é troca na fronteira quando a tag é dominante (PI_0319).
PLACAR: ACERTEI: nenhum GANHO (~75%); H1 EQUIVALENTE/HIGIENE (~70%; saiu idêntico); algum placebo com FP <= o de H3 (~55%; R3 empata em 8,5); H4 muda mais que os outros (~65%; −45% de horas e −1 detecção). ERREI: previ H3 melhor que a mediana dos placebos (~40%) — ficou 8,5 contra 9,0, diferença dentro do ruído de um episódio (0,086 FP/mês); previ que o PI_5134001 sairia neutro como no C3 e ele CUSTOU +135 h; esqueci de instrumentar o k mensal do PCA, que estava no pré-registro como medida — fica NÃO MEDIDO.

Uso:  PYTHONPATH=src .venv/bin/python experiments/52_remocao_de_sensores.py
      no worker: PYTHONPATH=src python scripts/worker_exp52.py --enqueue --queue default
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
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py"); E20 = E46.E20
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; CACHE = ROOT / "escores_rem52"; NS = (20000, 19000); GL = {}
P = "954005_624_"
HIGIENE = {"H1 sem PI_0340 (duplicata)": [P + "PI_0340"], "H2 sem PI_5134001 (outra unid.)": ["PI_5134001"],
           "H3 sem as duas": [P + "PI_0340", "PI_5134001"], "H4 sem PI_0319 (hipótese F1)": [P + "PI_0319"],
           "H5 sem PI_0340 e PI_0319": [P + "PI_0340", P + "PI_0319"]}


def arquivo(nome, n):
    return CACHE / f"p_{nome.split()[0]}_{n}.parquet"


def pontua(a):
    nome, remove, n = a; f = arquivo(nome, n)
    if f.exists():
        return a
    t0 = time.time(); tags = [c for c in GL["tags_p"] if c not in remove]
    s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), GL["g"], tags, GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f)
    print(f"      {nome.split()[0]:4s} n={n} ({len(tags)} tags): {time.time() - t0:.0f} s", flush=True); return a


def linha(nome, n):
    s = pd.read_parquet(arquivo(nome, n))["s"]
    e = s.ewm(halflife=pd.Timedelta(GL["hl_p"]), times=s.index).mean().where(s.notna())
    al = decide_v2(substitui(GL["proc"], "p", e), GL["det"])["is_anomaly"].astype(bool).reindex(GL["al_c"].index).fillna(False)
    r = avalia(al, GL["vig_c"], GL["EV"]["real"], inicio_periodo=INICIO8)
    pl = [avalia(al, GL["vig_c"], ev, inicio_periodo=INICIO8).det / len(ev) for k, ev in GL["EV"].items() if k != "real" and ev]
    return dict(det=r.det, inicio=r.inicio, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora, placebo=float(np.median(pl)) * 8)


def main():
    procs = int(os.environ.get("EXP52_PROCS", 6)); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; tags_p = list(modelos[-1]["normalizacao"]["pressao"]["cols"])
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(g=g, est=est, fim=fim, ini=INICIO8 - pd.Timedelta(days=10), tags_p=tags_p, det=det, proc=proc, al_c=al_c, vig_c=vig_c, EV=EV, hl_p=det["halflife"]["p"])
    usadas = {c for v in HIGIENE.values() for c in v}
    placebos = {f"R{i+1} sem {c.replace(P, '')}": [c] for i, c in enumerate([c for c in tags_p if c not in usadas][:5])}
    braços = {"C0 controle (12 tags)": []} | HIGIENE | placebos
    print(f"EXP 52: {len(braços)} braços x {len(NS)} n | canal p | PROCS {procs}", flush=True)
    print(f"   higiene: {list(HIGIENE)}\n   placebos de remoção: {list(placebos)}", flush=True)
    t0 = time.time()
    with Pool(procs) as pool:
        list(pool.imap_unordered(pontua, [(nome, rem, n) for nome, rem in braços.items() for n in NS], chunksize=1))
    print(f"ESCORES em {time.time() - t0:.0f} s", flush=True)
    for n in NS:
        f34 = ROOT / "escores_sub34" / f"p_A_{n}.parquet"
        if f34.exists():
            a = pd.read_parquet(f34)["s"]; b = pd.read_parquet(arquivo("C0 controle (12 tags)", n))["s"]; j = a.index.intersection(b.index)
            print(f"   V1 [n={n}] controle contra escores_sub34: diferença máxima = {float(np.nanmax(np.abs(a.loc[j].to_numpy() - b.loc[j].to_numpy()))):.2e}", flush=True)
    res = {}
    print("\nRESULTADOS (detecção/8 | início | FP | FP/mês | horas totais | h fora | placebo de eventos)", flush=True)
    for nome in braços:
        for n in NS:
            L = linha(nome, n); res[(nome, n)] = L
            print(f"   {nome:34s} n={n} | {L['det']}/8 | {L['inicio']} | FP {L['fp']:3d} | {L['fp_mes']:.3f} | {L['horas']:6.0f} h | {L['fora']:5.0f} h | {L['placebo']:.1f}", flush=True)
    c0 = {n: res[("C0 controle (12 tags)", n)] for n in NS}
    med_plac = float(np.median([res[(nome, n)]["fp"] for nome in placebos for n in NS]))
    print(f"\n   FP mediano dos {len(placebos)} placebos de remoção: {med_plac:.1f} | controle: {np.mean([c0[n]['fp'] for n in NS]):.1f}", flush=True)
    print("\nCLASSIFICAÇÃO (GANHO = FP <= 6, det >= 6, horas <= 1457 e melhor que a mediana dos placebos; HIGIENE = det e FP a ±1 do controle)", flush=True)
    for nome in list(HIGIENE) + list(placebos):
        d = float(np.mean([res[(nome, n)]["det"] for n in NS])); fp = float(np.mean([res[(nome, n)]["fp"] for n in NS]))
        h = float(np.mean([res[(nome, n)]["horas"] for n in NS])); dc = float(np.mean([c0[n]["det"] for n in NS])); fc = float(np.mean([c0[n]["fp"] for n in NS]))
        cl = "GANHO" if (fp <= 6 and d >= 6 and h <= 1457 and fp < med_plac) else "PIOR" if (d <= 5 or fp >= 12) else "EQUIVALENTE"
        hig = " | HIGIENE (a ±1 do controle)" if (abs(d - dc) <= 1 and abs(fp - fc) <= 1) else ""
        print(f"   {nome:34s} det {d:.1f} (controle {dc:.1f}) | FP {fp:.1f} (controle {fc:.1f}) -> {cl}{hig}", flush=True)
    pd.DataFrame([dict(braco=nome, n=n, **L) for (nome, n), L in res.items()]).to_csv(ROOT / "exp52_resultados.csv", index=False)


if __name__ == "__main__":
    main()
