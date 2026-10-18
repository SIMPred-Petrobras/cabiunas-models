#!/usr/bin/env python3
"""EXP 49 — COMPLEMENTARIDADE: PCA UNIDO A OUTRO MODELO, NO MESMO ORÇAMENTO DE HORAS, COM CONTROLE E PLACEBO.

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 48b (pós-hoc) mostrou que o PCA perde o evento de 27/02/2025 nos dois canais, enquanto K-Means e LSTM (em t) e análise fatorial e LSTM (em p) o detectam nos dois n, e que sob a calibração do EXP 46 alguns modelos ficam nominalmente à frente do PCA (K-Means em t 6,5 e LSTM 6,0 contra 5,5; análise fatorial em p 7,0 contra 6,5). Se modelos diferentes pegam eventos diferentes, a união pode somar detecção. O risco é conhecido: unir alarmes soma FP, e o EXP 40 mostrou que um segundo modelo não separa FP de acerto. Aqui a união é feita NO MESMO TOTAL DE HORAS de alarme da referência, para que ganho de detecção não seja só mais cobertura.
O QUE SE FAZ (8 eventos; escores PCA(0,95) do escores_sub34 e K-Means, análise fatorial e LSTM do EXP 44; n = 20.000 e 19.000; canais t e p isolados; EWMA 1 h; episódio >= 2 h; refratário 72 h; avaliador do EXP 46).
  Referência: o PCA sozinho a 1.457 h de alarme (limiar por bisseção sem rótulos, piso no quantil 5%).
  Fusões (2, FIXADAS AQUI): F1 = UNIÃO DOS ALARMES: cada modelo recebe o mesmo orçamento b e o alarme final é a união; b é escolhido por bisseção para que as horas da UNIÃO sejam 1.457 h (horas iguais à referência). F2 = MÁXIMO DOS ESCORES: o escore de cada modelo já vem em múltiplos do p99 do treino; o escore fundido é o máximo dos dois escores (após o EWMA de 1 h), com um limiar único calibrado para 1.457 h.
  Pares (6, FIXADOS AQUI): PCA com K-Means, com análise fatorial e com LSTM, em cada canal.
  CONTROLE: o PCA unido a ele mesmo (n = 20.000 com n = 19.000), pelas mesmas duas fusões e o mesmo orçamento: mede quanto da "complementaridade" vem só de unir dois detectores ruidosos do mesmo tipo.
MEDIDAS: detecção de 8 (média dos 2 n; a união usa o par de escores do MESMO n, e o controle usa os dois n), FP, horas, eventos detectados pela fusão e não pelo PCA sozinho, e o de 27/02/2025; placebo (eventos deslocados de +45 a +225 d, o mesmo alarme).
CRITÉRIOS (fixados antes). GANHO (por par, fusão e canal) = a detecção média da fusão >= a do PCA sozinho + 2, E acima do placebo da própria fusão, E acima do controle (PCA unido a ele mesmo, mesma fusão e canal) por >= 1. PIOR = <= PCA sozinho - 2. EQUIVALENTE no resto. O FP é relatado, mas o critério é a detecção a horas iguais. Piso de ruído com 8 eventos: +-2 (o PCA sozinho oscila 2 eventos só por mudar n); são 12 células (6 pares x 2 fusões) por canal contra a referência: qualquer GANHO é HIPÓTESE e exige confirmação com n = 21.000 e 18.000 (calculando o PCA a partir do harness) antes de ser tomado como achado.
EXPECTATIVAS: GANHO em alguma célula ~12%; EQUIVALENTE na maioria das células ~65%; PIOR ~23%; o controle (PCA com ele mesmo) soma >= +1 em ao menos um canal ~35%; a fusão detecta o evento de 27/02/2025 em ao menos um par ~70%; o FP da união (F1) >= o do PCA sozinho em a maioria das células ~60%; F2 (máximo dos escores) mais próximo do PCA sozinho que F1 ~65%.
INTERPRETAÇÃO PREVISTA. Se nenhuma célula ganha, a diferença entre modelos que o 48b viu é ruído de n e de política, e a complementaridade não é uma alavanca; se alguma ganha e o controle não explica, vira HIPÓTESE para a estrutura de voto, com confirmação e placebo próprios. Em qualquer caso, o 27/02/2025 fica documentado como evento que só alguns modelos veem.
LIMITES. 8 eventos (+-2); 2 valores de n; 12 células por canal (melhor de 12: por isso a confirmação); escores do EXP 44 (n = 20.000 e 19.000 só); canais isolados, sem voto nem CUSUM; hiperparâmetros por convenção; a união dobra o número de episódios candidatos e pode inflar o FP mesmo a horas iguais.

Uso:  PYTHONPATH=src .venv/bin/python experiments/49_complementaridade_de_modelos.py   (precisa de data_local/escores_lit44_w, gerado pelo EXP 48)
      variável: EXP49_PROCS (processos; padrão 4)
"""
from __future__ import annotations
import importlib.util, os, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py")
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.politica import ewma_vigiado, episodios, alarme_de

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; SHIFTS = E46.SHIFTS; ORC = 1457.0; NS = (20000, 19000); GL = {}


def carrega(c, nome, n):
    f = ROOT / "escores_sub34" / f"{c}_A_{n}.parquet" if nome == "PCA" else ROOT / "escores_lit44_w" / f"{c}_{nome}_{n}.parquet"
    return pd.read_parquet(f)["s"].loc[GL["ini"]:GL["fim"]]


def ewma(c, nome, n):
    k = (c, nome, n)
    if k not in GL["EW"]:
        s = carrega(c, nome, n); GL["EW"][k] = ewma_vigiado(s, GL["vig"].loc[s.index], "1.0h")
    return GL["EW"][k]


def horas(al):
    return avalia(al, GL["vig"], [], inicio_periodo=INICIO8).horas_alarme


def alarme_orcamento(e, b, dm=pd.Timedelta(hours=2)):
    """Alarme do EWMA `e` com limiar por bisseção para `b` horas (sem rótulos, piso no quantil 5%)."""
    ed = e.dropna(); lo, hi = float(ed.quantile(0.05)), float(ed.max())
    h = lambda t: horas(alarme_de(episodios(e, t, dur_min=dm), e.index))
    for _ in range(18):
        mid = float(np.sqrt(max(lo, 1e-12) * hi)); lo, hi = (mid, hi) if h(mid) > b else (lo, mid)
    return alarme_de(episodios(e, float(np.sqrt(max(lo, 1e-12) * hi)), dur_min=dm), e.index)


def uniao_horas_iguais(e1, e2, alvo=ORC):
    """F1: orçamento comum b por bisseção para que as horas da UNIÃO sejam `alvo`."""
    lo, hi = 50.0, alvo
    for _ in range(12):
        mid = 0.5 * (lo + hi); a = alarme_orcamento(e1, mid) | alarme_orcamento(e2, mid)
        if horas(a) > alvo:
            hi = mid
        else:
            lo = mid
    b = 0.5 * (lo + hi); return alarme_orcamento(e1, b) | alarme_orcamento(e2, b), b


def maximo_escores(e1, e2):
    return pd.concat([e1, e2], axis=1).max(axis=1, skipna=False).where(e1.notna() & e2.notna())


def mede(al, EV):
    out = {}
    for nome, ev in EV.items():
        r = avalia(al, GL["vig"], ev, inicio_periodo=INICIO8); out[nome] = ([bool(r.por_evento.get(str(t), {}).get("det", False)) for t in ev], r.fp)
    return out


def tarefa(a):
    c, alt, fus, n = a; EV = GL["EV"]
    if alt == "PCA":   # controle: PCA(n=20.000) com PCA(n=19.000), um único "n" de saída
        e1, e2 = ewma(c, "PCA", NS[0]), ewma(c, "PCA", NS[1])
    elif alt == "SOLO":
        e1 = ewma(c, "PCA", n); al = alarme_orcamento(e1, ORC); return a, mede(al, EV), float(horas(al)), None
    else:
        e1, e2 = ewma(c, "PCA", n), ewma(c, alt, n)
    if fus == "F1":
        al, b = uniao_horas_iguais(e1, e2)
    else:
        al = alarme_orcamento(maximo_escores(e1, e2), ORC); b = None
    return a, mede(al, EV), float(horas(al)), b


def main():
    procs = int(os.environ.get("EXP49_PROCS", 4)); reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL.update(vig=vig_c, fim=fim, ini=INICIO8 - pd.Timedelta(days=10), EV=EV, EW={}); canais = ("t", "p"); alts = ("KM", "FA", "LSTM")
    print(f"EXP 49: {len(EV['real'])} eventos | pares {alts} | fusões F1/F2 | PROCS {procs}", flush=True)
    tarefas = [(c, "SOLO", "F0", n) for c in canais for n in NS] + [(c, alt, fus, n) for c in canais for alt in alts for fus in ("F1", "F2") for n in NS] + [(c, "PCA", fus, NS[0]) for c in canais for fus in ("F1", "F2")]
    with Pool(procs) as p:
        out = {a: (m, h, b) for a, m, h, b in p.imap_unordered(tarefa, tarefas, chunksize=1)}
    nomes = {"KM": "K-Means", "FA": "análise fatorial", "LSTM": "LSTM", "PCA": "controle PCA+PCA"}
    def resumo(a):
        m, h, b = out[a]; real = m["real"]; pl = float(np.median([sum(v[0]) / len(v[0]) for k, v in m.items() if k != "real" and len(v[0])])) if any(k != "real" and len(v[0]) for k, v in m.items()) else float("nan")
        return sum(real[0]), real[1], h, pl * 8, real[0]
    ev_nomes = [t.strftime("%d/%m/%Y") for t in EV["real"]]
    print("\nPCA SOZINHO (referência) e FUSÕES a horas iguais (1.457 h): detecção de 8 | FP | horas | placebo", flush=True); ref = {}
    for c in canais:
        for n in NS:
            d, fp, h, pl, v = resumo((c, "SOLO", "F0", n)); ref[(c, n)] = (d, fp, v); print(f"   [{c}] PCA sozinho n={n}: {d}/8, FP {fp}, {h:.0f} h, placebo {pl:.1f}/8", flush=True)
    res = {}
    for c in canais:
        print(f"\n[{c}] fusões por par (média dos 2 n; entre colchetes por n)", flush=True)
        for alt in alts + ("PCA",):
            for fus in ("F1", "F2"):
                if alt == "PCA":
                    d, fp, h, pl, v = resumo((c, "PCA", fus, NS[0])); res[(c, alt, fus)] = (float(d), float(fp), pl, [v]); print(f"   {nomes[alt]:18s} {fus}: {d}/8 (um único valor; n=20.000 com 19.000) FP {fp}, {h:.0f} h, placebo {pl:.1f}/8", flush=True); continue
                xs = [resumo((c, alt, fus, n)) for n in NS]; md = float(np.mean([x[0] for x in xs])); mf = float(np.mean([x[1] for x in xs])); mp = float(np.mean([x[3] for x in xs]))
                res[(c, alt, fus)] = (md, mf, mp, [x[4] for x in xs]); print(f"   {nomes[alt]:18s} {fus}: {md:.1f}/8 [{xs[0][0]}, {xs[1][0]}] FP {mf:.1f} horas {np.mean([x[2] for x in xs]):.0f} placebo {mp:.1f}/8", flush=True)
    print("\nEVENTOS que a fusão detecta e o PCA sozinho não (nos 2 n; n=20.000 | n=19.000) e o de 27/02/2025", flush=True)
    for c in canais:
        for alt in alts:
            for fus in ("F1", "F2"):
                lin = []
                for k, n in enumerate(NS):
                    v = res[(c, alt, fus)][3][k]; r0 = ref[(c, n)][2]; ganhos = [ev_nomes[i] for i in range(len(v)) if v[i] and not r0[i]]; perdas = [ev_nomes[i] for i in range(len(v)) if r0[i] and not v[i]]
                    lin.append(f"n={n}: +{len(ganhos)}{ganhos} -{len(perdas)}{perdas}; 27/02: {'sim' if v[0] else 'não'}")
                print(f"   [{c}] {nomes[alt]:16s} {fus}: " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO (GANHO = >= PCA sozinho + 2 E acima do placebo E acima do controle por >= 1; PIOR = <= PCA sozinho - 2)", flush=True)
    for c in canais:
        refm = float(np.mean([ref[(c, n)][0] for n in NS])); print(f"   [{c}] PCA sozinho: {refm:.1f}/8", flush=True)
        for fus in ("F1", "F2"):
            ctrl = res[(c, "PCA", fus)][0]
            for alt in alts:
                md, mf, mp, _ = res[(c, alt, fus)]; cl = "GANHO (a confirmar em n=21.000 e 18.000)" if (md >= refm + 2 and md > mp and md >= ctrl + 1) else "PIOR" if md <= refm - 2 else "EQUIVALENTE"
                print(f"      {fus} {nomes[alt]:16s}: {md:.1f}/8 (placebo ~{mp:.1f}; controle {ctrl:.1f}) -> {cl}", flush=True)
    rows = [dict(canal=a[0], alt=a[1], fusao=a[2], n=a[3], det=sum(m['real'][0]), fp=m['real'][1], horas=h, b=b) for a, (m, h, b) in out.items()]
    pd.DataFrame(rows).to_csv(ROOT / "exp49_resultados.csv", index=False)


if __name__ == "__main__":
    main()
