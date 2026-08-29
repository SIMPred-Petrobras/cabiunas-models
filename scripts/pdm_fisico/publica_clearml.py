#!/usr/bin/env python3
"""Publica o detector fisico de 4 sinais do TC-330.03A no ClearML.

Por que existe: o Francisco ja publica as reproducoes dele em
`pca-walkforward::monitoramento_sistema_v*` no projeto TesteMLCab. Esta tarefa
poe o NOSSO detector na mesma regua e no mesmo lugar, para comparacao direta
pelo time -- mesma janela de deteccao (48 h), mesmo agrupamento de episodios
(2 h) e mesmo denominador de falso positivo (mes de OPERACAO, 730 h).

Nada aqui e reajustado: o ponto de operacao vem fixo (busca conjunta de 2.187
configuracoes ja concluida) e as metricas sao remedidas do zero a partir do
cache de sinais, para que o numero publicado seja o numero reproduzivel.

Uso:  PYTHONPATH=. python scripts/pdm_fisico/publica_clearml.py [--offline]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)
sys.path.insert(0, AQUI)

# O pacote cabiunas_pdm vivia no scratchpad e foi apagado. As constantes que ele
# fornecia estao replicadas aqui, com a origem de cada uma, para que esta tarefa
# nao dependa de nada fora do repositorio. A prova de que estao certas e a
# reproducao exata dos numeros ja validados (8/8, 21 episodios, 1,12 FP/mes).
import avalia as AV

GRID = "2min"
BLACKOUT = "6h"          # apaga as 6 h seguintes a cada religamento
SUSTAIN = 15             # 15 amostras de 2 min = 30 min acima do limite
THR_FAM = 2.0            # limiar base das familias t e p
THR_SPREAD = 3.0         # limiar base do spread de mancal
VIBRATION_TAGS = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
                  "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]

T0 = pd.Timestamp("2025-01-01", tz="UTC")
SIN = ["t", "p", "sp", "vb"]
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": THR_FAM, "p": THR_FAM, "sp": THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
KAPPA, H_CUSUM, CARGA = 0.75, 80, 0.25
REFRAT_H, DUR_MIN = 48, 120
ORC_FP = 1.15          # orcamento de FP/mes usado na selecao do LOEO aninhado


def reproduz():
    """Recalcula sinais -> EWMA -> degrau|CUSUM -> voto>=2 -> refratario -> duracao."""
    g = pd.read_parquet("grade2min.parquet")
    idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
    black = part.rolling(n_bl, min_periods=1).max().astype(bool)
    sel = idx >= T0
    mask = (estavel & ~black) & sel

    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = pd.Series(list(fal[fal >= T0]))

    z = np.load("piso_fisico_cache.npz")
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    out = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vbz}, index=idx)

    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in HL.items()}
    reset = ((~mask) | part).to_numpy()

    def cusum(zz):
        x = (zz - KAPPA).fillna(0.0).to_numpy()
        S = np.empty(len(x)); acc = 0.0
        for i in range(len(x)):
            acc = acc * CARGA if reset[i] else max(0.0, acc + x[i])
            S[i] = acc
        return S > H_CUSUM

    ON = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        n = SUSTAIN
        deg = ((E[c] > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
        ON[c] = (deg | pd.Series(cusum((E[c] / thr).clip(upper=20)), index=idx)) & mask
    voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask

    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel, mask, alvo, ON, idx, sel


def loeo_aninhado(alvo):
    """LOEO a partir da busca conjunta ja rodada: para cada evento retirado,
    escolhe a config so com os 7 restantes (orcamento de FP fixo) e pergunta se
    ela pega o retirado. Sem isso o 8/8 seria apenas ajuste no proprio alvo.

    A regra de desempate NAO e neutra e por isso e fixada aqui a priori: entre
    configs empatadas na deteccao de treino, fica a de menor custo em horas de
    falso positivo. Sem desempate algum o argmax cai na primeira linha do CSV --
    ordem de varredura, nao merito -- e o resultado sobe artificialmente para 8/8.
    O quadro completo das quatro regras vai como artefato."""
    if not os.path.exists("busca_conjunta.csv"):
        return None
    df = pd.read_csv("busca_conjunta.csv")
    df["set"] = df["quais"].fillna("").apply(lambda s: set(x for x in s.split(",") if x))
    cand = df[df["fp"] <= ORC_FP].reset_index(drop=True)
    if not len(cand):
        return None
    dias = [t.strftime("%Y-%m-%d") for t in alvo]
    lead = np.nan_to_num(cand["lead"].to_numpy(), nan=0.0)   # NaN envenena o argmax
    regras = {"sem_desempate": (0.0, 0.0, 0.0), "menos_horas_fp": (1.0, 0.0, 0.0),
              "menos_fp": (0.0, 10.0, 0.0), "mais_lead": (0.0, 0.0, 0.01)}

    def roda(wh, wf, wl):
        ok, perdidos, escolhas = 0, [], []
        for d in dias:
            tr = set(dias) - {d}
            n = cand["set"].apply(lambda s: len(s & tr)).to_numpy()
            sc = n * 1000.0 - cand["hm"].to_numpy() * wh - cand["fp"].to_numpy() * wf + lead * wl
            L = cand.iloc[int(np.argmax(sc))]
            acertou = d in L["set"]
            ok += acertou
            if not acertou:
                perdidos.append(d)
            escolhas.append(dict(evento=d, acertou=bool(acertou), kb=L.kb, kv=L.kv, ka=L.ka,
                                 h=L.h, cr=L.cr, R=L.R, D=L.D, fp_mes=round(float(L.fp), 3)))
        return ok, perdidos, pd.DataFrame(escolhas)

    quadro = pd.DataFrame([dict(regra=k, loeo=f"{roda(*v)[0]}/{len(dias)}",
                                perdidos=",".join(roda(*v)[1]))
                           for k, v in regras.items()])
    # fragilidade: quantas das configs dentro do orcamento pegam cada evento
    frag = pd.DataFrame([dict(evento=d,
                              configs_que_detectam=int(cand["set"].apply(lambda s: d in s).sum()),
                              de=len(cand),
                              fracao=round(float(cand["set"].apply(lambda s: d in s).mean()), 3))
                         for d in dias]).sort_values("fracao")
    ok, perdidos, esc = roda(*regras["menos_horas_fp"])       # regra publicada
    return ok, len(dias), esc, quadro, frag, perdidos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="so mede, nao publica")
    ap.add_argument("--nome", default="detector-fisico::TC33003A_4sinais_v1")
    args = ap.parse_args()

    al, mask, alvo, ON, idx, sel = reproduz()
    quente = mask & sel
    m = AV.avalia(al, alvo, quente)
    perm = AV.permuta(al, quente, m["det"], len(alvo))
    eps = AV.episodios(al)
    jan = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    fps = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jan)]

    # leads por evento, com a mesma regra da regua (primeiro alerta DENTRO da janela)
    linhas = []
    for t in alvo:
        d = al.loc[(al.index >= t - pd.Timedelta(hours=48)) & (al.index < t)]
        d = d[d.fillna(False)]
        linhas.append(dict(evento=t.strftime("%Y-%m-%d %H:%M"),
                           detectado=bool(len(d)),
                           lead_h=round((t - d.index[0]).total_seconds() / 3600, 2) if len(d) else np.nan,
                           censurado_48h=bool(len(d) and abs((t - d.index[0]).total_seconds() / 3600 - 48) < 0.05)))
    tab_ev = pd.DataFrame(linhas)
    tab_fp = pd.DataFrame([dict(inicio=str(a), fim=str(b),
                                horas=round((b - a).total_seconds() / 3600 + 2 / 60, 2))
                           for a, b in fps]).sort_values("horas", ascending=False)

    lo = loeo_aninhado(alvo)
    meses = m["horas_op"] / 730.0
    res = {
        "recall": f'{m["det"]}/{m["n_ev"]}',
        "recall_frac": m["det"] / m["n_ev"],
        "episodios": m["episodios"],
        "fp": m["fp"],
        "fp_por_mes_operacao": round(m["fp_mes"], 3),
        "alarmes_por_ano": round(m["fp_mes"] * 12, 1),
        "horas_fp_por_mes": round(m["h_fp_mes"], 1),
        "duty_cycle_pct": round(100 * m["duty"], 2),
        "lead_medio_h": round(m["lead_med"], 2),     # MEDIA, nao mediana
        "lead_min_h": round(m["lead_min"], 2),
        "leads_censurados_48h": int(tab_ev["censurado_48h"].sum()),
        "precisao_episodios_pct": round(100 * m["det"] / max(m["episodios"], 1), 1),
        "meses_operacao": round(meses, 1),
        "horas_operacao": round(m["horas_op"], 1),
        "permut_esperado_acaso": round(perm["nulo"], 2),
        "permut_p": perm["p"],
        "permut_cobertura": round(perm["cobertura"], 4),
        "maior_fp_pct_das_horas": round(100 * tab_fp["horas"].iloc[0] / tab_fp["horas"].sum(), 1) if len(tab_fp) else 0.0,
    }
    if lo:
        res["loeo_aninhado"] = f"{lo[0]}/{lo[1]}"
        res["loeo_frac"] = lo[0] / lo[1]
        res["loeo_eventos_perdidos"] = ",".join(lo[5])
        res["loeo_evento_mais_fragil"] = f'{lo[4].iloc[0]["evento"]} ({lo[4].iloc[0]["fracao"]:.0%} das configs no orcamento)'

    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    print("\n--- por evento ---\n", tab_ev.to_string(index=False), flush=True)
    print("\n--- falsos positivos ---\n", tab_fp.to_string(index=False), flush=True)
    if lo:
        print("\n--- LOEO aninhado (regra publicada: menos horas de FP) ---\n",
              lo[2].to_string(index=False), flush=True)
        print("\n--- sensibilidade a regra de desempate ---\n", lo[3].to_string(index=False), flush=True)
        print("\n--- fragilidade por evento ---\n", lo[4].to_string(index=False), flush=True)
    if args.offline:
        return

    from clearml import Task, Logger
    task = Task.init(project_name="TesteMLCab", task_name=args.nome,
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     auto_connect_frameworks=False)
    task.connect({
        "sinais": "t,p (erro rec. PCA max-por-sensor, piso 0.10) | sp (z do spread de mancal) | vb (max z das 10 TV_35*)",
        "mascara": "RUNNING_A>0.5 AND T5_AVG_A>300C, menos blackout de 6 h pos-partida",
        "ajuste_pca": "walk-forward mensal, FIT_POINTS=20000 amostras estaveis (666.7 h), PCA n_components=0.95, RobustScaler",
        "ewma_halflife": json.dumps(HL), "k_por_sinal": json.dumps(K),
        "cusum_kappa": KAPPA, "cusum_h": H_CUSUM, "cusum_carga_residual": CARGA,
        "voto_minimo": 2, "refratario_h": REFRAT_H, "duracao_minima_min": DUR_MIN,
        "sustain_min": SUSTAIN * 2, "blackout_pos_partida": BLACKOUT,
        "janela_deteccao_h": AV.JANELA_H, "gap_episodio_h": AV.GAP_EP_H,
        "denominador_fp": "mes de OPERACAO (730 h), nao de calendario",
        "janela_alvo": "a partir de 2025-01-01 (2024-01-16 e artefato de partida a frio: 0 pontos validos)",
        "orcamento_fp_loeo": ORC_FP,
    }, name="detector_fisico_config")

    lg: Logger = task.get_logger()
    for k, v in res.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lg.report_single_value(k, float(v))
    lg.report_text(
        "Ponto de operacao confirmado por duas rotas independentes: ajuste em cascata e busca "
        "conjunta (2.187 configs, 693 com 8/8; o menor FP entre elas e este ponto).\n"
        "O LOEO aninhado NAO confirma 8/8: sob qualquer regra de desempate razoavel o resultado "
        "e 7/8, e o evento que nao sobrevive e sempre 2025-11-04 -- detectado por apenas 8 das 72 "
        "configuracoes dentro do orcamento de FP. E o mesmo evento que ja havia caido em cinco "
        "intervencoes anteriores e o unico dos oito sem precursor fisico atribuivel. "
        "Leitura honesta: 7 das 8 paradas tem deteccao robusta a reajuste; a oitava e sorte do "
        "ponto de operacao.")
    task.upload_artifact("metricas", res)
    task.upload_artifact("por_evento", tab_ev)
    task.upload_artifact("falsos_positivos", tab_fp)
    if lo:
        task.upload_artifact("loeo_aninhado", lo[2])
        task.upload_artifact("loeo_sensibilidade_desempate", lo[3])
        task.upload_artifact("loeo_fragilidade_por_evento", lo[4])
    for f, tit in [("fig_anomalias_serie.png", "serie e anomalias"),
                   ("fig_anomalias_zoom.png", "72 h antes de cada parada"),
                   ("../../RELATORIO_DETECTOR_TC33003A.pdf", "relatorio completo")]:
        if os.path.exists(f):
            task.upload_artifact(os.path.basename(f), f)
            if f.endswith(".png"):
                lg.report_image("figuras", tit, local_path=f, iteration=0)
    task.flush(wait_for_uploads=True)
    print("\nClearML:", task.get_output_log_web_page(), flush=True)
    task.close()


if __name__ == "__main__":
    main()
