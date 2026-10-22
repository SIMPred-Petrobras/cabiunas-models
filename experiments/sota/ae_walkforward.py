#!/usr/bin/env python3
"""SOTA FASE 1b — O PIPELINE DE 17 AEs CNN-1D UNIVARIADOS EM WALK-FORWARD MENSAL, no protocolo único, contra o PCA v2. Tag ClearML `sota-ae`. Timebox: 1 dia.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. Decisão do usuário: refazer o AE com retreino mensal (os bundles atuais treinaram em todo 2025, o próprio período
de teste); se não chegar perto do PCA v2, registrar e seguir.
FIEL AO PIPELINE (`src/cnn1d_ae`, `configs/calibracao_v9_cnn_variants/v9_prod_sensores_2025.json`): 17 AEs, um por sensor (TC382_01..06_A, T5_AVG_A, TV_351..355 X/Y);
CNN-1D (2 Conv1D -> gargalo com downsampling -> 2 Conv1DTranspose), z-score após corte robusto (MAD, k = 3,5); janela de 48 min; erro = MAE da janela; health index =
EWMA do erro (meia-vida 4 h; 0,5 h no T5_AVG_A e no TC382_04_A), carregando o último valor fora de operação; níveis por quantil do health de cada sensor — WARN q 0,80 com
>= 2 sensores, ALARM q 0,90 com >= 2, CRITICAL q 0,95 com >= 1. ALARME DA MÁQUINA = nível ALARM (WARN e CRITICAL reportados à parte).
O QUE MUDA, E POR QUÊ (declarado):
  (1) PROTOCOLO: grade de 2 min e máscara de vigilância do v2 (a NGP_A não existe depois de out/2025). A janela de 96 amostras de 30 s vira 24 de 2 min (os mesmos 48 min).
  (2) WALK-FORWARD: em cada mês de fev/2025 a abr/2026, cada AE treina nos 180 d anteriores de operação vigiada, sem [T − 48 h, T + 2 d] dos trips já ocorridos (o
      original exclui ±48 h dos alarmes), com os últimos 10% (no tempo) para validação e parada antecipada.
  (3) LIMIARES DOS NÍVEIS: o original tira o quantil do health da PRÓPRIA série avaliada (`pipeline_multi.py`, thr_q = quantile(health[seq_run_full], q)), o que usa o
      teste para calibrar. Aqui o quantil vem do health nos 10% de validação de cada mês (dado anterior ao mês, não usado no gradiente).
  (4) HIPERPARÂMETROS FIXOS no centro do espaço do KerasTuner (sem busca, pelo timebox): filtros 32 e 16, kernels 5 e 5, strides 1 e 2, dropout 0,1, lr 1e-3, L2 1e-4;
      até 20 épocas, paciência 3, lote 256, janelas de treino a cada 3 passos. PyTorch em CPU (o original é Keras).
  (5) RUÍDO DE TREINO: 3 sementes (o piso de ruído das redes aqui é alto, EXP 07); a classificação usa a semente mediana em FP.
MEDIDAS: o protocolo completo (8 trips, 48 h, de pé e início, FP/mês, horas, lead censurado, F1, IC) do nível ALARM, WARN e CRITICAL, por semente; a comparação pareada com
o PCA v2 (base atual); régua SECUNDÁRIA (a do pipeline): incidentes HI/HIHI dos 17 sensores no diário de alarmes com a máquina operando, agrupados por sensor com
intervalo de 4 h, acertados se o health daquele sensor passa do q 0,90 dele em [início − 8 h, início]; recall macro e fração do tempo vigiado acima do q 0,90.
CRITÉRIO (fixado antes): PERTO DO PCA v2 = na semente mediana, detecção >= PCA − 1 E FP/mês <= 1,5 x PCA E horas <= 1,5 x PCA. Senão, NÃO CHEGA PERTO: registrar e seguir
(sem ajuste — o timebox não comporta e ajustar no teste seria calibrar com o teste).
EXPECTATIVAS: V1 passa (~90%); NÃO CHEGA PERTO (~75%: 17 univariados com voto de 2 sensores acendem muito em operação normal; o TC382_03 do pipeline original ficava
53% do tempo em alerta); detecção do ALARM >= 6/8 (~50%); recall macro da régua secundária >= 0,60 (~50%).
LIMITES: hiperparâmetros sem busca; 3 sementes; a régua secundária depende do diário (alarmes suprimidos ou perdidos não contam); timebox de 1 dia.
VERIFICAÇÃO: V1 nenhum treino usa dado do mês avaliado nem das janelas de trip conhecidas; os limiares de nível só usam a validação (anterior ao mês).

Uso local (encanamento: 2 sensores, 2 meses, 1 semente):  SOTA_TESTE=1 PYTHONPATH=src:experiments .venv/bin/python -m sota.ae_walkforward
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp ae_walkforward --tags sota-ae --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv import io as nvio
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.avalia import carrega_controle
from sota import protocolo as P

TESTE = bool(os.environ.get("SOTA_TESTE"))
ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "ae_walkforward"
SENSORES = [f"TC382_0{i}_A" for i in range(1, 7)] + ["T5_AVG_A"] + [f"TV_35{i}{e}_A" for i in range(1, 6) for e in "XY"]
L, STRIDE_TREINO, DIAS_TREINO = 24, 3, 180; HL = {"T5_AVG_A": 0.5, "TC382_04_A": 0.5}; HL0 = 4.0
NIVEIS = {"warn": (0.80, 2), "alarm": (0.90, 2), "critical": (0.95, 1)}
SEMENTES = (0,) if TESTE else (0, 1, 2)
MESES = pd.date_range("2025-02-01", "2026-04-01", freq="MS", tz="UTC")[: (2 if TESTE else None)]
GL = {}
ALARMES = Path(os.environ.get("ALARMES_CSV", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv"))


def modelo(semente):
    import torch, torch.nn as nn
    torch.manual_seed(semente)
    return nn.Sequential(nn.Conv1d(1, 32, 5, stride=1, padding=2), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.ReLU(),
                         nn.ConvTranspose1d(16, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(), nn.Dropout(0.1),
                         nn.ConvTranspose1d(16, 32, 5, stride=1, padding=2), nn.ReLU(), nn.Conv1d(32, 1, 5, padding=2))


def treina(X, semente):
    import torch
    torch.set_num_threads(1); n = len(X); nv = max(1, int(0.1 * n)); Xt, Xv = X[:n - nv], X[n - nv:]
    m = modelo(semente); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); f = torch.nn.L1Loss(); rng = np.random.default_rng(semente)
    tv = torch.tensor(Xv[:, None, :], dtype=torch.float32); melhor, pac, estado = np.inf, 0, None
    for _ in range(20):
        m.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // 256)):
            xb = torch.tensor(Xt[b][:, None, :], dtype=torch.float32); opt.zero_grad(); loss = f(m(xb), xb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lv = float(f(m(tv), tv))
        if lv < melhor - 1e-6:
            melhor, pac, estado = lv, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            pac += 1
            if pac >= 3:
                break
    m.load_state_dict(estado); m.eval(); return m, nv


def mae(m, W):
    import torch
    out = np.empty(len(W))
    with torch.no_grad():
        for i in range(0, len(W), 4096):
            x = torch.tensor(W[i:i + 4096][:, None, :], dtype=torch.float32); out[i:i + 4096] = (m(x) - x).abs().mean(dim=(1, 2)).numpy()
    return out


def ewma_hi(s, hl):
    a = 1.0 - 0.5 ** ((2 * 60) / 3600.0 / hl); out = np.empty(len(s)); prev = np.nan
    for i, v in enumerate(s):
        if np.isfinite(v):
            prev = v if np.isnan(prev) else a * v + (1 - a) * prev
        out[i] = prev
    return out


def um_sensor(arg):
    """Walk-forward de um sensor e uma semente: erro por instante (fim da janela), health index e limiares de nível de cada mês."""
    sensor, semente = arg; g, vig, longe_de, idx = GL["g"], GL["vig"], GL["longe_de"], GL["idx"]
    x = g[sensor].to_numpy("float64"); ok = np.isfinite(x) & vig
    from numpy.lib.stride_tricks import sliding_window_view as swv
    okw = np.r_[np.zeros(L - 1, bool), swv(ok, L).all(axis=1)]          # janela de L pontos válidos terminando em i
    err = np.full(len(x), np.nan); thr = {}
    for k, m0 in enumerate(MESES):
        m1 = MESES[k + 1] if k + 1 < len(MESES) else P.FIM + pd.Timedelta(minutes=2)
        i0, i1 = idx.searchsorted(m0), idx.searchsorted(m1); j0 = idx.searchsorted(m0 - pd.Timedelta(days=DIAS_TREINO))
        tr = np.flatnonzero(okw[j0:i0] & longe_de(m0)[j0:i0]) + j0; tr = tr[(tr - (L - 1)) >= j0][::STRIDE_TREINO]
        GL.setdefault("_v1", []).append(bool(len(tr) == 0 or tr.max() < i0))
        if len(tr) < 2000:
            continue
        base = x[np.concatenate([np.arange(e - L + 1, e + 1) for e in tr[:: max(1, len(tr) // 2000)]])]
        med = np.median(base); mad = 1.4826 * np.median(np.abs(base - med)) or 1e-6; cl = np.clip(base, med - 3.5 * mad, med + 3.5 * mad); mu, sd = cl.mean(), cl.std() or 1e-6
        z = lambda e: (swv(x, L)[np.asarray(e) - (L - 1)] - mu) / sd
        mod, nv = treina(z(tr), semente)
        ev = mae(mod, z(tr[-nv:])); hv = ewma_hi(ev, HL.get(sensor, HL0)); thr[f"{m0:%Y-%m}"] = {q: float(np.quantile(hv, q)) for q, _ in NIVEIS.values()}
        sc = np.flatnonzero(okw[i0:i1]) + i0
        if len(sc):
            err[sc] = mae(mod, z(sc))
    hi = ewma_hi(err, HL.get(sensor, HL0))
    return sensor, semente, err, hi, thr, all(GL.get("_v1", [True]))


def niveis(H, T, idx):
    """Série de cada nível: >= k sensores com health >= o limiar do mês vigente."""
    mes = idx.tz_localize(None).to_period("M").astype(str); out = {}
    for nome, (q, k) in NIVEIS.items():
        acima = np.zeros(len(idx), int)
        for s in H:
            lim = np.array([T[s].get(m, {}).get(q, np.inf) for m in mes]); acima += (np.nan_to_num(H[s], nan=-np.inf) >= lim).astype(int)
        out[nome] = pd.Series(acima >= k, index=idx)
    return out


def regua_secundaria(H, T, idx, vig):
    d = pd.read_csv(ALARMES); d["t"] = pd.to_datetime(d["Data da Ocorrência"]).dt.tz_localize("UTC")
    d = d[(d["Status"] == "ACT/UNACK") & d["Condição do Alarme"].isin(["HI", "HIHI"]) & (d.t >= P.INICIO) & (d.t <= P.FIM)]
    op = GL["op"]; d = d[op.reindex(d.t.dt.floor("2min")).fillna(False).to_numpy()]
    mes = idx.tz_localize(None).to_period("M").astype(str); rec, duty = {}, {}
    for s in H:
        lim = np.array([T[s].get(m, {}).get(0.90, np.inf) for m in mes]); acima = pd.Series(np.nan_to_num(H[s], nan=-np.inf) >= lim, index=idx)
        duty[s] = float(acima[vig].mean())
        ts = sorted(d[d["Tag Alarme"].str.endswith(s)].t); inc = [t for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > pd.Timedelta(hours=4)]
        if inc:
            rec[s] = (sum(bool(acima.loc[t - pd.Timedelta(hours=8):t].any()) for t in inc), len(inc))
    return rec, duty


def main():
    t0 = time.time(); SAIDA.mkdir(parents=True, exist_ok=True)
    g = nvio.carrega_grade(); S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); idx = g.index[(g.index >= g.index[0]) & (g.index <= P.FIM)]
    vig = S["mask"].astype(bool).reindex(idx).fillna(False).to_numpy(); trips = sorted(carrega_registro())
    def longe_de(m0):
        z = np.ones(len(idx), bool)
        for t in trips:
            if t < m0:
                z &= ~((idx >= t - JANELA) & (idx <= t + pd.Timedelta(days=2)))
        return z
    GL.update(g=g.loc[idx], vig=vig, longe_de=longe_de, idx=idx, op=(g["RUNNING_A"] > 0.5).fillna(False))
    sens = SENSORES[:2] if TESTE else SENSORES
    with Pool(int(os.environ.get("SOTA_PROCS", 8))) as pool:
        RES = list(pool.imap_unordered(um_sensor, [(s, sd) for sd in SEMENTES for s in sens], chunksize=1))
    v1 = all(r[5] for r in RES); print(f"V1 — nenhum treino usa o mês avaliado nem janela de trip conhecida; limiares só da validação -> {'PASSA' if v1 else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    if not v1:
        print("verificação falhou — parando.", flush=True); return
    al_pca, vig_c = carrega_controle("atual"); alvo = P.trips_alvo(trips); vigS = pd.Series(vig, index=idx); linhas = []
    for sd in SEMENTES:
        H = {r[0]: r[3] for r in RES if r[1] == sd}; T = {r[0]: r[4] for r in RES if r[1] == sd}; N = niveis(H, T, idx)
        for nome, al in N.items():
            m = P.mede(al, vig_c, alvo); c = P.ic(al, vig_c, alvo, n=400 if TESTE else P.N_BOOT); d = P.compara(al, al_pca, vig_c, alvo, n=400 if TESTE else P.N_BOOT)
            linhas.append(dict(semente=sd, nivel=nome, **m, ic_det=c["det"], ic_fp_mes=c["fp_mes"], d_det=d["d_det"], ic_d_det=d["ic_d_det"], d_fp_mes=d["d_fp_mes"],
                               ic_d_fp_mes=d["ic_d_fp_mes"], d_horas_mes=d["d_horas_mes"], difere=(d["difere_det"], d["difere_fp"], d["difere_horas"])))
        rec, duty = regua_secundaria(H, T, idx, vigS); GL.setdefault("SEC", {})[sd] = (rec, duty)
    Tb = pd.DataFrame(linhas); GL.update(Tb=Tb, mpca=P.mede(al_pca, vig_c, alvo))
    if TESTE:
        print(f"TESTE: encanamento ok em {time.time() - t0:.0f} s ({len(RES)} sensor-semente; {len(Tb)} linhas; saída suprimida).", flush=True); return
    relata(); Tb.to_csv(SAIDA / "ae_walkforward.csv", index=False); print(f"\ntempo total {time.time() - t0:.0f} s", flush=True)


def relata():
    Tb, mp = GL["Tb"], GL["mpca"]
    print(f"\nPCA v2 (base atual): {mp['det']}/8 | início {mp['inicio']} | FP {mp['fp']} ({mp['fp_mes']:.3f}/mês) | {mp['horas']:.0f} h | F1 {mp['f1']:.2f}", flush=True)
    print("17 AEs CNN-1D em walk-forward mensal, por nível e semente (comparação pareada contra o PCA v2)", flush=True)
    for r in Tb.itertuples():
        print(f"   semente {r.semente} {r.nivel:8s} {r.det}/8 [{r.ic_det[0]:.0f}-{r.ic_det[1]:.0f}] início {r.inicio} banda {r.banda} | FP {r.fp:3d} ({r.fp_mes:.3f}/mês) | {r.horas:6.0f} h "
              f"({r.horas_mes:.0f}/mês) | lead mediano {r.lead_mediano:.1f} h ({r.n_censurado} cens.) | F1 {r.f1:.2f} | Δdet {r.d_det:+d} ΔFP/mês {r.d_fp_mes:+.2f} Δh/mês {r.d_horas_mes:+.0f} "
              f"| difere (det, FP, horas) {r.difere}", flush=True)
    for sd, (rec, duty) in GL["SEC"].items():
        mac = np.mean([a / b for a, b in rec.values()]) if rec else np.nan
        print(f"   régua secundária, semente {sd}: recall macro {mac:.2f} em {len(rec)} sensores com incidente ({sum(b for _, b in rec.values())} incidentes) | "
              f"fração do tempo acima do q 0,90: mediana {np.median(list(duty.values())):.2f}, máx {max(duty.values()):.2f} ({max(duty, key=duty.get)})", flush=True)
    al = Tb[Tb.nivel == "alarm"].sort_values("fp"); r = al.iloc[len(al) // 2]
    perto = r.det >= mp["det"] - 1 and r.fp_mes <= 1.5 * mp["fp_mes"] and r.horas <= 1.5 * mp["horas"]
    print(f"\nCLASSIFICAÇÃO (nível ALARM, semente mediana em FP = {r.semente}): {'PERTO DO PCA v2' if perto else 'NÃO CHEGA PERTO — registrar e seguir'} "
          f"({r.det}/8 contra {mp['det']}/8; FP/mês {r.fp_mes:.3f} contra {mp['fp_mes']:.3f}; horas {r.horas:.0f} contra {mp['horas']:.0f})", flush=True)


if __name__ == "__main__":
    main()
