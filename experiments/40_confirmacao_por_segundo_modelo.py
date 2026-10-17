#!/usr/bin/env python3
"""EXP 40 — UM TERCEIRO ELEMENTO 'C' = OUTRO MODELO (OCSVM e outros) COMO CONFIRMAÇÃO (E) DO ALARME DO v2, PARA REDUZIR FALSOS POSITIVOS.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Ideia do usuário: além dos níveis A e B, um C que seja outro modelo (OCSVM). Em lógica de alarmes, um segundo detector independente como CONFIRMAÇÃO (E) só reduz FP se (i) os FP do C forem independentes dos FP do v2 e (ii) o C concordar mais com os acertos do que com os FP. O EXP 23 (quinto canal, OU) só acrescentou alarmes; o EXP 20/33/34 mostraram que o OCSVM sozinho não supera o PCA; mas a confirmação em E não foi testada. A ideia é coerente: o desenho abaixo a testa sem rótulos na regra e sem ajuste.
O QUE SE FAZ (8 eventos; os 18 episódios do detector publicado, sinais de produção, `ci.prever` intocado): para cada modelo C, o escore por família (t e p) em cache (`escores_fiel8`, harness fiel, retreino mensal causal, n = 20.000), EWMA de 1 h vigiado; NORMALIZAÇÃO CAUSAL: dividido pelo percentil 95 dos escores vigiados dos 60 dias ANTERIORES ao mês (sem rótulo); C(t) = o máximo dos dois canais normalizados. CONFIRMAÇÃO: o episódio do v2 é CONFIRMADO se max C na janela [nascimento - 1 h, nascimento + 6 h] >= 1,0 (acima do próprio p95 de referência); os não confirmados são descartados. Candidato PRINCIPAL: OCSVM (a sugestão do usuário); secundários: GMM e kNN (3 testes, correção: o p exigido dos secundários é 0,05/3).
MEDIDAS: o AUC (FP contra trip) do max C na janela; quantos FP e quantos trips são confirmados; detecção de 8 e FP depois da regra. PERMUTAÇÃO: a série C deslocada circularmente por 2.000 deslocamentos aleatórios de >= 14 dias; a estatística (FP descartados - trips descartados) com a mesma regra; p = P(ao acaso >= observado).
CRITÉRIOS (fixados antes). ÚTIL = após a regra a detecção fica >= 7/8 e os FP <= 6, e o p da permutação <= 0,05 (o principal) ou <= 0,017 (os secundários). NOCIVO = perde detecção sem cortar FP.
EXPECTATIVAS: o AUC do OCSVM entre FP e trips fica entre 0,35 e 0,65 (sem separação) ~65%; o C confirma >= 6 dos 9 trips e >= 6 dos 9 FP (os dois têm excursões de várias horas) ~55%; algum modelo ÚTIL ~8%.
INTERPRETAÇÃO PREVISTA. Se o OCSVM concordar com os acertos e discordar dos FP além do acaso, o A-B-C com confirmação é uma estrutura coerente para reduzir FP, a confirmar na sombra. Se não separar, os FP do v2 são excursões reais do padrão, que outro modelo de normalidade também vê (um FP 'verdadeiro' do ponto de vista estatístico, que só o desfecho distingue).
LIMITES. 18 episódios (9 FP e 9 de trip): o poder é baixo; o limiar do C fixado em 1,0 (p95); a janela de confirmação fixa; o C pode sofrer o mesmo viés de borda; nada se adota sem sombra.

RESULTADO (16/10/2026) -- NENHUM SEGUNDO MODELO (OCSVM, GMM, kNN) CONFIRMA OS ACERTOS MAIS QUE OS FP: a confirmação em E remove FP e trips na mesma taxa.
  (Correção de tratamento, feita depois da 1.ª execução e registrada: os 2 primeiros episódios de trip (fev/2025) não têm C, porque a normalização causal precisa de 60 dias anteriores; na 1.ª execução o NaN >= 1,0 era falso e eles eram descartados como 'não confirmados'; na versão final os episódios sem C são mantidos. O AUC não muda; a detecção depois da regra passa de 3/8 a 4/8.)
  C = OCSVM (o candidato principal): AUC(trip > FP) do max C na janela = 0,60; confirma 4 de 9 FP (44%) e 3 de 7 trips com C definido (43%; 2 sem C, mantidos); depois da regra: 4/8, 4 FP; permutação P(estatística >= 1) = 0,43 (limite 0,05): NEUTRO/INCONCLUSIVO. C = GMM: AUC 0,24 (os FP têm max C MAIOR que os trips); confirma 7 de 9 FP e 4 de 7 trips; depois da regra 4/8, 7 FP; P = 0,71: NEUTRO. C = kNN: AUC 0,41; confirma 5 de 9 FP e 3 de 7 trips; 4/8, 5 FP; P = 0,60: NEUTRO. Max C do OCSVM nos FP: 16,4; 0,05; 2,5; 1,1; 1,1; 0,7; 0,9; 1,0; 0,7; nos trips: 3,8; 0,9; 0,6; 16,9; 1,0; 0,8; 2,5.
LEITURA. (1) Os FP do v2 são excursões que outro modelo de normalidade também enxerga na mesma proporção dos acertos: o C não discrimina (AUC de 0,24 a 0,60, p da permutação de 0,43 a 0,71). Para cortar, por exemplo, 5 dos 9 FP com o OCSVM, a regra descarta 3 de 7 acertos: 4/8 com 4 FP. (2) Com este, seis testes independentes tentaram separar FP de acerto no disparo com informação além do voto: o padrão de canais (EXP 36), o transiente de carga e os sensores travados (37/37b), os alarmes analógicos, de falha de instrumento e de gás da planta (38/39b) e um segundo modelo (40). Nenhum separa: no disparo, com a informação disponível, os 9 FP e os 9 acertos são indistinguíveis; a única diferença é o desfecho (trip em 48 h ou não). (3) Isso muda a pergunta sobre 'reduzir FP': a terceira via só ajuda se trouxer informação sobre o DESFECHO que A e B não têm: o que falta são as tags de nível (os 4 ausentes causam 7 dos 11 trips: [[sete-tags-nivel-de-trip]]) e a resposta da operação; mais um modelo de normalidade não é essa informação.
PLACAR: ACERTEI: o AUC do OCSVM entre 0,35 e 0,65 (~65%: 0,60); nenhum ÚTIL (~92%). ERREI: o C confirma >= 6 dos 9 trips e >= 6 dos 9 FP (~55%: confirma 3 de 7 trips e 4 de 9 FP com o OCSVM).

Uso:  PYTHONPATH=src .venv/bin/python experiments/40_confirmacao_por_segundo_modelo.py
"""
from __future__ import annotations
import contextlib, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG
from nv.politica import ewma_vigiado

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48); MODELOS = ("ocsvm", "gmm", "knn")


def serie_c(nome, vig_g, idx):
    zs = []
    for c in ("t", "p"):
        s = pd.read_parquet(ROOT / "escores_fiel8" / f"{c}_{nome}_20000.parquet")["s"]; e = ewma_vigiado(s, vig_g, "1h").reindex(idx); z = pd.Series(np.nan, index=idx); v = vig_g.reindex(idx).fillna(False).to_numpy(bool)
        for m0 in pd.date_range(INICIO8.replace(day=1), idx[-1], freq="MS", tz="UTC"):
            m1 = m0 + pd.offsets.MonthBegin(1); a = idx.searchsorted(m0 - pd.Timedelta(days=60)); b = idx.searchsorted(m0); x = e.to_numpy()[a:b][v[a:b]]; x = x[np.isfinite(x)]
            if len(x) >= 2000:
                z.loc[m0:m1 - pd.Timedelta("2min")] = e.loc[m0:m1 - pd.Timedelta("2min")] / float(np.quantile(x, 0.95))
        zs.append(z)
    return pd.concat(zs, axis=1).max(axis=1, skipna=True)


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index; vig_g = vig_c.reindex(g.index).fillna(False)
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); eps = episodios_de_serie(al.loc[INICIO8:]); tipo = np.array([("trip" if any(a <= t and b >= t - JAN for t in reg) else "FP") for a, b in eps]); vig = vig_c.reindex(al_c.index).fillna(False)
    pub = avalia(al.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8); print(f"PUBLICADO: {pub.det}/8, FP {pub.fp}; {len(eps)} episódios ({(tipo == 'FP').sum()} FP, {(tipo == 'trip').sum()} trips)", flush=True)
    nasc = np.array([idx.searchsorted(a) for a, b in eps]); rng = np.random.default_rng(20261016); i_ini = idx.searchsorted(INICIO8); i_fim = len(idx); L = i_fim - i_ini
    for nome in MODELOS:
        C = serie_c(nome, vig_g, idx).to_numpy(); janela = lambda arr, ii: np.array([np.nanmax(arr[max(i - 30, 0):i + 181]) if np.isfinite(arr[max(i - 30, 0):i + 181]).any() else np.nan for i in ii])
        cep = janela(C, nasc); conf = (cep >= 1.0) | ~np.isfinite(cep)   # sem C (os 2 episódios de fev/2025, antes dos 60 d de referência): desconhecido, mantido
        fp = tipo == "FP"; tr = tipo == "trip"; auc = float(mannwhitneyu(cep[tr][np.isfinite(cep[tr])], cep[fp][np.isfinite(cep[fp])]).statistic / (np.isfinite(cep[tr]).sum() * np.isfinite(cep[fp]).sum()))
        fp_desc = int((fp & ~conf).sum()); tr_desc = int((tr & ~conf).sum()); obs = fp_desc - tr_desc; sh = []
        for _ in range(2000):
            off = int(rng.integers(14 * 720, L - 14 * 720)); C2 = C.copy(); C2[i_ini:] = np.roll(C[i_ini:], off); cc2 = janela(C2, nasc); c2 = (cc2 >= 1.0) | ~np.isfinite(cc2); sh.append(int((fp & ~c2).sum()) - int((tr & ~c2).sum()))
        p = float(np.mean(np.array(sh) >= obs)); al2 = al.copy()
        for (a, b), x in zip(eps, conf):
            if not x:
                al2.loc[a:b] = False
        r = avalia(al2.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8); lim = 0.05 if nome == "ocsvm" else 0.05 / 3
        print(f"\nC = {nome}: AUC(trip > FP) do max C na janela = {auc:.2f} | confirma {int((fp & conf).sum())} de {fp.sum()} FP e {int((tr & conf & np.isfinite(cep)).sum())} de {int((tr & np.isfinite(cep)).sum())} trips com C definido ({int((~np.isfinite(cep)).sum())} sem C, mantidos) | depois da regra: {r.det}/8, FP {r.fp} | permutação P(estatística >= {obs}) = {p:.3f} (média ao acaso {np.mean(sh):.2f}; limite {lim:.3f})", flush=True)
        print(f"   max C por episódio (FP): {[round(float(x), 2) for x in cep[fp]]} | (trips): {[round(float(x), 2) for x in cep[tr]]}", flush=True)
        print(f"   -> {'ÚTIL' if (r.det >= 7 and r.fp <= 6 and p <= lim) else 'NOCIVO' if (r.det < pub.det and r.fp >= pub.fp) else 'NEUTRO/INCONCLUSIVO'}", flush=True)


if __name__ == "__main__":
    main()
