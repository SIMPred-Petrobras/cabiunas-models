#!/usr/bin/env python3
"""SIMPred / Cabiúnas — constrói o bundle do detector do TC-33003A.

RODA NO AMBIENTE DE TREINO, não em produção. Lê o histórico e escreve em
`modelos/model_<ini>_<fim>_PCA4SINAIS/` os artefatos que o `cabiunas_inference.py`
consome. É **obrigatório rodar todo mês** — ver "cadência" abaixo.

O QUE ELE RESOLVE. O detector de pesquisa usa uma classe nossa (`ScorerMax`) para
pontuar o erro de reconstrução do PCA. Um pickle dela só abre com a nossa
biblioteca instalada — exatamente o problema que a Transpetro já teve na v1 do
deploy e corrigiu na v2. Aqui o bundle guarda **só objetos sklearn puros**
(`RobustScaler`, `PCA`) mais JSON; a aritmética que a `ScorerMax` fazia por cima
deles está escrita explicitamente no `cabiunas_inference.py`, à vista.

CADÊNCIA DE RETREINO: MENSAL, e isso não é preferência.
Os canais `t` e `p` são erro de reconstrução de PCA contra um baseline. PCA em
baseline velho descola conforme o ponto de operação anda (campanha, carga,
ambiente) e o resíduo cresce por motivo que não é saúde da máquina. Medido:

    cadência      ajustes   detecção   FP/mês   h/mês
    mensal             27       8/8      0,517    7,15
    trimestral         10       7/8      0,431    4,48
    semestral           5       6/8      0,431   16,94
    congelado           1       6/8      0,861  108,19

Congelar custa duas detecções e **quinze vezes** as horas de alarme falso. Se o
retreino mensal falhar, o detector degrada em silêncio — por isso o
`cabiunas_inference.py` recusa um bundle vencido em vez de seguir adiante.

Uso:
    python3 constroi_bundle.py --historico grade2min.parquet --mes 2026-04
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# ── Tags, por família ────────────────────────────────────────────────────────
TEMPERATURA = ["954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
               "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
               "954005_624_TI_0301", "TC382_01_A", "TC382_02_A", "TC382_03_A",
               "TC382_04_A", "TC382_05_A", "TC382_06_A", "T5_AVG_A"]
PRESSAO = ["954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
           "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
           "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
           "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301"]
VIBRACAO = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
            "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]
MANCAL_ALVO = "954005_624_TI_0305"                       # radial LNA
MANCAL_IRMAOS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]

FIT_POINTS = 20_000        # ~28 dias de operação estável
N_COMPONENTS = 0.95        # fração de variância retida
PHI = 0.10                 # piso do normalizador por sensor (ver abaixo)


def spread_mancal(X: pd.DataFrame) -> pd.Series:
    """Divergência do mancal alvo contra a mediana dos três irmãos.

    COM SINAL: o valor absoluto é aplicado depois, sobre o z-score, nunca sobre o
    spread. Trocar a ordem muda o resultado."""
    return X[MANCAL_ALVO] - X[MANCAL_IRMAOS].median(axis=1)


def ajusta_familia(base: pd.DataFrame, cols: list[str]) -> dict:
    """RobustScaler + PCA + os dois normalizadores, tudo sklearn puro.

    O score é o MÁXIMO do erro quadrático por sensor, cada um normalizado pelo
    seu p99 no baseline. O piso `PHI * mediana(p99)` é indispensável: sem ele o
    sensor mais quieto do baseline recebe um normalizador minúsculo e passa a
    dominar o máximo o tempo inteiro."""
    X = base[cols].dropna()
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(Xs)
    err = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2
    p99 = np.nanpercentile(err, 99, axis=0)
    sens_p99 = np.maximum(p99, PHI * np.nanmedian(p99))
    recon = np.max(err / sens_p99, axis=1)
    return dict(scaler=scaler, pca=pca, cols=list(cols),
                sens_p99=sens_p99.tolist(),
                recon_p99=float(np.nanpercentile(recon, 99)),
                n_fit=int(len(X)), n_componentes=int(pca.n_components_))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historico", default="grade2min.parquet",
                    help="grade de 2 min com todas as tags (índice de timestamp UTC)")
    ap.add_argument("--mes", required=True, help="mês a servir, YYYY-MM")
    ap.add_argument("--saida", default=None, help="pasta modelos/ (padrão: ../modelos)")
    ap.add_argument("--trips", default="falhas.csv",
                    help="CSV com uma coluna `evento`: os trips JÁ OCORRIDOS")
    a = ap.parse_args()

    corte = pd.Timestamp(a.mes + "-01", tz="UTC")
    g = pd.read_parquet(a.historico)
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = op & (g["T5_AVG_A"] > 300)

    # O baseline são os últimos FIT_POINTS pontos ESTÁVEIS ANTERIORES ao mês.
    # Anteriores, não "em torno": olhar para depois do corte vazaria o futuro.
    todas = TEMPERATURA + PRESSAO + VIBRACAO
    # float64 explícito: o histórico chega em float32 e a mediana/MAD do baseline
    # viram constantes gravadas no bundle. Em float32 o MAD oscila na 8ª casa entre
    # execuções da mesma entrada — irrelevante para o alarme, mas um artefato de
    # produção não deve depender da precisão com que o parquet foi escrito.
    base = (g.loc[estavel & (g.index < corte), todas]
            .dropna().tail(FIT_POINTS).astype("float64"))
    if len(base) < FIT_POINTS // 4:
        print(f"ERRO: só {len(base)} pontos estáveis antes de {a.mes}; "
              f"mínimo {FIT_POINTS//4}. Bundle não gerado.")
        return 1

    ini, fim = base.index[0], base.index[-1]
    print(f"baseline: {len(base)} pontos estáveis  {ini:%Y-%m-%d} .. {fim:%Y-%m-%d}"
          f"  ({(fim-ini).days} d de calendário)")

    ft = ajusta_familia(base, TEMPERATURA)
    fp = ajusta_familia(base, PRESSAO)
    print(f"  temperatura: {ft['n_componentes']} componentes, recon_p99 {ft['recon_p99']:.4f}")
    print(f"  pressão    : {fp['n_componentes']} componentes, recon_p99 {fp['recon_p99']:.4f}")

    b = spread_mancal(base)
    med = float(b.median())
    mad = float((b - b.median()).abs().median() * 1.4826)
    print(f"  spread do mancal: mediana {med:.2f} °C, MAD robusto {mad:.3f} °C")

    saida = Path(a.saida) if a.saida else Path(__file__).resolve().parent.parent / "modelos"
    dest = saida / f"model_{ini:%Y-%m-%d}_{fim:%Y-%m-%d}_PCA4SINAIS"
    dest.mkdir(parents=True, exist_ok=True)

    for nome, f in (("temperatura", ft), ("pressao", fp)):
        with open(dest / f"{nome}_scaler.pkl", "wb") as fh:
            pickle.dump(f["scaler"], fh)
        with open(dest / f"{nome}_pca.pkl", "wb") as fh:
            pickle.dump(f["pca"], fh)

    (dest / "normalizacao.json").write_text(json.dumps({
        "phi": PHI, "n_components": N_COMPONENTS,
        "temperatura": {k: ft[k] for k in ("cols", "sens_p99", "recon_p99",
                                           "n_fit", "n_componentes")},
        "pressao": {k: fp[k] for k in ("cols", "sens_p99", "recon_p99",
                                       "n_fit", "n_componentes")},
    }, indent=2), encoding="utf-8")

    (dest / "spread_mancal.json").write_text(json.dumps({
        "tag_alvo": MANCAL_ALVO, "tags_irmaos": MANCAL_IRMAOS,
        "mediana": med, "mad_robusto": mad,
        "comentario": "z = |(spread - mediana) / mad|; o abs vai no z, nunca no spread",
    }, indent=2), encoding="utf-8")

    (dest / "modelo.json").write_text(json.dumps({
        "equipamento": "TC-33003A", "arquitetura": "PCA4SINAIS",
        "versao_detector": "v2 (gatilho de dois níveis) + ponto de deploy",
        "mes_servido": a.mes,
        "baseline_inicio": f"{ini:%Y-%m-%dT%H:%M:%S%z}",
        "baseline_fim": f"{fim:%Y-%m-%dT%H:%M:%S%z}",
        "baseline_pontos": int(len(base)),
        "validade_dias": 62,
        "cadencia_retreino": "mensal (obrigatória; congelar custa 2 detecções e 15x as horas de FP)",
        "gerado_por": "constroi_bundle.py",
    }, indent=2), encoding="utf-8")

    # ── Os trips já ocorridos entram no bundle, e não é detalhe ──
    # A referência rolante do `vb` apaga ±7 d em torno de cada trip conhecido:
    # degradação que já se sabe que terminou em falha não pode virar o "normal"
    # contra o qual a próxima é medida. Em produção isso é legítimo — o trip já
    # aconteceu e está no registro. (Num LOEO seria vazamento, porque lá se
    # finge não conhecer o evento escondido; não é o caso aqui.)
    # Consequência operacional: este arquivo precisa ser ATUALIZADO a cada trip
    # novo. Se não for, a próxima degradação entra na referência e se cancela.
    try:
        trips = pd.read_csv(a.trips, parse_dates=["evento"])["evento"]
        trips = trips.dt.tz_convert("UTC") if trips.dt.tz is not None else trips.dt.tz_localize("UTC")
        trips = sorted(t.isoformat() for t in trips if t <= fim)
        print(f"  trips conhecidos até {fim:%Y-%m-%d}: {len(trips)}")
    except (FileNotFoundError, KeyError) as e:
        print(f"  AVISO: registro de trips não lido ({e}); referência do vb ficará "
              f"sem exclusão e a próxima degradação vai contaminá-la.")
        trips = []

    (dest / "detector.json").write_text(json.dumps({
        "sinais": ["t", "p", "sp", "vb"],
        "halflife": {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"},
        "base": {"t": 2.0, "p": 2.0, "sp": 3.0, "vb": 3.0},

        "k_nivel_a": {"t": 1.10, "p": 1.20, "sp": 0.90, "vb": 2.00},
        "voto_nivel_a": 3,
        "k_nivel_b": {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2},
        "voto_nivel_b": 2,

        "kappa": 0.75, "h_cusum": 80, "carga_reset": 0.25,
        "refratario_h": 72,
        "duracao_min": 120, "duracao_min_forte": 60,
        "escalada_idade_h": 96, "escalada_abs": 20.0,

        "tags_vibracao": VIBRACAO,
        "vb_exclusao_dias": 7.0,
        "trips_conhecidos": trips,

        "janela_entrada_dias": 60,
        "_nota_k_nivel_a": (
            "escolhidos por margem à borda, não por ótimo nos 8 eventos. O ponto "
            "ótimo {t:1,10 p:0,70 sp:0,90 vb:1,80} empata em tudo que se mede mas "
            "fica a um passo de grade de p=0,60, onde a detecção cai de 8/8 para "
            "7/8. Em 1,20 a margem até essa borda é 2,00x."),
        "_nota_janela": (
            "60 d não é folga arbitrária: medido contra o histórico completo em 61 "
            "execuções semanais, 45 d reproduz bit a bit e 30 d diverge em até "
            "17,1% dos pontos. Janela curta não dá erro, dá alarme diferente."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n-> {dest}")
    for f in sorted(dest.iterdir()):
        print(f"     {f.name:28s} {f.stat().st_size/1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
