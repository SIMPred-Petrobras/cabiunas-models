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
    python3 constroi_bundle.py --historico ../dados/2025_2026/data_*_raw.csv --mes 2026-04
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

# Portões do baseline. O baseline são os últimos FIT_POINTS pontos estáveis de TODO o passado,
# então só fica com menos de FIT_POINTS no começo do histórico. Um mês com muita parada NÃO
# o encolhe: faz os 20.000 pontos recuarem no calendário. O que denuncia isso é o espalhamento
# (span) e a folga até o corte, e não a contagem. Histórico dos 16 bundles: span 27 a 64 d,
# folga 0 a 3 d. Os limites ficam logo acima do máximo visto: alerta, não bloqueio -- um bundle
# com baseline velho ainda é melhor que seguir com o do mês anterior, que é mais velho.
#
# O QUE O ALERTA DIZ E O QUE NÃO DIZ. Marca SAÍDA DA FAIXA VALIDADA (todo número publicado vem de
# baselines de 27 a 64 d), não "bundle ruim": em 16 meses nem o span nem a folga predizem a
# carga do mês (Spearman |rho| <= 0,33, n.s.; `valida_alerta_p99.py` no repositório de pesquisa).
# Pelo mesmo teste NÃO há alerta de `recon_p99`: o salto mês a mês vai de -30% a +30% no
# normal (quantis 10-90%: 0,71 a 1,33) e não prediz a carga (rho +0,07 / +0,15, p 0,8 / 0,6).
SPAN_MAX_DIAS = 70
FOLGA_MAX_DIAS = 7
COD_ALERTA = 4             # bundle gerado, mas fora da faixa histórica: revisar antes de publicar


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


def _ler_historico(caminho) -> pd.DataFrame:
    """A grade de 2 min, de parquet, do CSV que vai no Drive ou do export de 30 s.

    O parquet é o formato do ambiente de treino. O CSV é o que a pasta do
    equipamento entrega (`dados/<ini>_<fim>/data_<ini>_<fim>_raw.csv`), e sem
    aceitá-lo este script não roda para quem só tem a pasta — que é justamente
    quem precisa retreinar todo mês.

    Conferido: bundle gerado do CSV é idêntico byte a byte ao gerado do parquet
    (sha256 dos 8 artefatos, pickles inclusive)."""
    c = Path(caminho)
    if not c.exists():
        raise FileNotFoundError(f"histórico não encontrado: {c}")
    if c.suffix.lower() in (".parquet", ".pq"):
        g = pd.read_parquet(c)
    elif c.suffix.lower() == ".csv":
        g = pd.read_csv(c, index_col=0, parse_dates=[0])
    else:
        raise ValueError(f"formato não suportado: {c.suffix} (use .parquet ou .csv)")
    if g.index.tz is None:
        g.index = g.index.tz_localize("UTC")
    else:
        g.index = g.index.tz_convert("UTC")
    # A MESMA grade que a inferência monta (`cabiunas_inference.preparar_grade`):
    # texto do PI -> NaN, faixa física, e o export de 30 s reduzido à mediana de
    # 2 min. Treinar sobre uma grade e servir sobre outra custa 0,344 -> 0,603
    # FP/mês sem mudar a detecção. Uma função só, nos dois lados.
    from cabiunas_inference import preparar_grade
    # float32 como o parquet: o CSV volta em float64 e mudaria os pickles.
    return preparar_grade(g.sort_index()).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historico", default="grade2min.parquet",
                    help="grade de 2 min (.parquet ou .csv) ou o export de 30 s do PI (.csv); índice de timestamp UTC")
    ap.add_argument("--mes", required=True, help="mês a servir, YYYY-MM")
    ap.add_argument("--saida", default=None, help="pasta modelos/ (padrão: ../modelos)")
    ap.add_argument("--trips", default="falhas.csv",
                    help="CSV com uma coluna `evento`: os trips JÁ OCORRIDOS")
    a = ap.parse_args()

    # O CORTE E SEMPRE NO DIA 1, e isso NAO e convencao -- e requisito medido.
    # O baseline sao os 20.000 pontos estaveis anteriores ao corte; mover o corte
    # muda QUAIS 20.000, e o detector e extremamente sensivel a isso. Medido
    # (drift_composicao.py), so deslocando o dia do retreino dentro do mes:
    #
    #     dia  1   banda 5/8  det 8/8  0,344 FP/mes    6,6 h/mes   <- o publicado
    #     dia  8   banda 3/8  det 5/8  0,947 FP/mes   77,3 h/mes
    #     dia 15   banda 4/8  det 7/8  0,947 FP/mes  154,6 h/mes
    #     dia 22   banda 5/8  det 7/8  0,689 FP/mes   76,9 h/mes
    #
    # Vinte e tres vezes as horas de alarme falso por mudar o dia do mes em que o
    # retreino roda. `--mes AAAA-MM` ja garante o dia 1 venha o operador a rodar
    # quando vier -- NAO trocar por data corrente, nem por "ultimos 30 dias".
    corte = pd.Timestamp(a.mes + "-01", tz="UTC")
    g = _ler_historico(a.historico)
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
    if len(base) < FIT_POINTS:
        print(f"ERRO: só {len(base)} pontos estáveis antes de {a.mes}; o validado é {FIT_POINTS}. "
              f"Bundle não gerado; a inferência segue com o anterior até ele vencer.")
        return 1

    ini, fim = base.index[0], base.index[-1]
    span, folga = (fim - ini).days, (corte - fim).days
    print(f"baseline: {len(base)} pontos estáveis  {ini:%Y-%m-%d} .. {fim:%Y-%m-%d}"
          f"  ({span} d de calendário, termina {folga} d antes do corte)")
    alertas = []
    if span > SPAN_MAX_DIAS:
        alertas.append(f"baseline espalhado por {span} d de calendário (histórico: 27 a 64 d, "
                       f"limite {SPAN_MAX_DIAS}): a máquina passou muito tempo parada ou fora de regime")
    if folga > FOLGA_MAX_DIAS:
        alertas.append(f"o baseline termina {folga} d antes do corte (histórico: 0 a 3 d, "
                       f"limite {FOLGA_MAX_DIAS}): sem regime estável recente")

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

    # A transformação vai em JSON, não em pickle. Pickle é código fechado: não se
    # lê, não se compara, exige a versão certa do sklearn e executa código ao
    # abrir. Os objetos guardavam só estes quatro vetores — conferido:
    # with_centering/with_scaling=True, unit_variance=False, whiten=False.
    #     Xs  = (X - center) / scale
    #     Z   = Xs @ componentsᵀ - mean_proj      (mean_proj = mean @ componentsᵀ)
    #     rec = Z @ components + mean
    # A ordem é a do sklearn, que centraliza DEPOIS de projetar; por isso o
    # mean_proj vai pronto. Inverter isso muda o resultado em ~6e-14.
    # Assim o bundle é 100% JSON e o dashboard reimplementa em qualquer linguagem.
    for nome, f in (("temperatura", ft), ("pressao", fp)):
        s, pc = f["scaler"], f["pca"]
        assert s.with_centering and s.with_scaling and not s.unit_variance
        assert not pc.whiten, "PCA com whiten=True mudaria a aritmética publicada"
        (dest / f"{nome}_transformacao.json").write_text(json.dumps({
            "formato": "robustscaler+pca, explicito",
            "aritmetica": [
                "Xs = (X - center) / scale",
                "Z  = Xs @ components.T - mean_proj",
                "rec = Z @ components + mean",
                "ORDEM IMPORTA: o sklearn centraliza DEPOIS de projetar, e por isso",
                "mean_proj = mean @ components.T ja vem calculado. Escrever",
                "(Xs - mean) @ components.T e algebricamente igual mas difere em",
                "~6e-14 -- suficiente para uma discussao de numero divergente."
            ],
            "cols": f["cols"],
            "center": [float(v) for v in s.center_],
            "scale": [float(v) for v in s.scale_],
            "mean": [float(v) for v in pc.mean_],
            "mean_proj": [float(v) for v in (np.reshape(pc.mean_, (1, -1)) @ pc.components_.T)[0]],
            "components": [[float(v) for v in linha] for linha in pc.components_],
        }, indent=2), encoding="utf-8")

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
        "baseline_span_dias": int(span),
        "baseline_folga_dias": int(folga),
        "alertas": alertas,
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
    for al in alertas:
        print(f"\nALERTA: {al}")
    if alertas:
        print(f"Bundle gerado mas FORA DA FAIXA HISTÓRICA: revisar antes de publicar (código {COD_ALERTA}).")
        return COD_ALERTA
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
