"""SIMPred / Cabiúnas — inferência do detector físico de 4 sinais do TC-33003A.

Módulo **autocontido**: depende só de numpy, pandas e scikit-learn. Não importa
nenhuma biblioteca interna nossa. Os quatro passos do contrato SIMPred:

    carregar_dados → preprocessar → carregar_modelo → prever

O QUE O DETECTOR É. Quatro sinais físicos, cada um com uma leitura direta:

    t   erro de reconstrução do PCA sobre 14 tags de temperatura
    p   erro de reconstrução do PCA sobre 12 tags de pressão
    sp  z robusto do spread do mancal radial LNA contra seus três irmãos
    vb  maior z robusto entre as 10 sondas de vibração, contra uma referência
        rolante de 400 h da própria sonda

Cada sinal acende por **degrau sustentado** (30 min acima do limiar) ou por
**CUSUM** (acúmulo lento que nunca chega a fazer degrau). O alarme sai de um
gatilho de DOIS NÍVEIS — um sensível e largo, um específico e estreito — porque
os quatro canais operam em percentis efetivos muito diferentes e disparam em
momentos descoordenados. Medido: nenhum nível sozinho passa de 4/8 na banda
acionável; a união faz 5/8, e ao custo do mais barato dos dois.

JANELA DE ENTRADA: 60 DIAS, e isso não é folga arbitrária.
O detector tem memória: a referência rolante do `vb` olha 400 h estáveis para
trás, o CUSUM acumula desde o último reset (corridas de até 35,7 d no histórico)
e o refratário dura 72 h. Rodar com janela curta não dá erro — dá **alarme
diferente**, em silêncio. Medido contra o histórico completo, em 61 execuções
semanais: 45 d já reproduz bit a bit; 30 d diverge em até 17,1% dos pontos.
60 d é 45 d com margem. Ver `README.md`.

Só os últimos dias da janela são resultado; o começo é aquecimento.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constantes estruturais (não são o ponto de operação — esse vem do bundle) ──
GRID = "2min"
BLACKOUT_H = 6.0       # apaga as 6 h seguintes a cada religamento
SUSTAIN = 15           # 15 amostras de 2 min = 30 min acima do limiar
GAP_EPISODIO_H = 2.0   # dois alarmes a menos de 2 h são o mesmo episódio
T5_ESTAVEL = 300.0     # abaixo disso a turbina não está em regime
POR_HORA = 30          # amostras de 2 min por hora

# Referência rolante do vb
VB_HORAS_BASE = 400.0
VB_GUARDA_H = 24.0     # a base termina 24 h antes do ponto, nunca colada nele
VB_PASSO_H = 6.0       # a referência é reavaliada a cada 6 h
EPOCA = pd.Timestamp("2024-01-01", tz="UTC")   # âncora dos blocos da referência
VB_FRACAO_BASE_MIN = 0.25   # fração mínima da base de 400 h para a sonda opinar

# Cabeça de aquecimento: quanto do INÍCIO de qualquer entrada é descartável.
# Não é margem de segurança, é medição. Pontuando 2026-03-01..2026-04-30 com 60 d
# de entrada e de novo com 484 d, comparei instante a instante:
#
#     dia  0-21 da entrada : vb difere em 100,0% dos instantes
#     dia 21-30            : vb difere em  44,9%
#     dia 30-61            : vb difere em   0,0%   (t e p, idem: 2,8% -> 0,0%)
#
# A causa é a referência rolante de 400 h: ela precisa de 400 h ESTÁVEIS, e a
# máquina fica de pé ~60% do tempo, então 400 h estáveis custam ~28 dias de
# calendário. Antes disso a sonda opina com base truncada (VB_FRACAO_BASE_MIN).
# Depois do dia 30 o resultado é idêntico, venha de 60 ou de 484 dias de entrada.
AQUECIMENTO_DIAS = 30


def corte_valido(df, dias_pedidos: int):
    """Onde a saída pode começar, e o aviso se o pedido invadiu o aquecimento.

    Devolve (corte, aviso|None). Reportar dentro dos primeiros AQUECIMENTO_DIAS
    não levanta erro — levanta número errado, que é pior. O `--dias 61` sobre um
    arquivo de 61 d cai inteiro na cabeça."""
    fim = df.index[-1]
    pedido = fim - pd.Timedelta(days=dias_pedidos)
    minimo = df.index[0] + pd.Timedelta(days=AQUECIMENTO_DIAS)
    if pedido >= minimo:
        return pedido, None
    dias_uteis = max((fim - minimo).total_seconds() / 86400.0, 0.0)
    return minimo, (
        f"pedidos {dias_pedidos} d, mas os primeiros {AQUECIMENTO_DIAS} d da "
        f"entrada são aquecimento da referência de 400 h. Reportando "
        f"{dias_uteis:.1f} d, a partir de {minimo:%Y-%m-%d %H:%M}. Para reportar "
        f"{dias_pedidos} d, alimente {dias_pedidos + AQUECIMENTO_DIAS} d de entrada.")


# ══════════════════════════════════════════════════ 1. carregar dados
def carregar_dados(csv_path) -> pd.DataFrame:
    """CSV com a 1ª coluna de timestamp (UTC); as demais são as tags."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def achar_csv(dados_dir, padrao: str = "data_*_raw.csv") -> Path:
    """O CSV de entrada mais recente em dados/.

    O nome segue a convenção das outras frentes do SIMPred —
    `dados/<ano_ini>_<ano_fim>/data_<ini>_<fim>_raw.csv` — e como a data de
    início vem logo depois do prefixo, ordenar por nome ordena por período.
    Com vários períodos na pasta, vale o último.

    NÃO cai para `*.csv` qualquer. A pasta também recebe recorte de análise e
    planilha exportada, e pegar o arquivo errado não levanta erro: levanta
    resultado. Quem quiser apontar outro arquivo usa `--csv`."""
    d = Path(dados_dir)
    achados = sorted(d.rglob(padrao))
    if not achados:
        raise FileNotFoundError(
            f"nenhum {padrao} em {d}. Esperado: "
            f"dados/<ini>_<fim>/data_<ini>_<fim>_raw.csv")
    return achados[-1]


# ══════════════════════════════════════════════════ 2. carregar modelo
def _transformacao(b: Path, fam: str) -> dict:
    """Os quatro vetores que definem a transformação da família, sem pickle.

    O bundle guardava `RobustScaler` e `PCA` em pickle. Mesmo sendo sklearn
    puro, pickle é código fechado: não se lê, não se compara entre versões, não
    se audita, exige a versão certa do sklearn (`InconsistentVersionWarning`) e
    executa código ao abrir. A recomendação do SIMPred é entregar modelo aberto.

    O que os dois objetos guardavam é isto e só isto — medido, não suposto:
    `with_centering=True`, `with_scaling=True`, `unit_variance=False`,
    `whiten=False`. Ou seja, quatro vetores e três linhas de aritmética:

        Xs  = (X - center) / scale               RobustScaler.transform
        Z   = Xs @ componentsᵀ - mean_proj       PCA.transform
        rec = Z @ components + mean              PCA.inverse_transform

    A ordem é a do sklearn, que centraliza DEPOIS de projetar; por isso o bundle
    traz `mean_proj = mean @ componentsᵀ` pronto. `(Xs - mean) @ componentsᵀ` dá
    o mesmo na álgebra e difere em ~6e-14 no ponto flutuante.

    São 56 números na temperatura e 84 na pressão. Em JSON, qualquer linguagem
    lê — o dashboard não precisa de Python nem de sklearn para reimplementar.

    O `.pkl` continua aceito para os bundles antigos, com aviso."""
    j = b / f"{fam}_transformacao.json"
    if j.exists():
        d = json.loads(j.read_text(encoding="utf-8"))
        return {k: np.asarray(d[k], dtype="float64")
                for k in ("center", "scale", "mean", "mean_proj", "components")}
    import pickle, warnings   # só no caminho legado; a inferência não usa pickle
    warnings.warn(
        f"{b.name}/{fam}: caindo para pickle ({fam}_scaler.pkl / {fam}_pca.pkl). "
        f"Rode scripts/migra_bundle_json.py para gerar {fam}_transformacao.json.",
        stacklevel=2)
    with open(b / f"{fam}_scaler.pkl", "rb") as fh:
        s = pickle.load(fh)
    with open(b / f"{fam}_pca.pkl", "rb") as fh:
        pc = pickle.load(fh)
    C = np.asarray(pc.components_, dtype="float64")
    M = np.asarray(pc.mean_, dtype="float64")
    return {"center": np.asarray(s.center_, dtype="float64"),
            "scale": np.asarray(s.scale_, dtype="float64"),
            "mean": M, "mean_proj": (np.reshape(M, (1, -1)) @ C.T)[0],
            "components": C}


def carregar_modelo(bundle_dir) -> dict:
    """Abre o bundle. Só JSON — nenhum pickle, nenhuma classe nossa."""
    b = Path(bundle_dir)
    m = {"dir": b}
    for nome in ("normalizacao", "spread_mancal", "modelo", "detector"):
        m[nome] = json.loads((b / f"{nome}.json").read_text(encoding="utf-8"))
    for fam in ("temperatura", "pressao"):
        m[f"{fam}_transformacao"] = _transformacao(b, fam)
    return m


def carregar_trips(caminho) -> list[pd.Timestamp]:
    """O registro de trips já ocorridos, lido na INFERÊNCIA — não no bundle.

    A referência rolante do `vb` apaga ±7 d em torno de cada trip conhecido:
    degradação que se sabe ter terminado em falha não pode virar o "normal"
    contra o qual a próxima é medida. Congelar essa lista dentro do bundle
    mensal seria errado de um jeito silencioso — em 27/04/2025 o bundle de abril
    (baseline até 31/03) ainda não sabia dos trips de 07/04 e 11/04, embora eles
    já tivessem acontecido. Com eles dentro da referência o MAD sobe e o z cai:
    medido, `vb` médio de 3,56 onde o valor correto era 8,63.

    O registro é do equipamento e muda quando a máquina falha, não quando o
    modelo é retreinado. Por isso vive fora do bundle e é relido a cada execução.

    Aceita CSV com coluna `evento` ou JSON com lista de instantes."""
    c = Path(caminho)
    if not c.exists():
        return []
    if c.suffix.lower() == ".json":
        bruto = json.loads(c.read_text(encoding="utf-8"))
        ts = pd.to_datetime(bruto if isinstance(bruto, list) else bruto["trips"], utc=True)
    else:
        ts = pd.to_datetime(pd.read_csv(c)["evento"], utc=True)
    return sorted(pd.Timestamp(t) for t in ts)


def carregar_modelos(modelos_dir) -> list[dict]:
    """Todos os bundles, do mais antigo ao mais novo.

    NÃO é conveniência — é correção. O bundle é mensal porque o PCA descola
    depressa (ver `checa_validade`). Pontuar os 60 dias de aquecimento com o
    bundle do mês corrente aplica o PCA de hoje a dado de dois meses atrás: o
    resíduo explode por deriva do baseline, não por saúde da máquina. Medido em
    agosto/2025, o sinal `p` suavizado chegou a divergir em 3388 do valor
    correto, e o CUSUM — que integra — carregou o erro para dentro do mês,
    deixando o canal aceso 17.326 amostras contra as 8.948 corretas.

    Cada trecho tem de ser pontuado pelo bundle que vigia nele, exatamente como
    o walk-forward do treino faz. Por isso produção guarda os bundles antigos:
    são 8 KB cada."""
    ds = sorted(p for p in Path(modelos_dir).glob("model_*_PCA4SINAIS") if p.is_dir())
    if not ds:
        raise FileNotFoundError(f"nenhum bundle em {modelos_dir}")
    ms = [carregar_modelo(d) for d in ds]
    return sorted(ms, key=lambda m: pd.Timestamp(m["modelo"]["baseline_fim"]))


def vigencia(modelos: list[dict], quando: pd.Timestamp) -> dict:
    """O bundle que vale num instante: o do MÊS SERVIDO que contém o instante.

    É o critério do walk-forward: o mês é pontuado pelo ajuste feito com dado
    ANTERIOR a ele, e o bundle de um mês só existe a partir do dia 1 desse mês.

    Uma versão anterior escolhia "o último bundle cujo baseline terminou antes do
    instante". No primeiro instante do mês as duas regras coincidem -- e é aí que
    `preprocessar` pergunta --, mas nos últimos dias do mês a regra antiga pegava o
    bundle do mês SEGUINTE, cujo baseline às vezes termina no dia 27 ou 29. Ao vivo
    isso não acontece (o bundle seguinte ainda não existe); ao reprocessar
    histórico com todos os bundles na pasta, são 9,8 dias pontuados com um modelo
    do futuro. Achado em 30/09/2026 pelo monitor de drift, que pergunta no fim do
    mês."""
    def inicio(m):
        ms = m["modelo"].get("mes_servido")
        return (pd.Timestamp(ms + "-01", tz="UTC") if ms
                else pd.Timestamp(m["modelo"]["baseline_fim"]))
    ordem = sorted(modelos, key=inicio)
    validos = [m for m in ordem if inicio(m) <= quando]
    return validos[-1] if validos else ordem[0]


def checa_validade(modelo: dict, agora: pd.Timestamp) -> str | None:
    """O bundle venceu? Devolve a mensagem de erro, ou None se está em dia.

    Não é burocracia. Os canais t e p são resíduo de PCA contra um baseline; com
    baseline velho o resíduo cresce por deriva do ponto de operação, não por
    saúde da máquina. Congelar o modelo leva a detecção de 8/8 para 6/8 e as
    horas de alarme falso de 7,15 para 108,19 por mês. Um bundle vencido deve
    PARAR a inferência, não degradá-la em silêncio."""
    fim = pd.Timestamp(modelo["modelo"]["baseline_fim"])
    dias = (agora - fim).days
    limite = int(modelo["modelo"].get("validade_dias", 62))
    if dias > limite:
        return (f"bundle vencido: baseline termina em {fim:%Y-%m-%d}, "
                f"{dias} dias atrás (limite {limite}). Rode constroi_bundle.py "
                f"para o mês corrente antes de inferir.")
    return None


# ══════════════════════════════════════════════════ 3. pré-processar
def _regrade(df: pd.DataFrame) -> pd.DataFrame:
    """Grade regular de 2 min. O detector conta amostras (SUSTAIN, blackout,
    refratário), então buraco na grade vira contagem errada.

    NÃO preenche buraco. Uma versão anterior fazia `ffill(limit=2)` para tolerar
    jitter do historiador, e isso inventava dado onde não havia: os pontos
    preenchidos caíam em transientes de parada, onde o PCA extrapola, e o
    resíduo saltava para ~1000. Esse valor entrava no EWMA e, pior, no CUSUM —
    que integra — deixando o canal `p` aceso 13.087 amostras contra as 8.948
    corretas, e o alarme falso em 63,0 h/mês contra 6,6.

    Onde o historiador não mediu, o sinal fica NaN e o instante não é pontuado.
    É o mesmo comportamento do treino, e é o certo: não medir não é "igual ao
    anterior". O `nearest` com tolerância abaixo do passo só alinha jitter — ele
    casa cada ponto da grade com a amostra real mais próxima, e não atravessa um
    buraco."""
    alvo = pd.date_range(df.index[0].ceil(GRID), df.index[-1].floor(GRID),
                         freq=GRID, tz="UTC")
    return df.reindex(alvo, method="nearest",
                      tolerance=pd.Timedelta(GRID) / 2 - pd.Timedelta("1s"))


def _faixa(tag: str) -> tuple[float, float]:
    """Faixa física por tipo de sensor. Fora dela é sentinela ou defeito, não
    medição -- o termopar aberto lê exatamente -40,5 °C. A mesma do treino."""
    if tag.startswith("TC382") or tag.startswith("T5"):
        return (-15.0, 900.0)                 # termopares de exaustão e média T5, °C
    if "_TI_" in tag or tag.startswith("TI_"):
        return (-15.0, 900.0)                 # termorresistências, °C
    if tag.startswith("TV_"):
        return (0.0, 200.0)                   # vibração
    if "_PDI" in tag or tag.startswith("PDI"):
        return (-5.0, 200.0)                  # pressão diferencial (PDI e PDIT)
    if "_PI_" in tag or tag.startswith("PI_"):
        return (-1.5, 200.0)                  # pressão manométrica
    return (-np.inf, np.inf)


def preparar_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Dado do historiador -> a grade de 2 min em que os bundles foram treinados.

    NÃO é limpeza opcional: é PARTE DO MODELO. Todo bundle foi ajustado, e todo
    número publicado foi medido, sobre uma grade montada assim a partir do export
    de 30 s do PI. Medido com este módulo, mesmo período e mesmos bundles
    (`scripts/pdm_fisico/paridade_entrada.py` no repositório de pesquisa):

        grade de 2 min montada como no treino    0,344 FP/mês   48,9 h/mês
        uma leitura de 30 s por ponto de 2 min   0,603 FP/mês   67,2 h/mês
        idem, sem o corte de faixa                0,603 FP/mês   95,7 h/mês

    A detecção é a mesma nos três -- nada denuncia a diferença, só o FP. O erro do
    PCA é elevado ao quadrado leitura a leitura: uma leitura ruim de 30 s (o
    PDI_0302 lendo -0,187 no meio de 1,379) pesa ~100x mais no canal `p` se for
    ela a escolhida para representar a janela.

    Três passos, na ordem do treino:
      1. texto do PI ("No Data", "Out of Serv"...) -> NaN;
      2. fora da faixa física do tipo de sensor (`_faixa`) -> NaN;
      3. se a entrada for mais fina que 2 min, MEDIANA de cada janela [t, t+2min),
         rotulada em t; e RUNNING_A como a FRAÇÃO da janela com a máquina ligada
         (média). A mediana de 4 leituras descarta a leitura isolada ruim; uma
         leitura só ou a média, não. Em float32, como o treino.

    IDEMPOTENTE: aplicada a uma grade já preparada devolve o mesmo dado e custa pouco
    (0,02 s em 90 dias de 2 min, contra ~1,1 s para reduzir o export de 30 s). Quem
    chama `diagnostico_entrada`, `monitor_drift` e `preprocessar` em seguida deve preparar
    UMA vez e passar o resultado às três.

    Entrada JÁ em 2 min passa pelos passos 1 e 2 e segue como veio: o pacote não
    tem como saber se ela foi montada pela mediana. Por isso o export de 30 s é a
    entrada preferida -- com ele a grade sai daqui, igual à do treino. Entrada mais
    grossa que 2 min é recusada: o detector conta amostras de 2 min."""
    X = df.apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        if c != "RUNNING_A":
            lo, hi = _faixa(c)
            if np.isfinite(lo) or np.isfinite(hi):
                X[c] = X[c].where((X[c] >= lo) & (X[c] <= hi))
    # a mediana dos intervalos, em Timedelta: `index.asi8` muda de unidade entre versões
    # do pandas (ns no 2.x, us no 3.x) e leria 2 min como 0,12 s
    passo = X.index.to_series().diff().median() if len(X) > 1 else pd.Timedelta(GRID)
    if passo > pd.Timedelta(GRID):
        raise ValueError(f"entrada com passo de {passo}, mais grossa que {GRID}: o detector "
                         f"conta amostras de {GRID}. Use o export de 30 s do PI.")
    if passo == pd.Timedelta(GRID):
        return X
    X = X.astype("float32")
    g = X.resample(GRID).median()
    if "RUNNING_A" in X:
        g["RUNNING_A"] = X["RUNNING_A"].resample(GRID).mean()
    return g.astype("float32")


def _grade(df: pd.DataFrame) -> pd.DataFrame:
    """A entrada do historiador na grade de 2 min do treino, sem buraco preenchido."""
    return _regrade(preparar_grade(df))


def _mascara(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Onde o detector tem direito de opinar.

    Três condições, e cada uma exclui um jeito diferente de errar:
      operando   — máquina parada não tem assinatura de falha
      T5 > 300   — abaixo disso não é regime, é partida ou parada
      blackout   — as 6 h após um religamento têm transiente térmico que imita
                   degradação em todos os quatro canais de uma vez
    """
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = op & (g["T5_AVG_A"] > T5_ESTAVEL)
    partida = op & ~op.shift(fill_value=False)
    n_bl = int(BLACKOUT_H * POR_HORA)
    blackout = partida.rolling(n_bl, min_periods=1).max().astype(bool)
    return (estavel & ~blackout), estavel, partida, op


def _recon_pca(modelo: dict, fam: str, X: pd.DataFrame) -> np.ndarray:
    """O score da família: MÁXIMO do erro por sensor, normalizado.

    É a aritmética que a classe `ScorerMax` fazia; escrita aqui para que o
    bundle não precise dela. Máximo, não média: uma falha que aparece em um
    sensor só some na média de catorze.

    O scaler e o PCA também estão escritos à vista, a partir dos vetores do
    `<fam>_transformacao.json` — ver `_transformacao`. Nada aqui depende de
    sklearn em tempo de inferência."""
    cfg = modelo["normalizacao"][fam]
    cols = cfg["cols"]
    sens = np.asarray(cfg["sens_p99"], dtype="float64")
    Xc = X[cols]
    ok = Xc.notna().all(axis=1).to_numpy()
    out = np.full(len(Xc), np.nan)
    if ok.any():
        tr = modelo[f"{fam}_transformacao"]
        Xs = (Xc[ok].astype("float64").to_numpy() - tr["center"]) / tr["scale"]
        Z = Xs @ tr["components"].T                         # PCA.transform:
        Z = Z - tr["mean_proj"]                             #   centraliza DEPOIS
        rec = Z @ tr["components"] + tr["mean"]             # PCA.inverse_transform
        err = (Xs - rec) ** 2
        out[ok] = np.max(err / sens, axis=1)
    return out / cfg["recon_p99"]


def _z_vibracao(g: pd.DataFrame, estavel: pd.Series, tags: list[str],
                trips: list, excl_dias: float) -> np.ndarray:
    """Maior z robusto entre as sondas, contra referência rolante da própria sonda.

    Cada sonda é comparada consigo mesma no passado recente, não com as outras:
    as amplitudes entre pontos de medição diferem por uma ordem de grandeza, e
    uma referência comum faria a sonda mais agitada dominar sempre. A base
    termina GUARDA_H antes do ponto — colada nele, a própria degradação entraria
    na referência e se cancelaria."""
    q = estavel.to_numpy()
    hot = np.flatnonzero(q)
    out = np.full(len(g), np.nan)
    if len(hot) == 0:
        return out
    Xh = g[tags].to_numpy()[hot].astype("float64")

    # A referência apaga ±excl_dias em torno de cada trip JÁ OCORRIDO. Degradação
    # que se sabe ter terminado em falha não pode virar o "normal" contra o qual a
    # próxima é medida — entraria na mediana e se cancelaria. Em produção isto é
    # legítimo: o trip está no registro. A lista vem do bundle e precisa ser
    # atualizada a cada trip novo.
    Xref = Xh.copy()
    if trips:
        th = g.index[hot]
        for t in trips:
            f = pd.Timestamp(t)
            fora = (th >= f - pd.Timedelta(days=excl_dias)) & (th <= f + pd.Timedelta(days=2))
            Xref[np.asarray(fora)] = np.nan

    n_base = int(VB_HORAS_BASE * POR_HORA)
    guarda = int(VB_GUARDA_H * POR_HORA)
    n, k_sondas = Xh.shape
    MED = np.full((n, k_sondas), np.nan)
    S = np.full((n, k_sondas), np.nan)

    # A referência é reavaliada a cada VB_PASSO_H, e o bloco é ancorado no
    # CALENDÁRIO — não em "a cada N amostras a partir do início do array". A
    # contagem por amostra é estado implícito: numa janela o contador recomeça do
    # zero e os blocos caem em instantes diferentes dos do treino. Ancorado numa
    # época fixa, a mesma hora cai sempre no mesmo bloco, venha de que janela vier.
    th = g.index[hot]
    blocos = ((th - EPOCA) // pd.Timedelta(hours=VB_PASSO_H)).astype("int64").to_numpy()
    inicios = np.flatnonzero(np.concatenate(([True], blocos[1:] != blocos[:-1])))

    for ii, k in enumerate(inicios):
        j = int(inicios[ii + 1]) if ii + 1 < len(inicios) else n
        fim = k - guarda
        ini = max(0, fim - n_base)
        # Fração mínima da base para a sonda opinar. Medido de 0,25 a 1,00: o
        # alarme final é IDÊNTICO em todas (mesma banda, mesmo FP, mesmos 21
        # episódios). Fica 0,25 por ser o valor do treino. Não é alavanca —
        # registrado para que ninguém volte a mexer aqui esperando efeito.
        if fim - ini < int(n_base * VB_FRACAO_BASE_MIN):
            continue
        W = Xref[ini:fim]
        med = np.nanmedian(W, axis=0)
        s = np.nanmedian(np.abs(W - med), axis=0) * 1.4826
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        MED[k:j] = med
        S[k:j] = s
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((Xh - MED) / S)
    out[hot] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def preprocessar(modelos, df: pd.DataFrame, trips=None) -> pd.DataFrame:
    """Dado bruto → os quatro sinais suavizados, mais a máscara e os resets.

    `modelos` é a lista de bundles (de `carregar_modelos`). Cada mês da janela é
    pontuado pelo bundle que vigia nele — ver `carregar_modelos` para o motivo.
    Aceita um bundle único por compatibilidade, mas aí o aquecimento inteiro sai
    pontuado pelo PCA do mês corrente, que é o defeito que essa lista corrige."""
    if isinstance(modelos, dict):
        modelos = [modelos]
    g = _grade(df)          # a grade do treino -- ver `preparar_grade`
    mask, estavel, partida, op = _mascara(g)
    det = modelos[-1]["detector"]

    # ── t, p e sp: por trecho de vigência ──
    # O corte é mensal porque o bundle é mensal. Dentro de um trecho tudo é
    # vetorizado; são poucos trechos numa janela de 60 dias.
    t = np.full(len(g), np.nan)
    p = np.full(len(g), np.nan)
    sp = np.full(len(g), np.nan)
    meses = pd.date_range(g.index[0].normalize().replace(day=1),
                          g.index[-1], freq="MS", tz="UTC")
    for i_m, m0 in enumerate(meses):
        m1 = meses[i_m + 1] if i_m + 1 < len(meses) else g.index[-1] + pd.Timedelta(GRID)
        fatia = (g.index >= max(m0, g.index[0])) & (g.index < m1)
        if not fatia.any():
            continue
        mod = vigencia(modelos, max(m0, g.index[0]))
        w = g.loc[fatia]
        t[fatia] = _recon_pca(mod, "temperatura", w)
        p[fatia] = _recon_pca(mod, "pressao", w)
        sm = mod["spread_mancal"]
        # float64 pelo mesmo motivo do bundle: a aritmética do detector não deve
        # herdar a precisão com que o historiador exportou o CSV.
        spread = (w[sm["tag_alvo"]].astype("float64")
                  - w[sm["tags_irmaos"]].astype("float64").median(axis=1))
        sp[fatia] = np.abs((spread - sm["mediana"]) / sm["mad_robusto"]).to_numpy()

    cru = pd.DataFrame({
        "t": t, "p": p, "sp": sp,
        # ESTAVEL, não `mask`: a referência rolante conta amostras em regime, e
        # tirar dela as 6 h de blackout deslocaria a janela de 400 h para trás
        # sem motivo. O blackout filtra a DECISÃO, não a construção da referência.
        # O vb não é fatiado por mês: a referência dele é rolante, não do bundle.
        # Só os trips ATÉ o fim do dado. Um trip posterior é futuro: usá-lo para
        # limpar a referência é o vazamento clássico desta função — deixa a
        # referência artificialmente limpa justamente antes da falha e infla o z
        # durante ela. Em produção o futuro não existe; aqui ele também não.
        "vb": _z_vibracao(g, estavel, det["tags_vibracao"],
                          [t for t in (trips if trips is not None
                                       else det.get("trips_conhecidos", []))
                           if pd.Timestamp(t) <= g.index[-1]],
                          det.get("vb_exclusao_dias", 7.0)),
    }, index=g.index)

    out = pd.DataFrame(index=g.index)
    for c, hl in det["halflife"].items():
        # O EWMA do pandas CARREGA o último valor por cima de um buraco: com uma tag fora por
        # 24 h, o `t` ficava congelado no último valor e o canal aceso (e o CUSUM somando) sem
        # leitura nenhuma. Onde o sinal cru não existe, o EWMA também não. O estado interno do
        # filtro NÃO muda -- ele segue carregado, como na validação, que atravessa parada e
        # blackout -- só o VALOR exposto nos instantes sem medição. No histórico isso toca 0,1 h
        # vigiada: o alarme de 2024-11 a 2026-04 é idêntico (`paridade_pacote.py`).
        out[c] = (cru[c].ewm(halflife=pd.Timedelta(hl), times=g.index).mean()
                  .where(cru[c].notna()))
    out["mask"] = mask
    out["reset"] = (~mask) | partida
    out["operando"] = op
    return out


# ══════════════════════════════════════════════════ 4. prever
def _cusum(x: np.ndarray, reset: np.ndarray, carga: float) -> np.ndarray:
    """S_i = max(0, S_{i-1} + x_i), com S ← S·carga em cada reset.

    Vetorizado por trecho entre resets: dentro de um trecho tem forma fechada
    S_i = C_i + max(a0, -min(0, min_{k<=i} C_k)), com C o cumsum do trecho.
    O mínimo precisa incluir C_i (j <= i), não só até i-1."""
    S = np.empty(len(x))
    a = 0.0
    ini = 0
    for b in list(np.flatnonzero(reset)) + [len(x)]:
        if b > ini:
            C = np.cumsum(x[ini:b])
            prefmin = np.minimum(np.minimum.accumulate(C), 0.0)
            S[ini:b] = C + np.maximum(a, -prefmin)
            a = S[b - 1]
        if b < len(x):
            a = a * carga
            S[b] = a
            ini = b + 1
    return S


def _episodios(alerta: pd.Series, gap_h: float = GAP_EPISODIO_H) -> list[tuple]:
    a = alerta.fillna(False).to_numpy()
    idx = alerta.index
    if not a.any():
        return []
    corte = np.flatnonzero(a[1:] != a[:-1]) + 1
    ini = np.concatenate(([0], corte))
    fim = np.concatenate((corte, [len(a)]))
    br = [(idx[i], idx[j - 1]) for i, j in zip(ini, fim) if a[i]]
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h):
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _acende(E: pd.Series, thr: float, mask: pd.Series, reset: np.ndarray,
            kappa: float, h_cusum: float, carga: float) -> pd.Series:
    """Um canal acende por degrau sustentado OU por CUSUM.

    Os dois são necessários e pegam coisas diferentes: o degrau pega a subida
    franca; o CUSUM pega a deriva que fica logo abaixo do limiar por dias e
    nunca chega a fazer degrau."""
    Em = E.where(mask)
    deg = ((Em > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN)
           .sum() >= SUSTAIN)
    acum = ((Em / thr).clip(upper=20) - kappa).fillna(0.0).to_numpy()
    cu = pd.Series(_cusum(acum, reset, carga) > h_cusum, index=E.index)
    return (deg | cu) & mask


def prever(modelos, proc: pd.DataFrame) -> pd.DataFrame:
    """Sinais → alarme. Devolve um quadro por instante.

    O ponto de operação vem do bundle mais recente: ele é o mesmo em todos
    (só o baseline muda de mês para mês), e é o corrente que vale."""
    det = (modelos[-1] if isinstance(modelos, list) else modelos)["detector"]
    sinais = det["sinais"]
    idx = proc.index
    mask = proc["mask"].astype(bool)
    reset = proc["reset"].to_numpy()
    base = det["base"]
    k_lo, k_hi = det["k_nivel_a"], det["k_nivel_b"]
    kappa, h_cusum, carga = det["kappa"], det["h_cusum"], det["carga_reset"]

    A = {c: _acende(proc[c], base[c] * k_lo[c], mask, reset, kappa, h_cusum, carga)
         for c in sinais}
    B = {c: _acende(proc[c], base[c] * k_hi[c], mask, reset, kappa, h_cusum, carga)
         for c in sinais}

    # Nível A: SENSÍVEL — 3 dos 4 canais em limiar baixo. Pega a deriva cedo.
    vA = pd.Series(sum(A[c].astype(int) for c in sinais) >= det["voto_nivel_a"],
                   index=idx) & mask
    # Nível B: ESPECÍFICO — 2 dos 4 em limiar alto, e um deles tem de ser mecânico
    # (mancal ou vibração). O portão é redundante dado o nível A no histórico que
    # temos, mas sobre o nível B sozinho ele vale 0,603 → 0,689 FP/mês: fica.
    vB = (pd.Series(sum(B[c].astype(int) for c in sinais) >= det["voto_nivel_b"],
                    index=idx) & mask & (B["sp"] | B["vb"]))
    voto = vA | vB

    forca = pd.concat([proc[c].where(mask) / (base[c] * k_hi[c]) for c in sinais],
                      axis=1).max(axis=1)

    # ── Escalada por idade: reanúncio dentro de episódio permanente ──
    # Um alarme que já dura dias e de repente fica MUITO mais forte é notícia
    # nova, mas o episódio único o esconderia. Aqui ele é cortado em dois para
    # que o segundo nasça — e nascer dentro da janela é o que a operação lê.
    f = forca.fillna(0.0).to_numpy()
    v = voto.to_numpy().copy()
    n_idade = int(det["escalada_idade_h"] * POR_HORA)
    n_gap = int(GAP_EPISODIO_H * POR_HORA) + 1
    dentro, inicio, ja_forte = False, 0, False
    for i in range(len(v)):
        if not v[i]:
            dentro, ja_forte = False, False
            continue
        agora_forte = f[i] > det["escalada_abs"]
        if not dentro:
            dentro, inicio, ja_forte = True, i, agora_forte
            continue
        if agora_forte and not ja_forte and (i - inicio) >= n_idade:
            v[max(inicio + 1, i - n_gap):i] = False
            inicio = i
        ja_forte = agora_forte
    voto = pd.Series(v, index=idx)

    # ── Refratário, com furo para o episódio forte E velho ──
    # Depois de um alarme, 72 h em que episódio novo é descartado: repetir não é
    # notícia. Mas se o que vem é forte e o bloqueio já é antigo, é PIORAR — e
    # piorar é notícia. Sem esse furo a régua de início cai de 6/8 para 4/8.
    al = pd.Series(False, index=idx)
    bloq = ini_bloq = None
    fortes = []
    for a, b in _episodios(voto):
        forte = float(forca.loc[a:b].max()) > det["escalada_abs"]
        velho = (ini_bloq is not None
                 and (a - ini_bloq).total_seconds() / 3600 >= det["escalada_idade_h"])
        if bloq is not None and a <= bloq and not (forte and velho):
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=det["refratario_h"])
        ini_bloq = a
        if forte:
            fortes.append((a, b))

    # ── Duração mínima: 120 min, ou 60 min se o episódio for forte ──
    final = pd.Series(False, index=idx)
    for a, b in _episodios(al):
        dur = (b - a).total_seconds() / 60 + 2
        e_forte = any(x >= a and y <= b for x, y in fortes)
        if dur >= det["duracao_min"] or (e_forte and dur >= det["duracao_min_forte"]):
            final.loc[a:b] = True

    res = pd.DataFrame(index=idx)
    for c in sinais:
        res[c] = proc[c]
    res["forca"] = forca
    # Os flags POR CANAL, não só a contagem. Sem eles a tela mostra "3 canais"
    # e não diz quais — e quem está de plantão precisa saber se o que acendeu
    # foi vibração (mecânico, urgente) ou resíduo de pressão (pode ser processo).
    # Vêm de A[c]/B[c], que já respeitam a máscara: recalcular o limiar por fora
    # daria canal aceso com a máquina parada, que é contradição na mesma tela.
    for c in sinais:
        res[f"a_{c}"] = A[c].to_numpy()
        res[f"b_{c}"] = B[c].to_numpy()
    res["canais_nivel_a"] = sum(A[c].astype(int) for c in sinais)
    res["canais_nivel_b"] = sum(B[c].astype(int) for c in sinais)
    res["vigiado"] = mask
    res["is_anomaly"] = final
    res["severity"] = np.where(final, "alarme",
                               np.where(voto & mask, "atencao", "normal"))
    return res


def resumo_episodios(res: pd.DataFrame, desde: pd.Timestamp | None = None) -> pd.DataFrame:
    """Um alarme por linha — é assim que a operação lê, não ponto a ponto.

    `res` tem de ser a série INTEIRA. `desde` filtra quais episódios sair, sem
    mexer no início nem na duração deles.

    POR QUÊ ISSO IMPORTA. O detector não guarda estado: cada execução repontua a
    janela do zero. Se a janela for recortada ANTES de achar os episódios, todo
    episódio mais velho que ela ganha um início falso — a borda do recorte.
    Medido, num episódio real de 02/04 04:32 que durou 52,4 h, reportando as
    últimas 24 h de hora em hora:

        execução 03/04 00:00  ->  inicio 02/04 04:32   19,50 h   (certo)
        execução 03/04 08:00  ->  inicio 02/04 08:00   24,03 h   (falso)
        execução 03/04 16:00  ->  inicio 02/04 16:00   24,03 h   (falso)
        execução 04/04 00:00  ->  inicio 03/04 00:00   24,03 h   (falso)

    Quem identifica episódio pelo início vê um alarme NOVO a cada execução —
    cinquenta notificações para um evento — e a duração congela nas 24 h, então
    o operador nunca vê que já dura dois dias. É o erro clássico de integrar
    detector sem estado, e não aparece em teste: só em produção."""
    linhas = []
    for a, b in _episodios(res["is_anomaly"]):
        if desde is not None and b < desde:
            continue
        linhas.append(dict(inicio=a, fim=b,
                           horas=round((b - a).total_seconds() / 3600 + 2 / 60, 2),
                           forca_max=round(float(res["forca"].loc[a:b].max()), 2),
                           canais_max=int(res["canais_nivel_a"].loc[a:b].max())))
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════ 5b. diagnóstico da entrada
TAGS_MASCARA = ("RUNNING_A", "T5_AVG_A")


def tags_exigidas(modelos: list[dict]) -> dict[str, list[str]]:
    """Que colunas o bundle corrente precisa, por família."""
    m = modelos[-1]
    n, sp, det = m["normalizacao"], m["spread_mancal"], m["detector"]
    return {
        "temperatura": list(n["temperatura"]["cols"]),
        "pressao": list(n["pressao"]["cols"]),
        "mancal": [sp["tag_alvo"], *sp["tags_irmaos"]],
        "vibracao": list(det["tags_vibracao"]),
        "mascara": list(TAGS_MASCARA),
    }


def diagnostico_entrada(modelos: list[dict], df: pd.DataFrame) -> dict:
    """A entrada dá para confiar? Medido, porque perder tag não levanta erro.

    `_recon_pca` exige TODAS as colunas da família no instante (`notna().all`).
    Uma tag de catorze fora tira a família inteira do ar — e o detector segue
    alarmando pelos outros canais, com o resultado mudado e nenhuma mensagem.
    Medido nos 484 d de 2025-2026, apagando uma tag por vez:

        1 de 14 termorresistências  ->  canal t morto 100% do tempo, 20 -> 22 episódios
        1 de 12 tomadas de pressão  ->  canal p morto 100% do tempo, 20 -> 15 episódios
        1 de 10 sondas de vibração  ->  vb indisponível em 3,8%,     20 -> 20 episódios
        RUNNING_A                   ->  20 -> 0 episódios: CEGO, reportando "normal"

    A vibração aguenta porque o `vb` é o MÁXIMO entre as sondas; temperatura e
    pressão não aguentam porque são reconstrução conjunta. E a máscara é o pior
    caso: sem ela nada é vigiado e a tela fica verde para sempre.

    Veredito: "ok" | "degradado" | "cego". `cego` tem de parar a execução.

    Mede sobre a grade preparada (`preparar_grade`): uma tag que só traz texto do
    PI ("No Data") tem valor em toda linha do CSV e nenhuma medição."""
    df = preparar_grade(df)
    exig = tags_exigidas(modelos)
    familias, ausentes_todas = {}, []
    for fam, cols in exig.items():
        ausentes = [c for c in cols if c not in df.columns]
        vazias = [c for c in cols if c in df.columns and not df[c].notna().any()]
        mortas = sorted(set(ausentes) | set(vazias))
        ausentes_todas += mortas
        cobertura = {c: round(float(df[c].notna().mean()), 4)
                     for c in cols if c in df.columns}
        # a vibração tolera perda parcial (o vb é máximo entre sondas); as demais não
        tolerante = fam == "vibracao"
        familias[fam] = {
            "tags": len(cols),
            "tags_mortas": mortas,
            "tolerante_a_perda": tolerante,
            "cobertura_minima": round(min(cobertura.values()), 4) if cobertura else 0.0,
            "pior_tag": min(cobertura, key=cobertura.get) if cobertura else None,
            "operante": (len(mortas) < len(cols)) if tolerante else (not mortas),
        }

    cego = not familias["mascara"]["operante"] or all(
        not familias[f]["operante"] for f in ("temperatura", "pressao", "mancal", "vibracao"))
    degradado = any(not familias[f]["operante"] for f in familias)
    return {
        "veredito": "cego" if cego else ("degradado" if degradado else "ok"),
        "tags_mortas": sorted(set(ausentes_todas)),
        "familias": familias,
        "mensagem": (
            "ENTRADA CEGA: a máscara de operação ou todos os canais estão fora. "
            "O detector reportaria 'normal' para sempre — não é resultado, é ausência de medição."
            if cego else
            "Entrada degradada: há família sem todas as tags. O detector segue "
            "alarmando pelos canais restantes, com resultado diferente do validado."
            if degradado else "Entrada completa.")
    }


# ══════════════════════════════════════════════════ 5c. monitor de drift nos dados
DRIFT_LIMIAR_SIGMA = 10.0     # desvio de uma tag contra o baseline do bundle em vigor
DRIFT_JANELA_DIAS = 7
MANUTENCAO_MIN_H = 24.0       # HSX_6240001A ligado por pelo menos isso = manutenção
TRAVADO_FRACAO = 0.10         # IQR semanal abaixo desta fração do típico do PRÓPRIO sensor
TRAVADO_SEMANAS = 3           # ... por tantas semanas seguidas
TRAVADO_REF_MIN = 4           # semanas anteriores com dado para saber o "típico"
# Sensores cuja dispersão semanal tem mais de um modo POR NATUREZA: não são avaliados para
# "travado". O PI_0319 (gás do motor de partida) é pressurizado em ciclo de ~7 h em operação,
# e semanas com a linha despressurizada têm IQR ~0,006 contra ~0,3 a ~45 nas demais. A regra
# relativa o acusava em 8 das 10 semanas em que acionou no histórico (57 semanas, abr/2025 a
# abr/2026) -- todas falsas. O PDI_0301, parado em -0,401 desde nov/2025, é o único caso real e
# continua sendo acusado (12/23 e 12/30). Uma variante estatística (excluir quem já teve semanas
# baixas na referência) mata o falso e o verdadeiro juntos, e há um só caso real para calibrar:
# por isso a exceção é uma lista explícita, com o motivo físico, e não um parâmetro ajustado.
SENSORES_DE_ESTADO = ("954005_624_PI_0319",)


def _manutencoes(g: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Trechos com o modo de manutenção (HSX_6240001A) ligado >= 24 h; trechos a
    menos de 24 h um do outro são fundidos. Lista vazia se a tag não vier."""
    if "HSX_6240001A" not in g:
        return []
    h = (g["HSX_6240001A"].fillna(0) > 0.5).to_numpy()
    d = np.diff(np.concatenate(([0], h.astype(int), [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    tr = []
    for a, b in zip(g.index[ini], g.index[fim]):
        if tr and a - tr[-1][1] <= pd.Timedelta(hours=MANUTENCAO_MIN_H):
            tr[-1][1] = b
        else:
            tr.append([a, b])
    return [(a, b) for a, b in tr if b - a >= pd.Timedelta(hours=MANUTENCAO_MIN_H)]


def _sensores_travados(g: pd.DataFrame, mask: pd.Series, cols: list[str]) -> list[dict]:
    """Sensores cujo espalhamento caiu muito abaixo do PRÓPRIO normal e ficou lá.

    Compara o IQR de cada uma das últimas TRAVADO_SEMANAS semanas com a mediana do
    IQR semanal das semanas anteriores da janela de entrada. Um sensor que por
    natureza varia pouco não é punido por isso; só alerta quem cai a menos de 10%
    do próprio típico e fica lá.

    Por que não outras formas, testadas em `ks_drift.py`: a razão contra o IQR do
    baseline dispara em 15 de 62 semanas, puxada por sensores que variam pouco por
    natureza; e contar valores repetidos não funciona numa grade interpolada, onde
    até sensor travado muda um pouco a cada amostra. Esta regra dispara em 6 de 59
    semanas, só em dois sensores e só depois da manutenção de nov/2025: o
    diferencial do filtro de gás de selagem (PDI_0301) e o gás do motor de partida
    (PI_0319)."""
    fim = g.index[-1]
    iqr = []
    for k in range(13):
        a, b = fim - pd.Timedelta(days=7 * (k + 1)), fim - pd.Timedelta(days=7 * k)
        s = mask & (g.index > a) & (g.index <= b)
        if s.sum() < 2 * 720:
            iqr.append(None); continue
        X = g.loc[s, cols]
        iqr.append((X.quantile(0.75) - X.quantile(0.25)))
    rec = iqr[:TRAVADO_SEMANAS]
    ref = [x for x in iqr[TRAVADO_SEMANAS:] if x is not None]
    if any(x is None for x in rec) or len(ref) < TRAVADO_REF_MIN:
        return []
    tip = pd.concat(ref, axis=1).median(axis=1)
    out = []
    for c in cols:
        if c in SENSORES_DE_ESTADO or not np.isfinite(tip[c]) or tip[c] <= 0:
            continue
        r = [float(x[c] / tip[c]) for x in rec]
        if all(v < TRAVADO_FRACAO for v in r):
            out.append(dict(tag=c, razao_iqr=round(r[0], 3), semanas=TRAVADO_SEMANAS))
    return out


def monitor_drift(modelos: list[dict], df: pd.DataFrame) -> dict:
    """O baseline do bundle em vigor ainda representa os sensores? Não muda alarme.

    POR QUÊ (`scripts/pdm_fisico/drift_eventos.py` no repositório de pesquisa). A
    máquina volta de cada manutenção com um "normal" novo em muitos sensores de uma
    vez -- nas semanas com manutenção, 14 tags saltam > 3 sigma juntas (mediana),
    contra 4 nas demais. O retreino é mensal e leva até um mês para absorver. Em
    nov/2025 o diferencial do filtro de gás de selagem (PDI_0301) foi de +0,245
    para -0,400 kgf/cm² depois da manutenção de 05-15/11 -- instrumento rezerado
    ou trocado --, e o canal p ficou aceso 96% do mês.

    O QUE SE MEDE. Para cada tag de t e p, e para o spread do mancal: a mediana da
    última semana vigiada e da anterior, em sigmas do baseline do bundle em vigor
    (centro e IQR gravados no próprio bundle; sigma = IQR/1,349). A vibração fica de
    fora: o vb já tem referência móvel.

    O QUE ALERTA, calibrado em 62 semanas de 2025-2026:
      degrau_persistente  uma tag >= 10 sigma nas DUAS semanas, mesmo sinal: mudança
                          de nível -- instrumento, regime ou a própria máquina. A
                          persistência NÃO separa instrumento de máquina (a excursão do
                          mancal de out/2025 durou duas semanas). 10 sigma numa semana
                          só acontece em 11% das semanas.
      sensor_travado      o IQR das últimas 3 semanas abaixo de 10% do típico do
                          próprio sensor (`_sensores_travados`).
      atencao             >= 10 sigma só na última semana. Pode ser a própria
                          máquina (em 29/10/2025 o TI_0305 chegou a 37 sigma).

    O KS FOI TESTADO E FICOU DE FORA (`ks_drift.py`): o p-valor dá < 0,05 em 99,9%
    das comparações (autocorrelação), e o D satura -- o limiar calibrado cai em
    1,000, o máximo, então não sobra escala. Das duas semanas que só ele pegava, uma
    é saturação num sensor quantizado e a outra já tinha "atencao" pela mediana.
      ok                  nenhum dos dois.

    O QUE NÃO FAZ. Não recentra nem retreina sozinho: recentrar depois de manutenção
    foi testado e reprovado (`recentragem.py`) -- o bundle do mês seguinte, treinado
    com dado de antes e depois, já absorve o salto. O alerta serve para a equipe
    confirmar com a instrumentação e saber que o mês corrente está com baseline
    desatualizado nessas tags."""
    g = _grade(df)
    mask, *_ = _mascara(g)
    fim = g.index[-1]
    m = vigencia(modelos, fim)
    w1 = mask & (g.index > fim - pd.Timedelta(days=DRIFT_JANELA_DIAS))
    w0 = mask & (g.index <= fim - pd.Timedelta(days=DRIFT_JANELA_DIAS)) & \
        (g.index > fim - pd.Timedelta(days=2 * DRIFT_JANELA_DIAS))
    man = _manutencoes(g)
    ult = man[-1][1] if man else None
    manut = dict(ultima_manutencao_fim=(ult.strftime("%Y-%m-%dT%H:%M:%SZ") if ult is not None else None),
                 dias_desde_manutencao=(round((fim - ult).total_seconds() / 86400, 1)
                                        if ult is not None else None))
    if w1.sum() < 720:
        return dict(veredito="sem_dado", mensagem="menos de 1 dia vigiado na última semana",
                    bundle=m["dir"].name, limiar_sigma=DRIFT_LIMIAR_SIGMA,
                    janela_dias=DRIFT_JANELA_DIAS, tags_persistentes=[], tags_recentes=[],
                    sensores_travados=[], tags_acima_3sigma=0, maiores_desvios=[], **manut)
    desvio = {}
    for fam in ("temperatura", "pressao"):
        tr = m[f"{fam}_transformacao"]; cols = m["normalizacao"][fam]["cols"]
        sig = tr["scale"] / 1.349
        for j, c in enumerate(cols):
            if c not in g or not np.isfinite(sig[j]) or sig[j] <= 0:
                continue
            d1 = (g.loc[w1, c].median() - tr["center"][j]) / sig[j]
            d0 = ((g.loc[w0, c].median() - tr["center"][j]) / sig[j]) if w0.sum() >= 720 else np.nan
            desvio[c] = (float(d0), float(d1))
    sp = m["spread_mancal"]
    if all(c in g for c in [sp["tag_alvo"], *sp["tags_irmaos"]]):
        # MEDIANA dos irmãos, como no bundle (`spread_mancal`) e na inferência; com a média o
        # desvio saía viesado em ~0,6 sigma na mediana (a média dos irmãos fica ~0,8 °C acima)
        b = g[sp["tag_alvo"]] - g[sp["tags_irmaos"]].median(axis=1)
        d1 = (b[w1].median() - sp["mediana"]) / sp["mad_robusto"]
        d0 = ((b[w0].median() - sp["mediana"]) / sp["mad_robusto"]) if w0.sum() >= 720 else np.nan
        desvio["spread_mancal"] = (float(d0), float(d1))
    cols_trav = [c for fam in ("temperatura", "pressao")
                 for c in m["normalizacao"][fam]["cols"] if c in g]
    travados = _sensores_travados(g, mask, cols_trav)
    L = DRIFT_LIMIAR_SIGMA
    persist = [c for c, (a, b) in desvio.items()
               if np.isfinite(a) and abs(a) >= L and abs(b) >= L and np.sign(a) == np.sign(b)]
    recente = [c for c, (a, b) in desvio.items() if abs(b) >= L and c not in persist]
    if persist:
        ver = "degrau_persistente"
        msg = (f"nível novo em {', '.join(persist)} há >= 2 semanas contra o baseline do bundle "
               f"-- troca/rezero de instrumento, mudança de regime ou a própria máquina (em "
               f"out/2025 uma excursão do mancal durou duas semanas). Confirmar com a operação "
               f"e a instrumentação; o retreino do mês seguinte absorve o que for nível novo.")
    elif travados:
        ver = "sensor_travado"
        msg = (f"espalhamento abaixo de {TRAVADO_FRACAO:.0%} do próprio normal há "
               f"{TRAVADO_SEMANAS} semanas em {', '.join(x['tag'] for x in travados)} -- "
               f"instrumento possivelmente isolado, travado ou em falha. Confirmar com a "
               f"instrumentação: enquanto isso, o que ele mede não está sendo vigiado.")
    elif recente:
        ver = "atencao"
        msg = (f"desvio forte na última semana em {', '.join(recente)} -- pode ser a máquina "
               f"(é o que o detector vigia) ou instrumento. Se persistir, vira degrau.")
    else:
        ver, msg = "ok", "baseline do bundle representa os sensores"
    top = sorted(desvio.items(), key=lambda kv: -abs(kv[1][1]))[:5]
    return dict(veredito=ver, mensagem=msg, bundle=m["dir"].name,
                limiar_sigma=L, janela_dias=DRIFT_JANELA_DIAS,
                tags_persistentes=persist, tags_recentes=recente, sensores_travados=travados,
                tags_acima_3sigma=int(sum(abs(b) >= 3 for _, b in desvio.values())),
                maiores_desvios=[dict(tag=c, semana_anterior=round(a, 1) if np.isfinite(a) else None,
                                      ultima_semana=round(b, 1)) for c, (a, b) in top],
                **manut)


# ══════════════════════════════════════════════════ 6. contrato do dashboard
SINAIS_DESCRICAO = {
    "t": ("Resíduo PCA — temperatura",
          "Erro de reconstrução das 14 termorresistências contra o baseline do mês."),
    "p": ("Resíduo PCA — pressão",
          "Erro de reconstrução das 12 tomadas de pressão contra o baseline do mês."),
    "sp": ("Spread do mancal",
           "z robusto do TI_0305 (radial LNA) contra os três mancais irmãos."),
    "vb": ("Vibração",
           "Maior z robusto entre as 10 sondas TV_* contra referência rolante de 400 h."),
}


def contrato_dashboard(modelos: list[dict], res: pd.DataFrame,
                       desde: pd.Timestamp | None = None,
                       proc: pd.DataFrame | None = None,
                       series: bool = False,
                       diagnostico: dict | None = None,
                       drift: dict | None = None) -> dict:
    """O que o dashboard consome: um dicionário JSON-serializável.

    Não é o CSV com outro nome. O CSV é a série; isto é o que a tela precisa
    para ser honesta sem reimplementar o detector:

      · os LIMIARES, para desenhar a linha junto da curva. Sem eles o dashboard
        chuta uma escala e a curva deixa de significar algo;
      · a OBSERVABILIDADE. O detector só enxerga a máquina de pé e em regime —
        aqui ela ficou 58% do tempo. Uma tela que mostra "normal" quando o
        detector está cego mente por omissão, e esse é o erro mais caro que um
        painel de preditiva comete;
      · a VALIDADE do bundle. Vencido, o número na tela não vale;
      · os EPISÓDIOS, que é como a operação lê — um alarme por linha, não ponto
        a ponto.

    `series=True` inclui a série inteira; para janelas longas prefira o CSV."""
    m = modelos[-1]
    det, mod = m["detector"], m["modelo"]
    sinais = det["sinais"]
    fim_dado = res.index[-1]

    base_fim = pd.Timestamp(mod["baseline_fim"])
    validade = int(mod.get("validade_dias", 62))
    vence_em = base_fim + pd.Timedelta(days=validade)

    # `res` é a série inteira; `desde` é onde começa o que se reporta. Os
    # episódios saem da série inteira (ver `resumo_episodios`); as contagens,
    # da fatia.
    if desde is None:
        desde = res.index[0]
    fatia = res.loc[res.index >= desde]
    sev = fatia["severity"]
    atual = str(sev.iloc[-1])
    # desde quando o estado atual não muda
    sev_td = res["severity"]
    troca = (sev_td != sev_td.shift()).to_numpy().nonzero()[0]
    desde_estado = res.index[troca[-1]] if len(troca) else res.index[0]

    eps = resumo_episodios(res, desde=desde)
    horas_alarme = float(eps["horas"].sum()) if len(eps) else 0.0
    horas_janela = (fatia.index[-1] - fatia.index[0]).total_seconds() / 3600 + 2 / 60

    fora = {}
    if proc is not None:
        for c in sinais:
            if c in proc:
                fora[c] = int(pd.isna(proc[c]).sum())

    out = {
        "schema": "simpred.cabiunas.tc33003a/1",
        "equipamento": mod["equipamento"],
        "gerado_em": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),

        "detector": {
            "versao": mod["versao_detector"],
            "arquitetura": mod["arquitetura"],
            "regra": (f"Nível A: >= {det['voto_nivel_a']} dos {len(sinais)} canais "
                      f"acima do limiar sensível. Nível B: >= {det['voto_nivel_b']} "
                      f"acima do limiar específico, com portão sp|vb. Alarme = "
                      f"A ou B, sustentado {SUSTAIN * 2} min, duração mínima "
                      f"{det['duracao_min']} min ({det['duracao_min_forte']} se forte), "
                      f"refratário {det['refratario_h']} h."),
            "sinais": [{
                "id": c,
                "nome": SINAIS_DESCRICAO[c][0],
                "descricao": SINAIS_DESCRICAO[c][1],
                "unidade": "adimensional (z robusto acumulado por CUSUM)",
                "limiar_nivel_a": det["k_nivel_a"][c],
                "limiar_nivel_b": det["k_nivel_b"][c],
            } for c in sinais],
            "voto_nivel_a": det["voto_nivel_a"],
            "voto_nivel_b": det["voto_nivel_b"],
        },

        "bundle": {
            "nome": m["dir"].name,
            "baseline_inicio": mod["baseline_inicio"],
            "baseline_fim": mod["baseline_fim"],
            "baseline_pontos": mod["baseline_pontos"],
            "validade_dias": validade,
            "vence_em": vence_em.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dias_restantes": int((vence_em - fim_dado).days),
            "vencido": bool(checa_validade(m, fim_dado) is not None),
            "bundles_carregados": len(modelos),
            "cadencia_retreino": mod["cadencia_retreino"],
        },

        "janela": {
            "inicio": fatia.index[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fim": fim_dado.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "instantes": int(len(fatia)),
            "entrada_desde": res.index[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "passo_segundos": 120,
            "horas": round(horas_janela, 1),
        },

        # A tela TEM de mostrar isto. "normal" com o detector cego é mentira.
        "observabilidade": {
            "instantes_vigiados": int(fatia["vigiado"].sum()),
            "fracao_vigiada": round(float(fatia["vigiado"].mean()), 4),
            "criterio": "RUNNING_A > 0.5 e T5_AVG_A > 300 °C (máquina de pé e em regime)",
            "aquecimento_dias": AQUECIMENTO_DIAS,
            "instantes_sem_sinal": fora,
            # Perder uma tag não levanta erro; ver `diagnostico_entrada`.
            "entrada": diagnostico or {"veredito": "nao_avaliado"},
            # O baseline ainda representa os sensores? Não muda alarme; ver `monitor_drift`.
            "drift_dados": drift or {"veredito": "nao_avaliado"},
        },

        "estado_atual": {
            "severity": atual,
            "desde": desde_estado.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vigiado": bool(res["vigiado"].iloc[-1]),
            "forca": (None if pd.isna(res["forca"].iloc[-1])
                      else round(float(res["forca"].iloc[-1]), 2)),
            "canais_nivel_a": int(res["canais_nivel_a"].iloc[-1]),
            "canais_nivel_b": int(res["canais_nivel_b"].iloc[-1]),
            # De a_<c>, que o detector calculou com a máscara aplicada. Comparar
            # o sinal cru ao limiar aqui daria canal aceso com a máquina parada.
            "canais_nivel_a_acesos": [c for c in sinais if bool(res[f"a_{c}"].iloc[-1])],
            "canais_nivel_b_acesos": [c for c in sinais if bool(res[f"b_{c}"].iloc[-1])],
        },

        "resumo": {
            "episodios": int(len(eps)),
            "instantes_alarme": int((sev == "alarme").sum()),
            "instantes_atencao": int((sev == "atencao").sum()),
            "horas_alarme": round(horas_alarme, 2),
            "horas_alarme_por_mes": round(horas_alarme / max(horas_janela / 730.0, 1e-9), 2),
        },

        "episodios": [{
            # CHAVE ESTÁVEL. Duas execuções que veem o mesmo episódio devolvem
            # o mesmo `id` — é por ele que a integração deduplica, nunca por
            # posição na lista nem por hora de geração.
            "id": f"{mod['equipamento']}:{r.inicio:%Y%m%dT%H%M%SZ}",
            "em_curso": bool(r.fim >= res.index[-1] - pd.Timedelta(minutes=2)),
            "inicio": r.inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fim": r.fim.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "horas": float(r.horas),
            "forca_max": float(r.forca_max),
            "canais_max": int(r.canais_max),
            # quais canais participaram em algum momento do episódio
            "canais": [c for c in sinais
                       if bool(res[f"a_{c}"].loc[r.inicio:r.fim].any())],
        } for r in eps.itertuples()],
    }

    if series:
        out["series"] = {
            "ts": [i.strftime("%Y-%m-%dT%H:%M:%SZ") for i in res.index],
            **{c: [None if pd.isna(v) else round(float(v), 4)
                   for v in res[c]] for c in sinais},
            "severity": [str(v) for v in sev],
            "vigiado": [bool(v) for v in res["vigiado"]],
            **{f"a_{c}": [bool(v) for v in res[f"a_{c}"]] for c in sinais},
        }
    return out
