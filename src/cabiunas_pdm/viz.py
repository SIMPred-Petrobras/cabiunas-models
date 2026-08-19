"""Figuras da decisão do detector, no padrão dos notebooks do Transpetro.

Referência: ``Transpetro-modelos/notebooks/*/plot_anomalias.ipynb``. A gramática
visual é copiada de lá, elemento por elemento:

* série do sensor em **steelblue**, rotulada ``Série temporal: <tag>``;
* instantes de anomalia em **vermelho**, plotados *sobre a própria leitura*;
* falha real como **linha vertical laranja tracejada**;
* painel de score em **roxo**, com o limite em **vermelho tracejado** e a área
  excedente preenchida de vermelho translúcido;
* cruzamento bruto do limite em **laranja pequeno** e alerta já com persistência
  em **vermelho** — a mesma distinção que o Transpetro faz entre
  ``threshold_flags`` e ``anomaly_flags``;
* ``figsize`` 12 de largura, 2,5 por painel, estilo ``seaborn-v0_8-whitegrid``,
  legenda em ``upper left`` com ``ncol=3``, ``suptitle`` em negrito.

Três coisas do projeto Cabiúnas não existem no Transpetro e aparecem como
acréscimos, sem mudar a gramática:

1. o limite é **percentil do baseline do mês** (re-baseline mensal), então é uma
   linha em escada em vez de uma reta;
2. há **três sinais** (temperatura, pressão/óleo, spread do mancal), cada um com
   seu painel de score, e um painel final com quantos estão sustentados — é essa
   contagem que decide o alerta;
3. a máquina passa boa parte do tempo **parada**, e sem operação não há score: as
   paradas ficam sombreadas em cinza para o buraco no gráfico não ser lido como
   falta de dado.
"""
from __future__ import annotations


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


# ------------------------------------------------ paleta e medidas do Transpetro
COR_SERIE = "steelblue"
COR_ANOMALIA = "red"
COR_SCORE = "purple"
COR_EVENTO = "orange"
COR_LIMITE = "red"
COR_CRUZAMENTO = "orange"
# parada bem clara de propósito: é fundo. As faixas de episódio (falso
# positivo em grafite) precisam se destacar dela.
COR_PARADA = "0.91"
COR_CONTAGEM = "purple"
# O vermelho passa a ser EXCLUSIVO do alerta confirmado — a decisão do sistema.
# Antes ele marcava também "este sinal isolado está acima do limite", que acontece
# 47% do tempo na temperatura: o painel de score ficava vermelho e o do sensor
# limpo, com a mesma cor significando coisas diferentes.
COR_SINAL_QUENTE = "#e08214"

LARGURA = 12.0          # FIGSIZE_WIDTH do Transpetro
ALTURA_PAINEL = 2.5     # PANEL_HEIGHT do Transpetro
RECT_TITULO = [0, 0, 1, 0.96]


def usar_estilo() -> None:
    """Estilo dos notebooks do Transpetro (chamar uma vez por notebook)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 110,
        # o Transpetro salva a 180; aqui 150 mantém os notebooks navegáveis, já
        # que cada figura tem vários painéis e 16 meses de série
        "savefig.dpi": 150,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


# --------------------------------------------------------------- utilidades
MESES = ["jan", "fev", "mar", "abr", "mai", "jun",
         "jul", "ago", "set", "out", "nov", "dez"]


def num(valor: float, casas: int = 2) -> str:
    """Número no padrão brasileiro (vírgula decimal)."""
    return f"{valor:,.{casas}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def _pct(valor: float) -> str:
    """Percentil sem zeros à direita: 99.99 -> '99,99'."""
    return f"{valor:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def auto_freq(index: pd.DatetimeIndex, alvo: int = 3000) -> str | None:
    """Passo de reamostragem que deixa a figura leve sem esconder picos.

    Necessário aqui e não no Transpetro: a grade é de 30 s e o período tem
    16 meses (1,4 milhão de pontos por sinal).
    """
    if len(index) <= alvo:
        return None
    span = index[-1] - index[0]
    minutos = max(int(span.total_seconds() / 60 / alvo), 1)
    for candidato in (1, 2, 5, 10, 15, 30, 60, 120, 240):
        if minutos <= candidato:
            return f"{candidato}min"
    return "1D"


def decimar(objeto, freq: str | None, how: str = "median"):
    """Reamostra série/frame para desenho. ``max`` preserva picos de score/flag."""
    if freq is None:
        return objeto
    return getattr(objeto.resample(freq), how)()


def sombrear_paradas(ax, operabilidade: pd.DataFrame, freq: str | None = None,
                     rotulo: bool = False) -> None:
    """Cinza onde a máquina não estava operando (ali não existe score)."""
    operando = operabilidade["in_operation"]
    if freq:
        operando = operando.resample(freq).max().fillna(False)
    parada = ~operando.astype(bool)
    if not parada.any():
        return
    bloco = (parada != parada.shift(fill_value=False)).cumsum()
    primeiro = True
    for _, trecho in parada[parada].groupby(bloco[parada]):
        ax.axvspan(trecho.index[0], trecho.index[-1], color=COR_PARADA, zorder=0,
                   label="Máquina parada" if (rotulo and primeiro) else None)
        primeiro = False


def marcar_eventos(ax, eventos: list[dict], rotulo: str | None = "Falha real (trip)") -> None:
    """Linha vertical laranja tracejada em cada falha real (padrão Transpetro)."""
    for i, evento in enumerate(eventos):
        ax.axvline(evento["inicio"], color=COR_EVENTO, linestyle="--", linewidth=1.4,
                   zorder=5, label=rotulo if i == 0 else None)


def marcar_alerta(ax, alerta: pd.Series, rotulo: str | None = "Alerta confirmado",
                  freq: str | None = None) -> None:
    """Faixa vermelha onde o **sistema alarmou** — desenhada em todos os painéis.

    Um elemento visual, um significado: alinhado entre sensores e scores, deixa
    claro que o alarme é raro mesmo quando um sinal isolado passa do limite muito.
    """
    ativo = alerta.fillna(False).astype(bool)
    if freq:
        ativo = ativo.resample(freq).max().fillna(False).astype(bool)
    if not ativo.any():
        return
    bloco = (ativo != ativo.shift(fill_value=False)).cumsum()
    primeiro = True
    for _, trecho in ativo[ativo].groupby(bloco[ativo]):
        ax.axvspan(trecho.index[0], trecho.index[-1], color=COR_ANOMALIA, alpha=0.16,
                   zorder=1, label=rotulo if primeiro else None)
        primeiro = False


CORES_EPISODIO = {"detecção": COR_ANOMALIA,
                  "falso positivo": "#1f2933",      # grafite, para não virar "parada"
                  "antes de parada (não contado)": COR_EVENTO}


def pontos_por_tipo(resultado, freq: str | None = None) -> dict[str, pd.DatetimeIndex]:
    """Instantes de alerta agrupados pelo **tipo do episódio** a que pertencem.

    Devolve os instantes já na grade de desenho (``freq``), para que sejam
    plotados como pontos sobre a leitura do sensor — que é a forma que se lê
    melhor num eixo de 16 meses. A cor passa a carregar a informação que antes
    dependia de faixas: o que antecipou falha, o que foi falso positivo e o que
    veio antes de uma parada não catalogada.
    """
    alerta = resultado.alert.fillna(False).astype(bool)
    if freq:
        alerta = alerta.resample(freq).max().fillna(False).astype(bool)
    instantes = alerta[alerta].index
    spans = dict(resultado.episode_spans)
    saida: dict[str, list] = {}
    for _, linha in resultado.episodes_table().iterrows():
        inicio = linha["inicio"]
        fim = spans.get(inicio, inicio)
        # na grade diária o instante desenhado pode cair antes do início real do
        # episódio; a tolerância de um passo evita perder o ponto
        dentro = instantes[(instantes >= inicio.normalize()) & (instantes <= fim)]
        saida.setdefault(linha["tipo"], []).extend(dentro)
    return {tipo: pd.DatetimeIndex(sorted(set(v))) for tipo, v in saida.items()}


def marcar_episodios_por_tipo(ax, resultado, alpha: float = 0.28,
                              largura_minima: str = "8h", rotular: bool = True) -> None:
    """Uma faixa por episódio de alerta, colorida pelo que ele **foi**.

    Numa visão de 16 meses, pintar todo alerta da mesma cor não informa: o leitor
    não distingue o que antecipou falha do que foi falso alarme. Aqui a cor conta:
    vermelho = antecipou falha catalogada, cinza = falso positivo, laranja = veio
    antes de uma parada não catalogada (não contado como erro).

    As faixas usam o **início e fim reais** de cada episódio, não a série decimada
    — desenhar a partir do dado reamostrado por dia infla a extensão em ~1,8×.
    ``largura_minima`` só garante que um episódio de 2 minutos ainda seja visível
    em um eixo de 16 meses.
    """
    minima = pd.Timedelta(largura_minima)
    spans = dict(resultado.episode_spans)
    vistos: set[str] = set()
    # marcador no topo além da faixa: num eixo de 16 meses, 8 h de largura são
    # 0,07% da figura — menos de um pixel. A faixa mostra a DURAÇÃO real; o
    # marcador garante que um episódio de 30 min não desapareça.
    for _, linha in resultado.episodes_table().iterrows():
        inicio = linha["inicio"]
        ax.scatter([inicio], [0.965], marker="v", s=26, clip_on=False, zorder=7,
                   color=CORES_EPISODIO.get(linha["tipo"], COR_ANOMALIA),
                   transform=ax.get_xaxis_transform())
    for _, linha in resultado.episodes_table().iterrows():
        inicio = linha["inicio"]
        fim = max(spans.get(inicio, inicio), inicio + minima)
        tipo = linha["tipo"]
        rotulo = None
        if rotular and tipo not in vistos:
            rotulo = tipo
            vistos.add(tipo)
        ax.axvspan(inicio, fim, color=CORES_EPISODIO.get(tipo, COR_ANOMALIA),
                   alpha=alpha, zorder=1, label=rotulo)


def rug_episodios(ax, episodios, altura: float = 0.96, cor: str = COR_ANOMALIA,
                  rotulo: str | None = None, marcador: str = "v",
                  tamanho: float = 22) -> None:
    """Marcadores no topo do painel — legível mesmo com dezenas de episódios."""
    if not len(episodios):
        return
    ax.scatter(list(episodios), [altura] * len(episodios), marker=marcador, s=tamanho,
               color=cor, zorder=6, clip_on=False, label=rotulo,
               transform=ax.get_xaxis_transform())


def _formatar_eixo_x(ax, index: pd.DatetimeIndex) -> None:
    span = index[-1] - index[0]
    if span <= pd.Timedelta("3d"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    elif span <= pd.Timedelta("60d"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:   # mês em português (o locale do sistema não é garantido)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{MESES[mdates.num2date(x).month - 1]}/"
                         f"{mdates.num2date(x):%y}"))


def _ylim_robusto(ax, serie: pd.Series, incluir: pd.Series | None = None,
                 folga: float = 0.08, ate=None) -> None:
    """Eixo y nos percentis 0,2–99,8, garantindo que o limite caiba, e avisa se cortar.

    Desvio deliberado do Transpetro: aqui vários sensores saltam para o fim de
    escala no instante do trip (termopar de mancal indo a ~850 °C) e o score pode
    subir três ordens de grandeza. Sem recorte, o salto domina o eixo e a subida
    que **antecede** a falha — o objeto da análise — desaparece.

    Com ``ate``, os limites são calculados **só com o que vem antes** daquele
    instante (a falha): é o trecho que interessa ler. O que vem depois do trip
    aparece cortado, com o pico anotado.
    """
    limpo = serie.dropna()
    if len(limpo) < 10:
        return
    referencia = limpo.loc[:ate] if ate is not None else limpo
    if len(referencia) < 10:
        referencia = limpo
    baixo, alto = np.nanpercentile(referencia.to_numpy(), [0.2, 99.8])
    if incluir is not None and len(incluir.dropna()):
        baixo = min(baixo, float(incluir.min()))
        alto = max(alto, float(incluir.max()) * 1.05)
    if not np.isfinite(baixo) or not np.isfinite(alto) or alto <= baixo:
        return
    margem = (alto - baixo) * folga
    ax.set_ylim(baixo - margem, alto + margem)
    pico = float(limpo.max())
    if pico > alto + margem:
        # casas decimais conforme a magnitude: score de 0,04 com 1 casa virava "0,0"
        casas = 3 if abs(pico) < 1 else (2 if abs(pico) < 10 else 1)
        ax.annotate(f"pico {num(pico, casas)} fora de escala", xy=(0.995, 0.92),
                    xycoords="axes fraction", ha="right", fontsize=7, color="0.35")


def _ylim_log(ax, serie: pd.Series, baixo: float = 2.0, alto: float = 99.98) -> None:
    """Eixo log recortado em décadas em torno de 1,0 (o limite normalizado)."""
    ax.set_yscale("log")
    valores = serie.to_numpy(dtype=float)
    valores = valores[np.isfinite(valores) & (valores > 0)]
    if valores.size < 10:
        return
    piso, teto = np.percentile(valores, [baixo, alto])
    ax.set_ylim(min(10.0 ** np.floor(np.log10(piso)), 0.1),
                max(10.0 ** np.ceil(np.log10(teto)), 10.0))


# ------------------------------------------------------- painéis reutilizáveis
def painel_sensor(ax, serie: pd.Series, anomalias: pd.DatetimeIndex,
                  eventos: list[dict], operabilidade: pd.DataFrame | None = None,
                  freq: str | None = None, primeiro: bool = True, ate=None) -> None:
    """Painel de sensor no padrão Transpetro: azul + anomalias em vermelho."""
    nome = serie.name.replace("954005_624_", "") if serie.name else "sensor"
    if operabilidade is not None:
        sombrear_paradas(ax, operabilidade, freq, rotulo=primeiro)
    ax.plot(serie.index, serie.to_numpy(), color=COR_SERIE, linewidth=1.0, alpha=0.85,
            label=f"Série temporal: {nome}")
    valores = serie.reindex(anomalias).dropna()
    if not valores.empty:
        ax.scatter(valores.index, valores.to_numpy(), s=28, color=COR_ANOMALIA,
                   edgecolor="none", zorder=4, label="Anomalia")
    marcar_eventos(ax, eventos, rotulo="Falha real (trip)" if primeiro else None)
    _ylim_robusto(ax, serie, ate=ate)
    ax.set_ylabel(nome, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=True)


NOME_METRICA = {"pca": "Erro de reconstrução (MSE)", "ae": "Erro de reconstrução (MSE)",
                "iforest": "Score de anomalia"}


def painel_score(ax, score: pd.Series, limite: pd.Series, cruzou: pd.Series,
                 sustentado: pd.Series, nome: str, eventos: list[dict],
                 operabilidade: pd.DataFrame | None = None, freq: str | None = None,
                 percentil: float | None = None, normalizar: bool = False,
                 ate=None, metrica: str = "Score de anomalia") -> None:
    """Painel de score no padrão Transpetro, com o limite do mês em escada.

    ``normalizar=True`` divide o score pelo limite daquele mês e usa eixo log: o
    limite passa a ser a reta 1,0. É o que torna a visão de 16 meses legível — em
    unidade crua, um pico de 63 mil achata o resto da série, e o limite (que muda
    a cada re-baseline) sobe e desce de forma que atrapalha mais do que informa.
    """
    if operabilidade is not None:
        sombrear_paradas(ax, operabilidade, freq)
    if normalizar:
        score = score / limite
        limite = pd.Series(1.0, index=score.index)
    rotulo_limite = ("Limite do mês"
                     + (f" (percentil {_pct(percentil)} do baseline)"
                        if percentil is not None else ""))
    ax.plot(limite.index, limite.to_numpy(), color=COR_LIMITE, linestyle="--",
            linewidth=1.3, drawstyle="steps-post", label=rotulo_limite)
    acima = (score > limite).fillna(False).to_numpy()
    ax.fill_between(score.index, limite.to_numpy(), score.to_numpy(), where=acima,
                    color=COR_LIMITE, alpha=0.18)
    ax.plot(score.index, score.to_numpy(), color=COR_SCORE, linewidth=1.0, zorder=4,
            label=f"{metrica}: {nome}")
    # trecho em que ESTE sinal está acima do limite e já sustentou o tempo exigido:
    # sobreposto em âmbar sobre a própria curva, em vez de pontos vermelhos
    quente = score.where(sustentado.reindex(score.index).fillna(False).astype(bool))
    if quente.notna().any():
        ax.plot(quente.index, quente.to_numpy(), color=COR_SINAL_QUENTE, linewidth=2.6,
                alpha=0.95, zorder=5, solid_capstyle="round",
                label="Este sinal conta para a confirmação")
    marcar_eventos(ax, eventos, rotulo=None)
    if normalizar:
        _ylim_log(ax, score)
    else:
        _ylim_robusto(ax, score, incluir=limite, ate=ate)
    ax.set_ylabel(f"{metrica.split(' (')[0]} / limite" if normalizar else metrica,
                  fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ordem = sorted(range(len(labels)),
                   key=lambda i: (not labels[i].startswith(("Score", "Erro")),
                                  not labels[i].startswith("Limite")))
    ax.legend([handles[i] for i in ordem], [labels[i] for i in ordem],
              loc="upper left", fontsize=7, ncol=3, frameon=True)
    ax._refazer_legenda = True


def painel_confirmacao(ax, ativos: pd.Series, confirm: int, eventos: list[dict],
                       operabilidade: pd.DataFrame | None = None,
                       freq: str | None = None) -> None:
    """Quantos sinais estão sustentados ao mesmo tempo — é o que dispara o alerta."""
    if operabilidade is not None:
        sombrear_paradas(ax, operabilidade, freq)
    ax.fill_between(ativos.index, 0, ativos.to_numpy(), step="mid",
                    color=COR_CONTAGEM, alpha=0.55, label="Sinais sustentados")
    ax.axhline(confirm, color=COR_LIMITE, linestyle="--", linewidth=1.3,
               label=f"Confirmação exigida ({confirm} sinais)")
    marcar_eventos(ax, eventos, rotulo=None)
    teto = max(3, int(ativos.max() or 3))
    ax.set_ylim(0, teto + 0.9)
    ax.set_yticks(range(0, teto + 1))
    ax.set_ylabel("Nº de sinais", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)


# ------------------------------------------------------------ zoom por evento
def plot_evento(resultado, evento: dict, sensores: list[str] | None = None,
                antes: str = "5d", depois: str = "12h", freq: str | None = None,
                modo: str = "consolidado"):
    """Painéis por sensor + um painel de score por sinal, em volta de uma falha.

    ``modo="consolidado"`` (padrão) devolve uma figura só, como o modo
    ``consolidated`` do Transpetro. ``modo="separado"`` devolve uma lista de
    figuras, uma por painel — equivalente ao modo ``separated``.
    """
    from .replay import SENSORS_BY_MECHANISM

    inicio = evento["inicio"] - pd.Timedelta(antes)
    fim = evento["fim"] + pd.Timedelta(depois)
    sensores = sensores or SENSORS_BY_MECHANISM.get(evento["mecanismo"], [])
    sensores = [s for s in sensores if s in resultado.sensors.columns]

    ratio_index = resultado.scores.loc[inicio:fim].index
    if len(ratio_index) == 0:
        raise ValueError(f"sem score na janela {inicio} .. {fim}")
    freq = freq or auto_freq(ratio_index, alvo=2500)

    janela = decimar(resultado.sensors.loc[inicio:fim, sensores], freq, "median")
    scores = decimar(resultado.scores.loc[inicio:fim], freq, "max")
    limites = decimar(resultado.limits.loc[inicio:fim], freq, "max")
    cruz = decimar(resultado.crossings.loc[inicio:fim].astype(float), freq, "max") > 0
    flags = decimar(resultado.flags.loc[inicio:fim].astype(float), freq, "max") > 0
    alerta = decimar(resultado.alert.loc[inicio:fim].astype(float), freq, "max")
    ativos = decimar(resultado.n_active.loc[inicio:fim], freq, "max")
    oper = resultado.operability.loc[inicio:fim]
    anomalias = alerta[alerta > 0].index

    veredito = (f"detectada {num(evento['lead_h'], 1)} h antes" if evento["detectada"]
                else "NÃO detectada nas 48 h anteriores")
    titulo = (f"Análise de anomalias — TC-330.03A | {evento['inicio']:%d/%m/%Y %H:%M} — "
              f"falha de {evento['mecanismo']} ({veredito})")
    pct = resultado.trial["threshold"]

    # os limites do eixo y vêm do trecho ANTERIOR à falha: é o que se quer ler.
    # Depois do trip os sensores vão ao fim de escala e o score sobe ordens de
    # grandeza — sem isso, o pós-falha domina o eixo e o precursor desaparece.
    corte = evento["inicio"]
    alerta_bool = alerta > 0

    def desenhar_sensor(ax, coluna, primeiro):
        painel_sensor(ax, janela[coluna], anomalias, [evento], oper, freq, primeiro,
                      ate=corte)
        marcar_alerta(ax, alerta_bool,
                      rotulo="Alerta confirmado" if primeiro else None)

    metrica = NOME_METRICA.get(resultado.trial["model"], "Score de anomalia")

    def desenhar_score(ax, sinal):
        painel_score(ax, scores[sinal], limites[sinal], cruz[sinal], flags[sinal],
                     sinal, [evento], oper, freq, percentil=pct, ate=corte,
                     metrica=metrica)
        marcar_alerta(ax, alerta_bool, rotulo="Alerta confirmado")
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc="upper left", fontsize=7, ncol=3, frameon=True)

    if modo == "separado":
        figuras = []
        for i, coluna in enumerate(janela.columns):
            fig, ax = plt.subplots(figsize=(LARGURA, ALTURA_PAINEL * 1.4))
            desenhar_sensor(ax, coluna, i == 0)
            ax.set_xlabel("Tempo (data/hora)", fontsize=10)
            _formatar_eixo_x(ax, janela.index)
            fig.suptitle(f"{titulo} | {coluna}", fontsize=14, fontweight="bold", y=0.995)
            fig.tight_layout(rect=RECT_TITULO)
            figuras.append(fig)
        for sinal in scores.columns:
            fig, ax = plt.subplots(figsize=(LARGURA, ALTURA_PAINEL * 1.6))
            desenhar_score(ax, sinal)
            ax.set_xlabel("Tempo (data/hora)", fontsize=10)
            _formatar_eixo_x(ax, scores.index)
            fig.suptitle(f"{titulo} | score {sinal}", fontsize=14, fontweight="bold",
                         y=0.995)
            fig.tight_layout(rect=RECT_TITULO)
            figuras.append(fig)
        return figuras

    n_sensores, n_sinais = len(janela.columns), len(scores.columns)
    alturas = [1.0] * n_sensores + [1.0] * n_sinais + [0.55]
    fig, axes = plt.subplots(len(alturas), 1, sharex=True,
                             figsize=(LARGURA, ALTURA_PAINEL * (sum(alturas) * 0.92)),
                             gridspec_kw={"height_ratios": alturas})
    axes = np.atleast_1d(axes)
    for i, coluna in enumerate(janela.columns):
        desenhar_sensor(axes[i], coluna, i == 0)
    for j, sinal in enumerate(scores.columns):
        desenhar_score(axes[n_sensores + j], sinal)
    painel_confirmacao(axes[-1], ativos, resultado.confirm, [evento], oper, freq)
    marcar_alerta(axes[-1], alerta_bool, rotulo=None)
    axes[-1].set_xlabel("Tempo (data/hora)", fontsize=10)
    _formatar_eixo_x(axes[-1], scores.index)
    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------------------- visão do período todo
def plot_periodo(resultado, titulo: str | None = None, freq: str | None = None,
                 normalizar: bool = True):
    """Um painel de score por sinal no período inteiro + painel de confirmação.

    Por padrão o score vem dividido pelo limite do mês (``normalizar=True``): em
    16 meses com re-baseline mensal, é a única forma de a figura permanecer
    legível. Nos zooms por evento (uma janela, um limite) o score aparece na
    unidade crua, como no Transpetro.
    """
    freq = freq or auto_freq(resultado.scores.index)
    scores = decimar(resultado.scores, freq, "max")
    limites = decimar(resultado.limits, freq, "max")
    cruz = decimar(resultado.crossings.astype(float), freq, "max") > 0
    flags = decimar(resultado.flags.astype(float), freq, "max") > 0
    ativos = decimar(resultado.n_active, freq, "max")
    pct = resultado.trial["threshold"]

    n = len(scores.columns)
    alturas = [1.0] * n + [0.6]
    fig, axes = plt.subplots(len(alturas), 1, sharex=True,
                             figsize=(LARGURA, ALTURA_PAINEL * (sum(alturas) * 0.95)),
                             gridspec_kw={"height_ratios": alturas})
    axes = np.atleast_1d(axes)
    alerta_d = decimar(resultado.alert.astype(float), freq, "max") > 0
    for i, sinal in enumerate(scores.columns):
        painel_score(axes[i], scores[sinal], limites[sinal], cruz[sinal], flags[sinal],
                     sinal, resultado.events, resultado.operability, freq, percentil=pct,
                     normalizar=normalizar,
                     metrica=NOME_METRICA.get(resultado.trial["model"], "Score de anomalia"))
        marcar_alerta(axes[i], alerta_d, rotulo="Alerta confirmado" if i == 0 else None)
        if i == 0:
            h, l = axes[i].get_legend_handles_labels()
            axes[i].legend(h, l, loc="upper left", fontsize=7, ncol=3, frameon=True)
        if i == 0:
            sombrear_paradas(axes[i], resultado.operability, freq, rotulo=True)
            axes[i].legend(loc="upper left", fontsize=7, ncol=3, frameon=True)

    ax = axes[-1]
    painel_confirmacao(ax, ativos, resultado.confirm, resultado.events,
                       resultado.operability, freq)
    falsos = set(resultado.false_positives)
    detectores = {ep for e in resultado.events for ep in e.get("episodios", [])}
    rug_episodios(ax, [e for e in resultado.episodes if e in detectores],
                  rotulo="Antecipou falha catalogada")
    rug_episodios(ax, [e for e in resultado.episodes
                       if e not in detectores and e not in falsos],
                  altura=0.88, cor=COR_EVENTO, rotulo="Antes de parada (não contado)")
    rug_episodios(ax, resultado.false_positives, altura=0.80, cor="0.35",
                  rotulo="Falso positivo")
    ax.set_xlabel("Tempo (data/hora)", fontsize=10)
    # legenda abaixo do eixo: os marcadores de episódio ocupam o topo do painel
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=5, frameon=False,
              fontsize=7)
    _formatar_eixo_x(ax, scores.index)
    fig.suptitle(titulo or f"Análise de anomalias — TC-330.03A | período completo | "
                 f"{resultado.label()}", fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------ a série crua, antes de qualquer modelo
def plot_series(sensores: pd.DataFrame, operabilidade: pd.DataFrame,
                eventos: list[dict], colunas: list[str] | None = None,
                freq: str | None = None, titulo: str | None = None):
    """A série de vários sensores no período todo, com as falhas marcadas.

    Não depende de modelo nenhum: é o material bruto, no mesmo formato visual das
    figuras de anomalia, para que uma coisa possa ser comparada com a outra.
    """
    colunas = [c for c in (colunas or list(sensores.columns)) if c in sensores.columns]
    freq = freq or auto_freq(sensores.index)
    janela = decimar(sensores[colunas], freq, "median")

    fig, axes = plt.subplots(len(colunas), 1, sharex=True,
                             figsize=(LARGURA, ALTURA_PAINEL * len(colunas) * 0.85))
    axes = np.atleast_1d(axes)
    vazio = pd.DatetimeIndex([])
    for i, coluna in enumerate(colunas):
        painel_sensor(axes[i], janela[coluna], vazio, eventos, operabilidade, freq,
                      primeiro=(i == 0))
    axes[-1].set_xlabel("Tempo (data/hora)", fontsize=10)
    _formatar_eixo_x(axes[-1], janela.index)
    fig.suptitle(titulo or "Série temporal — TC-330.03A | sensores de referência e "
                 "falhas reais", fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------------ as falhas ao longo do tempo
CORES_MECANISMO = {"mancal": "#c0392b", "oleo": "#2c7fb8", "selagem": "#8e44ad"}


def plot_falhas(eventos: list[dict], operabilidade: pd.DataFrame,
                alarmes: pd.DataFrame | None = None, freq: str = "1D",
                titulo: str | None = None, inicio=None, fim=None):
    """As falhas no decorrer do tempo, com o contexto que explica cada uma.

    Painel 1 — horas operadas por dia: mostra que a máquina passa longos períodos
    parada, e que "tempo de calendário" não é "tempo de exposição".
    Painel 2 — alarmes por dia, com os de nível (que derrubam a máquina) em
    vermelho: dá a densidade de alarme com que a operação convive.
    Painel 3 — a linha do tempo das falhas, uma faixa por mecanismo.
    """
    # a planilha de alarmes cobre 2022→2026, a série só 2025→2026: sem recortar,
    # o eixo estica três anos para trás e as falhas viram um amontoado à direita
    inicio = pd.Timestamp(inicio) if inicio is not None else operabilidade.index[0]
    fim = pd.Timestamp(fim) if fim is not None else operabilidade.index[-1]
    operabilidade = operabilidade.loc[inicio:fim]
    eventos = [e for e in eventos if inicio <= e["inicio"] <= fim]
    operando = operabilidade["in_operation"]
    passo = operando.index.to_series().diff().dt.total_seconds().median() or 30.0
    horas = operando.resample(freq).sum() * passo / 3600

    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(LARGURA, ALTURA_PAINEL * 2.6),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.7]})

    ax = axes[0]
    ax.bar(horas.index, horas.to_numpy(), width=1.0, color=COR_SERIE, alpha=0.85,
           label="Horas operadas por dia")
    marcar_eventos(ax, eventos)
    ax.set_ylabel("h operadas/dia", fontsize=9)
    ax.set_ylim(0, 25)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)

    ax = axes[1]
    if alarmes is not None and len(alarmes):
        ativos = alarmes.loc[inicio:fim]
        ativos = ativos[ativos["ativado"]]
        por_dia = ativos.resample(freq).size()
        nivel = ativos[ativos["nivel_trip"]].resample(freq).size().reindex(
            por_dia.index, fill_value=0)
        ax.bar(por_dia.index, por_dia.to_numpy(), width=1.0, color=COR_SERIE,
               alpha=0.75, label=f"Alarmes ativados por dia ({num(len(ativos), 0)})")
        ax.bar(nivel.index, nivel.to_numpy(), width=1.0, color=COR_ANOMALIA,
               label=f"…de nível (derruba a máquina) ({num(int(nivel.sum()), 0)})")
        ax.set_yscale("log")
    marcar_eventos(ax, eventos, rotulo=None)
    ax.set_ylabel("Alarmes/dia", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)

    ax = axes[2]
    mecanismos = sorted({e["mecanismo"] for e in eventos})
    for i, evento in enumerate(eventos):
        y = mecanismos.index(evento["mecanismo"])
        cor = CORES_MECANISMO.get(evento["mecanismo"], COR_ANOMALIA)
        ax.scatter([evento["inicio"]], [y], s=110, color=cor, zorder=5,
                   edgecolor="white", linewidth=0.8)
        # datas alternam acima/abaixo: 07/04 e 11/04/2025 ficam a 4 dias uma da
        # outra e os rótulos se sobreporiam
        desloca = 11 if i % 2 == 0 else -17
        ax.annotate(f"{evento['inicio']:%d/%m/%y}", xy=(evento["inicio"], y),
                    xytext=(0, desloca), textcoords="offset points", ha="center",
                    fontsize=7, color="0.25")
    marcar_eventos(ax, eventos, rotulo=None)
    ax.set_yticks(range(len(mecanismos)))
    ax.set_yticklabels(mecanismos, fontsize=8)
    ax.set_ylim(-0.6, len(mecanismos) - 0.1)
    ax.set_ylabel("Mecanismo", fontsize=9)
    ax.set_xlabel("Tempo (data/hora)", fontsize=10)

    for eixo in axes:
        sombrear_paradas(eixo, operabilidade, freq)
        eixo.set_xlim(inicio, fim)
    _formatar_eixo_x(axes[-1], pd.DatetimeIndex(horas.index))
    fig.suptitle(titulo or f"Falhas no decorrer do tempo — TC-330.03A | "
                 f"{len(eventos)} falhas em {horas.sum() / 24:.0f} dias de operação",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ----------------------------------------- falha ou alarme? a prova, trip a trip
def plot_evidencia_falhas(operabilidade: pd.DataFrame, trips: list[tuple[str, str]],
                          alarmes: pd.DataFrame | None = None, janela: str = "6h",
                          titulo: str | None = None):
    """Para cada trip: a máquina realmente parou, e o alarme disparou junto?

    Existe porque a nomenclatura confunde. Não há registro de manutenção neste
    projeto: o que chamamos de **falha** é derivado — alarme de nível (LL/HH)
    coincidente com uma parada real da máquina (``RUNNING_A`` 1→0). Esta figura
    mostra as duas evidências lado a lado, uma vez por trip, para que a definição
    possa ser conferida em vez de acreditada.
    """
    from .replay import duracao_da_parada

    operando = operabilidade["in_operation"].astype(float)
    delta = pd.Timedelta(janela)
    linhas = (len(trips) + 1) // 2
    fig, axes = plt.subplots(linhas, 2, figsize=(LARGURA, 1.5 * linhas), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for eixo, (quando, mecanismo) in zip(axes, trips):
        t = pd.Timestamp(quando)
        trecho = operando.loc[t - delta:t + delta]
        eixo.fill_between(trecho.index, 0, trecho.to_numpy(), step="post",
                          color=COR_SERIE, alpha=0.55, label="Máquina operando")
        eixo.axvline(t, color=COR_EVENTO, linestyle="--", linewidth=1.4,
                     label="Alarme de nível (trip)")
        if alarmes is not None:
            ativos = alarmes.loc[t - delta:t + delta]
            ativos = ativos[ativos["ativado"]]
            if len(ativos):
                eixo.scatter(ativos.index, np.full(len(ativos), 1.06), marker="|", s=70,
                             color=COR_ANOMALIA, clip_on=False,
                             label=f"Alarmes ativados ({len(ativos)})")
        parada = duracao_da_parada(operabilidade, t)
        rotulo_parada = (f"parada de {num(parada['horas'], 1)} h" if parada["parou"]
                         else "SEM parada nas 2 h seguintes")
        eixo.set_title(f"{t:%d/%m/%Y %H:%M} · {mecanismo} · {rotulo_parada}",
                       fontsize=8.5, fontweight="bold")
        eixo.set_ylim(-0.05, 1.15)
        eixo.set_yticks([0, 1])
        eixo.set_yticklabels(["parada", "operando"], fontsize=7)
        eixo.tick_params(axis="x", labelsize=7)
        eixo.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    for eixo in axes[len(trips):]:
        eixo.set_visible(False)
    # legenda única no rodapé: dentro dos painéis ela cobriria a faixa de operação
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels[:2] + ["Alarmes ativados na janela"],
               loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(titulo or "Falha ou alarme? — cada trip é um alarme de nível que "
                 "coincide com parada real da máquina",
                 fontsize=13, fontweight="bold", y=0.999)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------- erro de reconstrução por sensor
def plot_erro_por_sensor(erros: pd.DataFrame, evento: dict,
                         operabilidade: pd.DataFrame | None = None,
                         freq: str | None = None, titulo: str | None = None):
    """Abre o erro de reconstrução da família nos sensores que a compõem.

    Painel de cima: o **erro de reconstrução (MSE)** — a média sobre os sensores,
    que é exatamente o score daquela família.
    Painel de baixo: a **participação de cada sensor** no erro daquele instante
    (cada coluna soma 100%). Responde "quem está puxando o erro", que a média
    esconde.
    """
    freq = freq or auto_freq(erros.index, alvo=2000)
    dados = decimar(erros, freq, "mean")
    total = dados.mean(axis=1)
    participacao = dados.div(dados.sum(axis=1), axis=0).fillna(0.0)

    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(LARGURA, ALTURA_PAINEL * 2.1),
                             gridspec_kw={"height_ratios": [1.0, 1.3]})
    ax = axes[0]
    if operabilidade is not None:
        sombrear_paradas(ax, operabilidade.loc[dados.index[0]:dados.index[-1]], freq)
    ax.plot(total.index, total.to_numpy(), color=COR_SCORE, linewidth=1.1,
            label="Erro de reconstrução (MSE, média dos sensores) = score da família")
    ax.fill_between(total.index, 0, total.to_numpy(), color=COR_SCORE, alpha=0.15)
    marcar_eventos(ax, [evento])
    _ylim_robusto(ax, total, ate=evento["inicio"])
    ax.set_ylabel("Erro (MSE)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)

    ax = axes[1]
    # X e Y com o mesmo tamanho de C: com shading="auto" o matplotlib centra as
    # células nos pontos, que é o que faz o rótulo do sensor cair na linha certa
    malha = ax.pcolormesh(participacao.index, np.arange(len(participacao.columns)),
                          participacao.to_numpy().T, cmap="Reds", vmin=0, vmax=1,
                          shading="auto")
    ax.set_yticks(np.arange(len(participacao.columns)))
    ax.set_yticklabels([c.replace("954005_624_", "") for c in participacao.columns],
                       fontsize=7)
    ax.axvline(evento["inicio"], color=COR_EVENTO, linestyle="--", linewidth=1.4)
    ax.set_ylabel("Sensor", fontsize=9)
    ax.set_xlabel("Tempo (data/hora)", fontsize=10)
    ax.grid(False)
    barra = fig.colorbar(malha, ax=ax, pad=0.01, fraction=0.03)
    barra.set_label("participação no erro do instante", fontsize=8)
    barra.ax.tick_params(labelsize=7)
    _formatar_eixo_x(ax, dados.index)

    fig.suptitle(titulo or f"Erro de reconstrução por sensor — "
                 f"{evento['inicio']:%d/%m/%Y %H:%M}, falha de {evento['mecanismo']}",
                 fontsize=13, fontweight="bold", y=0.999)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# -------------------------------------------------- o score e os limiares
def plot_limiares(resultado, titulo: str | None = None, bins: int = 90):
    """Distribuição do score de cada sinal com o limite de cada mês marcado.

    Dá a noção que a série temporal não dá: **onde** o limite cai dentro da
    distribuição do score, e quanto do tempo fica acima dele. O limite é um
    percentil do baseline, então há um por mês — daí o feixe de linhas.
    """
    scores, limites = resultado.scores, resultado.limits
    pct = resultado.trial["threshold"]
    n = len(scores.columns)
    fig, axes = plt.subplots(n, 1, figsize=(LARGURA, ALTURA_PAINEL * n * 0.72))
    axes = np.atleast_1d(axes)

    for eixo, sinal in zip(axes, scores.columns):
        valores = scores[sinal].dropna()
        valores = valores[valores > 0]
        if valores.empty:
            continue
        eixo.hist(valores.to_numpy(), bins=np.logspace(
            np.log10(valores.min()), np.log10(valores.max()), bins),
            color=COR_SCORE, alpha=0.55,
            label=f"{NOME_METRICA.get(resultado.trial['model'], 'Score de anomalia')}: {sinal}")
        mensais = limites[sinal].dropna().unique()
        for i, limite in enumerate(sorted(mensais)):
            eixo.axvline(limite, color=COR_LIMITE, linestyle="--", linewidth=0.9,
                         alpha=0.45,
                         label=(f"Limite de cada mês (percentil {_pct(pct)} do "
                                f"baseline) — {len(mensais)} meses") if i == 0 else None)
        mediana = float(np.median(mensais))
        eixo.axvline(mediana, color=COR_LIMITE, linewidth=1.8,
                     label=f"Limite mediano = {num(mediana, 3)}")
        # só o tempo com score definido (isto é, em operação): incluir as paradas
        # no denominador diluiria a conta
        valido = scores[sinal].notna()
        acima = float((scores[sinal] > limites[sinal])[valido].mean())
        eixo.annotate(f"{acima:.1%} do tempo em operação acima do limite",
                      xy=(0.995, 0.88), xycoords="axes fraction", ha="right",
                      fontsize=8, color="0.3")
        eixo.set_xscale("log")
        eixo.set_ylabel("Amostras", fontsize=9)
        eixo.legend(loc="upper left", fontsize=7, ncol=3, frameon=True)
    axes[-1].set_xlabel(
        f"{NOME_METRICA.get(resultado.trial['model'], 'Score de anomalia')} (escala log)",
        fontsize=10)
    fig.suptitle(titulo or "Onde cai o limite dentro da distribuição do score — "
                 "TC-330.03A", fontsize=14, fontweight="bold", y=0.999)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------------- sensor único no período todo
def plot_sensor(resultado, coluna: str, alarmes: pd.Series | None = None,
                inicio=None, fim=None, freq: str | None = None,
                titulo: str | None = None):
    """Um sensor no período inteiro, com anomalias do modelo e falhas reais.

    Com ``alarmes`` (índice de tempo das ativações da planilha), a figura fica
    sendo a comparação direta: o que o modelo marcou × o que a planilha
    registrou × quando a máquina de fato caiu.
    """
    serie = resultado.sensors[coluna].loc[inicio:fim]
    freq = freq or auto_freq(serie.index)
    serie_d = decimar(serie, freq, "median")
    alerta_d = decimar(resultado.alert.loc[inicio:fim].astype(float), freq, "max")
    anomalias = alerta_d[alerta_d > 0].index
    eventos = [e for e in resultado.events
               if (inicio is None or e["inicio"] >= pd.Timestamp(inicio))
               and (fim is None or e["inicio"] <= pd.Timestamp(fim))]

    fig, ax = plt.subplots(figsize=(LARGURA, ALTURA_PAINEL * 1.6))
    painel_sensor(ax, serie_d, anomalias, eventos,
                  resultado.operability.loc[inicio:fim], freq, primeiro=True)
    if alarmes is not None and len(alarmes):
        dentro = alarmes[(alarmes >= serie_d.index[0]) & (alarmes <= serie_d.index[-1])]
        if len(dentro):
            baixo, _ = ax.get_ylim()
            ax.scatter(dentro, np.full(len(dentro), baixo), marker="|", s=90,
                       color=COR_SERIE, alpha=0.6, label="Alarme na planilha")
            ax.legend(loc="upper left", fontsize=7, ncol=4, frameon=True)
    ax.set_xlabel("Tempo (data/hora)", fontsize=10)
    _formatar_eixo_x(ax, serie_d.index)
    fig.suptitle(titulo or f"Análise de anomalias — TC-330.03A | {coluna}",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# --------------------------------------------------------- duração dos alertas
CORES_TIPO = {"detecção": COR_ANOMALIA,
              "antes de parada (não contado)": COR_EVENTO,
              "falso positivo": "0.35"}


def plot_episodios(resultado, titulo: str | None = None):
    """Cada episódio de alerta pelo **tempo que durou**.

    A métrica de falso positivo conta episódios, não duração — e é aí que mora a
    diferença entre um alarme de 10 minutos e um alarme travado por uma semana.
    """
    tabela = resultado.episodes_table()
    fig, ax = plt.subplots(figsize=(LARGURA, ALTURA_PAINEL * 1.6))
    for tipo, cor in CORES_TIPO.items():
        parte = tabela[tabela["tipo"] == tipo]
        if parte.empty:
            continue
        ax.scatter(parte["inicio"], parte["duracao_h"].clip(lower=0.05), s=48, color=cor,
                   edgecolor="white", linewidth=0.6, zorder=4,
                   label=f"{tipo} ({len(parte)})")
        ax.vlines(parte["inicio"], 0.05, parte["duracao_h"].clip(lower=0.05),
                  color=cor, alpha=0.35, linewidth=1.2)
    for referencia, texto in ((1.0, "1 h"), (24.0, "1 dia"), (168.0, "1 semana")):
        ax.axhline(referencia, color="0.6", linestyle=":", linewidth=1.0)
        ax.annotate(texto, xy=(0.002, referencia), xycoords=("axes fraction", "data"),
                    fontsize=7, color="0.4", va="bottom")
    marcar_eventos(ax, resultado.events)
    ax.set_xlim(resultado.scores.index[0], resultado.scores.index[-1])
    ax.set_yscale("log")
    ax.set_ylabel("Duração do alerta (h)", fontsize=9)
    ax.set_xlabel("Tempo (data/hora)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=4, frameon=True)
    _formatar_eixo_x(ax, pd.DatetimeIndex(resultado.scores.index))
    fig.suptitle(titulo or "Análise de anomalias — TC-330.03A | duração de cada alerta",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=RECT_TITULO)
    return fig


# ------------------------------------------------------------------ placar
def plot_placar(resultado, titulo: str | None = None):
    """Acertos, perdas e falsos positivos em uma figura só (leitura de gestão)."""
    tabela = resultado.events_table()
    fig, axes = plt.subplots(1, 2, figsize=(LARGURA, ALTURA_PAINEL * 1.8),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    ax = axes[0]
    rotulos = [f"{pd.Timestamp(t):%d/%m/%y}\n{m}"
               for t, m in zip(tabela["falha"], tabela["mecanismo"])]
    leads = tabela["antecedencia_h"].fillna(0.0).to_numpy()
    cores = [COR_ANOMALIA if d else "0.75" for d in tabela["detectada"]]
    barras = ax.bar(rotulos, leads, color=cores, edgecolor="white")
    for barra, detectada, lead in zip(barras, tabela["detectada"], tabela["antecedencia_h"]):
        texto = f"{num(lead, 0)} h" if detectada else "não detectada"
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.8,
                texto, ha="center", fontsize=8,
                color="black" if detectada else "0.4")
    ax.set_ylabel("Antecedência do alerta (h)", fontsize=9)
    ax.set_ylim(0, max(float(leads.max()), 1.0) * 1.25)
    ax.set_title(f"Falhas reais: {int(tabela['detectada'].sum())} de {len(tabela)} "
                 f"antecipadas", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[1]
    m = resultado.metrics
    detec = m["eventos_detectados"]
    perdidas = m["eventos_total"] - detec
    fp = m["fp_episodios"]
    ax.bar(["antecipadas", "perdidas", "falsos\npositivos"], [detec, perdidas, fp],
           color=[COR_ANOMALIA, "0.75", "0.35"], edgecolor="white")
    for i, valor in enumerate([detec, perdidas, fp]):
        ax.text(i, valor + max(fp, 1) * 0.03, str(valor), ha="center", fontsize=9)
    ax.set_title(f"{num(m['fp_por_mes'])}/mês de falso positivo em "
                 f"{num(m['dias_avaliados'], 0)} dias de operação",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Episódios", fontsize=9)
    fig.suptitle(titulo or f"Análise de anomalias — TC-330.03A | {resultado.label()}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
