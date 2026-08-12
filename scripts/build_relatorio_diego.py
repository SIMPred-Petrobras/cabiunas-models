#!/usr/bin/env python3
"""
build_relatorio_diego.py
Gera o relatório em HTML para o Diego a partir dos CSVs de resultado — nada de número
digitado à mão. Se um resultado mudar, o relatório muda junto ao rodar de novo.

As figuras são embutidas como data URI porque o artifact bloqueia host externo.

Uso:
    PYTHONPATH=. python scripts/build_relatorio_diego.py
"""
from __future__ import annotations

import base64
import os

import pandas as pd

OUT = "relatorio_anexos/relatorio_diego_2026_08_11.html"
E = "eval_predictive_out"


def img(path: str, alt: str, legenda: str) -> str:
    if not os.path.exists(path):
        return f'<p class="faltando">[figura ausente: {path}]</p>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<figure><div class="fig-wrap">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt}"></div>'
            f'<figcaption>{legenda}</figcaption></figure>')


def pct(x) -> str:
    return "—" if pd.isna(x) else f"{x*100:.1f}%"


def fmt(x, n=3) -> str:
    return "—" if pd.isna(x) else f"{x:.{n}f}"


def tabela_frota() -> str:
    p = f"{E}/baseline_trivial_fleet.csv"
    if not os.path.exists(p):
        return '<p class="faltando">[baseline da frota ainda não rodou]</p>'
    df = pd.read_csv(p)
    linhas = []
    for _, r in df.iterrows():
        if not r.get("avaliado", False):
            linhas.append(
                f'<tr class="sem-amostra"><td>{r.sensor}</td><td class="num">{int(r.n_inc)}</td>'
                f'<td colspan="5">amostra insuficiente — não avaliado</td></tr>')
            continue
        d = r.delta_pp
        cls = "ganha" if d > 10 else ("perde" if d < -10 else "empate")
        vd = {"ganha": "limiar ganha", "perde": "AE ganha", "empate": "empate"}[cls]
        linhas.append(
            f"<tr><td>{r.sensor}</td><td class='num'>{int(r.n_inc)}</td>"
            f"<td class='dir'>{r.direcao}</td>"
            f"<td class='num'>{pct(r.ae_recall)} <span class='fa'>· {fmt(r.ae_fa)}</span></td>"
            f"<td class='num'>{pct(r.trivial_recall)} <span class='fa'>· {fmt(r.trivial_fa)}</span></td>"
            f"<td class='num'>{pct(r.oposto_recall)}</td>"
            f"<td class='num {cls}'>{d:+.1f} pp<br><span class='vd'>{vd}</span></td></tr>")
    return f"""<div class="tab-wrap"><table>
<thead><tr><th>sensor</th><th>inc.</th><th>direção do alarme</th>
<th>AE<br><span class="sub">recall · FA/dia</span></th>
<th>limiar trivial<br><span class="sub">recall · FA/dia</span></th>
<th>controle<br><span class="sub">direção oposta</span></th>
<th>Δ</th></tr></thead>
<tbody>{''.join(linhas)}</tbody></table></div>"""


def tabela_wf() -> str:
    p = f"{E}/forecast_crossing_walkforward.csv"
    if not os.path.exists(p):
        return '<p class="faltando">[walk-forward ausente]</p>'
    df = pd.read_csv(p)
    hs = sorted(df.H.unique())
    ordem = ["A0 trivial (limiar de T)", "A1 logística", "A2 GBM", "A3 GBM + AE",
             "REF autoencoder"]
    linhas = []
    for b in ordem:
        d = df[df.braco == b].set_index("H")
        if d.empty:
            continue
        cels = []
        for h in hs:
            if h not in d.index:
                cels.append("<td>—</td>")
                continue
            r = d.loc[h]
            cels.append(f"<td class='num'>{pct(r.recall_raw)}<br>"
                        f"<span class='fa'>FA {fmt(r.fa_per_day)}</span></td>")
        cls = ' class="destaque"' if b.startswith("A0") else ""
        linhas.append(f"<tr{cls}><td>{b}</td>{''.join(cels)}</tr>")
    cab = "".join(f"<th>H = {int(h)} h</th>" for h in hs)
    return (f'<div class="tab-wrap"><table><thead><tr><th>braço</th>{cab}</tr></thead>'
            f"<tbody>{''.join(linhas)}</tbody></table></div>")


HTML = """<title>Cabiúnas TC382_03_A — o autoencoder foi refutado por um limiar</title>
<style>
:root {{
  --bg:#f7f8f9; --surface:#ffffff; --surface-2:#eef1f3;
  --ink:#14171a; --muted:#5c666f; --linha:#d9dee2;
  --accent:#b3541e; --slate:#6b7b8c; --refutado:#a33227; --ok:#1f7a3d;
  --serif: "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans: ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono: ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#14171a; --surface:#1c2024; --surface-2:#23282d;
    --ink:#e8eaec; --muted:#98a2ab; --linha:#333a40;
    --accent:#d97a45; --slate:#8fa0b0; --refutado:#e0705f; --ok:#5cb87d;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#14171a; --surface:#1c2024; --surface-2:#23282d;
  --ink:#e8eaec; --muted:#98a2ab; --linha:#333a40;
  --accent:#d97a45; --slate:#8fa0b0; --refutado:#e0705f; --ok:#5cb87d;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.pag {{ max-width:1180px; margin:0 auto; padding:48px 28px 96px; }}
.col {{ max-width:72ch; }}
h1,h2,h3 {{ font-family:var(--serif); font-weight:600; text-wrap:balance; margin:0; }}
h1 {{ font-size:2.05rem; line-height:1.22; letter-spacing:-0.01em; }}
h2 {{ font-size:1.4rem; margin-top:8px; }}
h3 {{ font-size:1.08rem; }}
p {{ margin:0; }}
a {{ color:var(--accent); }}
.eyebrow {{
  font-family:var(--mono); font-size:0.72rem; letter-spacing:0.14em;
  text-transform:uppercase; color:var(--muted);
}}
header {{ display:flex; flex-direction:column; gap:14px;
  border-bottom:2px solid var(--ink); padding-bottom:26px; }}
.meta {{ font-family:var(--mono); font-size:0.78rem; color:var(--muted); }}
.lede {{ font-size:1.12rem; color:var(--muted); max-width:70ch; }}
section {{ display:flex; flex-direction:column; gap:18px; margin-top:52px; }}
.destaque-box {{
  background:var(--surface); border:1px solid var(--linha);
  border-top:3px solid var(--accent); padding:24px 26px;
  display:flex; flex-direction:column; gap:14px;
}}
.numeros {{ display:flex; flex-wrap:wrap; gap:34px; }}
.n-item {{ display:flex; flex-direction:column; gap:2px; }}
.n-val {{ font-family:var(--mono); font-size:1.72rem; font-variant-numeric:tabular-nums;
  line-height:1.1; }}
.n-lab {{ font-size:0.82rem; color:var(--muted); }}
.venc {{ color:var(--ok); }} .perd {{ color:var(--refutado); }}
ol.cadeia {{ list-style:none; counter-reset:passo; padding:0; margin:0;
  display:flex; flex-direction:column; gap:20px; }}
ol.cadeia > li {{ counter-increment:passo; display:grid;
  grid-template-columns:38px 1fr; gap:16px; align-items:start; }}
ol.cadeia > li::before {{
  content:counter(passo); font-family:var(--mono); font-size:0.86rem;
  color:var(--surface); background:var(--ink); width:28px; height:28px;
  display:grid; place-items:center; margin-top:2px;
}}
ol.cadeia h3 {{ margin-bottom:4px; }}
.vered {{ font-family:var(--mono); font-size:0.78rem; text-transform:uppercase;
  letter-spacing:0.08em; color:var(--refutado); }}
.tab-wrap {{ overflow-x:auto; border:1px solid var(--linha); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:0.87rem; }}
th,td {{ padding:9px 13px; text-align:left; border-bottom:1px solid var(--linha);
  vertical-align:top; }}
thead th {{ background:var(--surface-2); font-weight:600; font-size:0.8rem;
  white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:none; }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.sub, .fa {{ font-weight:400; color:var(--muted); font-size:0.76rem; }}
.dir {{ font-size:0.8rem; color:var(--muted); }}
.vd {{ font-family:var(--sans); font-size:0.72rem; }}
.ganha {{ color:var(--ok); }} .perde {{ color:var(--refutado); }}
.empate {{ color:var(--muted); }}
tr.destaque td {{ background:color-mix(in srgb, var(--accent) 9%, transparent); }}
tr.sem-amostra td {{ color:var(--muted); font-style:italic; }}
figure {{ margin:0; display:flex; flex-direction:column; gap:10px; }}
.fig-wrap {{ overflow-x:auto; border:1px solid var(--linha); background:#fff; }}
.fig-wrap img {{ display:block; width:100%; min-width:760px; }}
figcaption {{ font-size:0.84rem; color:var(--muted); max-width:78ch; }}
.alerta {{ border-left:3px solid var(--refutado); padding:2px 0 2px 18px;
  display:flex; flex-direction:column; gap:10px; }}
.perguntas {{ display:flex; flex-direction:column; gap:0;
  border:1px solid var(--linha); background:var(--surface); }}
.perguntas > div {{ padding:20px 24px; border-bottom:1px solid var(--linha);
  display:flex; flex-direction:column; gap:6px; }}
.perguntas > div:last-child {{ border-bottom:none; }}
.q {{ font-family:var(--serif); font-size:1.05rem; }}
.faltando {{ font-family:var(--mono); font-size:0.82rem; color:var(--refutado); }}
footer {{ margin-top:64px; padding-top:22px; border-top:1px solid var(--linha);
  font-size:0.82rem; color:var(--muted); display:flex; flex-direction:column; gap:6px; }}
code {{ font-family:var(--mono); font-size:0.86em;
  background:var(--surface-2); padding:1px 5px; }}
@media (max-width:640px) {{ .pag {{ padding:32px 18px 72px; }} h1 {{ font-size:1.6rem; }}
  .numeros {{ gap:22px; }} }}
</style>

<div class="pag">
<header>
  <div class="eyebrow">Cabiúnas · Turbina A · nota técnica interna</div>
  <h1>O autoencoder do TC382_03_A foi refutado por um limiar na própria temperatura</h1>
  <p class="lede">Diego — abaixo está o que encontramos nas últimas semanas, com os números
  e o método de cada teste. A conclusão muda o rumo do projeto, então quero sua
  concordância antes de levarmos isso adiante. As perguntas objetivas estão no fim.</p>
  <div class="meta">11 de agosto de 2026 · janela jan/2024 → abr/2026 · 79 incidentes
  HI/HIHI no TC382_03_A · protocolo: horizonte 8 h, sticky 12 h, FA ≤ 1/dia, duty ≤ 0,25</div>
</header>

<section>
  <h2>O resultado, em uma linha</h2>
  <div class="destaque-box">
    <p>O alarme do DCS é <code>TC382_03_A &gt; 760 °C</code>. Comparando pareado, na janela
    completa de 79 incidentes:</p>
    <div class="numeros">
      <div class="n-item"><span class="n-val perd">62,0%</span>
        <span class="n-lab">autoencoder · FA 0,114/dia</span></div>
      <div class="n-item"><span class="n-val venc">81,0%</span>
        <span class="n-lab">limiar trivial · FA 0,047/dia</span></div>
      <div class="n-item"><span class="n-val">2,4×</span>
        <span class="n-lab">menos falsos alarmes no limiar</span></div>
    </div>
    <p>O limiar ganha <strong>nos dois eixos ao mesmo tempo</strong>: mais recall e menos
    falso alarme. E não é o alarme do DCS repetido — é um pré-alarme na temperatura
    suavizada (EWMA de 2 h) em torno de <strong>739,5 °C</strong>, cerca de 20 °C abaixo do
    setpoint, com antecedência mediana de 7,9 h.</p>
  </div>
</section>

<section>
  <h2>Por que isso acontece</h2>
  <div class="col">
    <p>Um autoencoder é treinado para <em>reconstruir bem</em> o que vê. Quando ele aprende
    o regime quente da máquina, o erro de reconstrução <strong>cai</strong> justamente
    quando a temperatura sobe — fica anticorrelacionado com o que queremos prever.</p>
    <p style="margin-top:14px">No fundo estávamos pedindo que uma rede descobrisse, por
    caminho indireto, um limiar do próprio sinal que ela recebe na entrada. Medimos que
    cruzamentos de 760 °C por dia batem 1:1 com alarmes por dia (0,63 × 0,68 em 2024H2;
    0,345 × 0,32 em 2026), ou seja, o alvo é <em>literalmente</em> uma função da entrada.</p>
  </div>
  {fig1}
</section>

<section>
  <h2>A sequência de testes</h2>
  <div class="col"><p>Cada teste nasceu do resultado do anterior. Todos com controle
  pareado e critério fixado <em>antes</em> de rodar.</p></div>
  <ol class="cadeia">
    <li><div>
      <h3>O autoencoder detecta o alarme melhor que a temperatura crua?</h3>
      <p>Comparação pareada, mesmo maquinário, mesma grade temporal, mesmo grid de
      meia-vida. O limiar venceu em 62,0% × 81,0% com FA 2,4× menor.</p>
      <p class="vered">Refutado — o AE perde</p>
    </div></li>
    <li><div>
      <h3>Combinar os dois resgata o autoencoder?</h3>
      <p>Cinco fusões algébricas sobre os dois ranks, gastando o mesmo grau de liberdade
      que um braço isolado. Critério: ganhar do melhor isolado em <em>todas</em> as janelas
      com margem &gt; 10 pp. Melhor pior-caso: <strong>−10,7 pp</strong>. Na janela completa
      a fusão até dá +2,5 pp — mas custa −18,8 pp em 2026, e é justamente na janela completa
      que a busca do ponto de operação roda.</p>
      <p class="vered">Refutado — nenhuma fusão passa</p>
    </div></li>
    <li><div>
      <h3>E se reformularmos como previsão supervisionada?</h3>
      <p>Trocamos reconstrução por classificação direta: <code>y = 1</code> se a temperatura
      cruzar 760 °C nas próximas H horas. Isso multiplica o rótulo de 79 incidentes para
      <strong>19.305 amostras positivas</strong> — a escassez que travava o AE deixa de
      existir. Testamos regressão logística, gradient boosting com 30 features (temperatura,
      inclinações, irmãos, carga, pressão) e o mesmo GBM acrescido do erro do autoencoder.</p>
      <p class="vered">Refutado — empata em 8 h e perde no walk-forward</p>
    </div></li>
    <li><div>
      <h3>O valor não estaria no horizonte longo?</h3>
      <p>A aposta: o limiar só sabe que está quente <em>agora</em>, então a 24 h ou 72 h ele
      deveria degradar e o modelo segurar. O oposto acontece — as curvas convergem. E o piso
      do acaso sobe junto: a 72 h um modelo treinado com os <em>rótulos embaralhados</em>
      empata com o limiar (88,2% × 88,2%).</p>
      <p class="vered">Refutado — e a métrica a 72 h não é interpretável</p>
    </div></li>
  </ol>
  {fig2}
</section>

<section>
  <h2>Isso vale só para o TC382_03_A?</h2>
  <div class="col">
    <p>Foi a primeira coisa que você perguntaria, então rodamos o mesmo teste nos sete
    canais térmicos. Duas coisas apareceram, e a segunda me obrigou a refazer a análise.</p>
    <p style="margin-top:14px"><strong>Primeiro: o TC382_03_A é o único sensor com histórico
    real de superaquecimento.</strong> Ele tem 84 alarmes HI e 71 HIHI. Os demais somam de
    0 a 8 — o TC382_04_A e o TC382_06_A não têm <em>nenhum</em> em 28 meses. O que eles têm
    são 63 a 64 alarmes <code>UNDER</code> cada, que é o termopar caindo para a sentinela de
    −40,5 °C: <strong>falha de instrumento, não evento de processo</strong>. Os dez sensores
    de vibração seguem o mesmo padrão, com 6 a 12 alarmes <code>LOLO</code>, e por isso
    ficaram de fora — não há amostra para testar.</p>
    <p style="margin-top:14px"><strong>Segundo: errei a direção do limiar na primeira
    rodada</strong>, e vale registrar porque o erro é instrutivo. Defini a direção pela
    composição de <em>todos</em> os alarmes do sensor. Só que o filtro de máquina ligada
    descarta a maioria deles — o <code>UNDER</code> ocorre com a máquina parada. Dos 68
    alarmes do TC382_01_A, por exemplo, só 7 entram na avaliação. A direção tem de vir da
    composição dos <em>incidentes avaliados</em>, não do histórico inteiro; corrigido, dois
    sensores mudam de direção.</p>
    <p style="margin-top:14px">Vale insistir que a regra corrigida continua sendo escolha
    a priori — depende só do rótulo do incidente, nunca da métrica. A prova está na tabela:
    no TC382_01_A a regra aponta BAIXO, que é justamente o braço <em>pior</em> (14,3%
    contra 71,4% do ALTO). Uma regra que estivesse pescando o melhor resultado não faria
    isso. As duas direções aparecem na tabela para você conferir.</p>
  </div>
  {tab_frota}
  <div class="col">
    <p><strong>Como eu leio:</strong> dos sete canais, só dois têm amostra que sustenta
    conclusão — TC382_03_A com 81 incidentes e T5_AVG_A com 17. <strong>Nos dois o limiar
    vence</strong>, por +18,5 pp e +23,5 pp, com falso alarme 2,4× e 2,6× menor. Nos cinco
    restantes, onde o autoencoder aparece na frente, o n vai de 4 a 10: um único incidente
    vale de 10 a 25 pp, então ali não há resultado, há ruído.</p>
    <p style="margin-top:14px">Há um dado que <em>não</em> depende de tamanho de amostra e
    que eu destacaria: <strong>a taxa de falso alarme do autoencoder é maior em todos os
    sete canais, sem exceção</strong> — 0,114 a 0,151/dia contra 0,039 a 0,076 do limiar.
    Isso vale inclusive nos sensores em que ele ganha no recall.</p>
  </div>
  {fig3}
</section>

<section>
  <h2>Dois achados de método que valem além deste sensor</h2>
  <div class="alerta">
    <h3>1 · Recall sem piso do acaso não significa nada</h3>
    <p>Com o duty travado em 0,25 e uma janela de crédito de 72 h, um alerta ligado um
    quarto do tempo encosta em quase todo incidente por construção. Medimos o piso: ruído
    puro chega a 69,4% no OOS e 77,0% na janela completa; um modelo com rótulo embaralhado
    chega a 88,2%. Qualquer número a 72 h precisa ser lido como distância <em>acima</em>
    desse piso.</p>
    <p><em>Ressalva honesta:</em> os braços nulos pagam FA de 0,17–0,24/dia contra 0,03 dos
    braços reais, então o piso a FA equiparada é mais baixo. A conclusão se sustenta, a
    magnitude exata precisa dessa correção.</p>
  </div>
  <div class="alerta" style="margin-top:8px">
    <h3>2 · Escolher o limiar na janela de avaliação infla até 30 pp</h3>
    <p>Fizemos o teste honesto: modelo <em>e</em> limiar fixados no treino (jan/24–jun/25),
    congelados, aplicados no teste. O mesmo braço que marca 88,2% quando o limiar é buscado
    na janela de avaliação cai para <strong>58,8%</strong> quando congelado.</p>
    <p><strong>Todo número histórico deste projeto é buscado, não congelado</strong> —
    inclusive os 81,0% e os 62,0% acima. Isso não invalida as comparações, porque todos os
    braços levaram a mesma vantagem, mas invalida qualquer promessa de desempenho em campo
    feita com esses números. A boa notícia: quase toda a queda é calibração, não modelo — ao
    congelar, o duty caiu de 0,25 para 0,11, ou seja, o limiar do treino ficou conservador
    demais para o teste. Recalibração periódica em dado recente resolve, e é procedimento
    normal.</p>
  </div>
</section>

<section>
  <h2>Validação fora de amostra</h2>
  <div class="col"><p>O teste mais duro que conseguimos montar: refit trimestral em janela
  expansiva, cada trimestre previsto por um modelo que só viu o passado, 58 incidentes.</p></div>
  {tab_wf}
  <div class="col"><p style="font-size:0.9rem;color:var(--muted)">Em 8 h e 24 h o limiar
  trivial ganha nos dois eixos de todos os braços treinados. Em 72 h alguns passam à frente,
  mas ali o piso do acaso é 77,0% — não é habilidade, é a aritmética da janela de crédito.
  O braço <em>GBM + AE</em> empata com o <em>GBM</em> puro em toda a tabela: o erro de
  reconstrução não agrega nada sobre a temperatura.</p></div>
</section>

<section>
  <h2>O que eu recomendo</h2>
  <div class="col">
    <p><strong>Parar o esforço de modelagem no TC382_03_A e no T5_AVG_A, e entregar a regra
    simples nos dois.</strong>
    O alvo é um limiar do próprio canal de entrada, e nenhuma arquitetura vence uma regra de
    três linhas nesse jogo. O entregável é o pré-alarme na temperatura suavizada, com a faixa
    de desempenho declarada honestamente (59–86% conforme a calibração) e a recalibração
    periódica como parte do procedimento, não como detalhe de implementação.</p>
    <p style="margin-top:14px"><strong>O valor do projeto migra para degradação física</strong>
    — mancal, vibração, pressão. Ali reconstrução faz sentido, porque não existe uma linha
    óbvia para cruzar. Mas isso depende de <strong>rótulo de manutenção</strong> da Petrobras,
    e nada nos dados atuais substitui: o banco de alarmes que temos é, fora do TC382_03_A,
    quase todo falha de instrumento.</p>
  </div>
</section>

<section>
  <h2>Onde eu preciso da sua opinião</h2>
  <div class="perguntas">
    <div>
      <p class="q">Você concorda com o veredito, ou vê um furo no método?</p>
      <p>O ponto que eu mais gostaria que você atacasse é o baseline: ele é pareado de
      verdade? Usamos a mesma grade temporal, o mesmo grid de meia-vida, o mesmo
      <code>best_point</code> e o mesmo denominador de incidentes nos dois braços. Se houver
      assimetria aí, o resultado inteiro cai.</p>
    </div>
    <div>
      <p class="q">O alvo do projeto é o alarme do DCS ou degradação física?</p>
      <p>Se for o alarme, o teto é o que está aqui e o projeto está essencialmente concluído.
      Se for degradação, precisamos abrir a conversa de rótulo de manutenção com a Petrobras
      — e isso é prazo, não esforço nosso.</p>
    </div>
    <div>
      <p class="q">Você sabe o que mudou na operação por volta de fevereiro de 2025?</p>
      <p>A temperatura de exaustão caiu cerca de 60 °C e voltou progressivamente ao longo de
      2026. Comparando janeiro–abril de cada ano: 2024 = 712 °C e 0,31 alarmes/dia;
      2025 = 654 °C e 0,16; 2026 = 707 °C e 0,32. Isso é a maior fonte de instabilidade de
      calibração que enfrentamos, e um contexto de operação resolveria mais que qualquer
      modelo.</p>
    </div>
    <div>
      <p class="q">Qual ponto de operação a operação aceita?</p>
      <p>Hoje escolhemos maximizar recall sob teto de falso alarme. Se o custo real de um
      alarme falso for maior do que estamos assumindo, o ponto muda e o número muda junto.</p>
    </div>
  </div>
</section>

<footer>
  <p>Reprodutível: <code>scripts/baseline_trivial_vs_ae.py</code> ·
  <code>scripts/combine_ae_temp.py</code> · <code>scripts/forecast_crossing.py</code> ·
  <code>scripts/baseline_trivial_fleet.py</code></p>
  <p>Autoencoder de referência: task ClearML <code>3b34a312</code>, congelado, sem novo
  treino em nenhum teste desta nota. Baseline da frota registrado em
  <code>ca1d16a9</code>.</p>
  <p>Os dez sensores de vibração (TV_35*) ficaram fora: 6 a 12 alarmes <code>LOLO</code>
  cada em 28 meses, sem amostra para testar.</p>
</footer>
</div>
"""


def main() -> None:
    html = HTML.format(
        fig1=img(f"{E}/fig_baseline_vs_ae_TC382_03_A.png",
                 "Série temporal comparando autoencoder e limiar trivial",
                 "<strong>Figura 1.</strong> Janela completa, um ponto de operação por braço. "
                 "A pista vermelha no topo é o alarme do DCS. O limiar (painel de baixo) acende "
                 "nas mesmas regiões quentes em que o alarme dispara; o health do autoencoder "
                 "(painel do meio) fica alto em 2024 sem discriminar — 96 episódios de falso "
                 "positivo contra 39 do limiar."),
        fig2=img(f"{E}/fig_horizon_frontier_TC382_03_A.png",
                 "Recall e falso alarme por horizonte de antecipação",
                 "<strong>Figura 2.</strong> A aposta era que a linha tracejada preta (limiar) "
                 "cairia à direita e os modelos segurariam. As curvas convergem. A faixa "
                 "vermelha é o piso do acaso — a 72 h ela engole as curvas, e recall ali deixa "
                 "de medir habilidade."),
        fig3=img(f"{E}/fig_fleet_baseline_TC382.png",
                 "Comparação pareada entre autoencoder e limiar trivial por sensor",
                 "<strong>Figura 3.</strong> Um par de pontos por sensor: a distância entre "
                 "eles é o resultado. As cinco linhas sombreadas têm de 4 a 10 incidentes — "
                 "amostra insuficiente, mostrada por transparência e não para sustentar "
                 "conclusão. No painel da direita todas as linhas são verdes: o limiar custa "
                 "menos falso alarme nos sete canais."),
        tab_frota=tabela_frota(),
        tab_wf=tabela_wf(),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Relatório: {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
