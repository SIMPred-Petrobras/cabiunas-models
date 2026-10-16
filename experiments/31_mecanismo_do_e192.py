#!/usr/bin/env python3
"""EXP 31 — O QUE O `escalada_idade_h = 192` FAZ DE FATO? (diagnóstico do mecanismo, episódio por episódio, e juízo de coerência)

REGISTRO (16/10/2026). Este é um diagnóstico PÓS-HOC, pedido pelo usuário ('como especialista, acha coerente esse ajuste?'), escrito DEPOIS de ler `prever` em `cabiunas_inference.py`; não há critério pré-registrado, só a descrição do mecanismo com os números do harness de 8 eventos.

O MECANISMO NO CÓDIGO. `escalada_idade_h` entra em DOIS lugares de `prever`: (1) REANÚNCIO: um voto contínuo que já dura >= escalada_idade_h e de repente fica 'forte' (força > escalada_abs = 20, em unidades do limiar alto) é CORTADO em dois episódios, para que o segundo nasça e a operação receba notícia nova; (2) FURO DO REFRATÁRIO: um episódio forte cujo início está a >= escalada_idade_h do início do episódio que gerou o bloqueio atravessa os 72 h de refratário ('piorar é notícia'; o comentário do código diz que sem esse furo a régua de início cai de 6/8 para 4/8).

Uso:  PYTHONPATH=src .venv/bin/python experiments/31_mecanismo_do_e192.py

RESULTADO (16/10/2026).
  OUTUBRO/2025. O voto bruto tem uma corrida CONTÍNUA de 22/10 00:52 a 31/10 09:40 (225 h). Com 96 h: aos 4,7 dias (26/10 17:16) a força passa de 20 e a idade (112 h) é >= 96: o voto é CORTADO e o trecho forte (26/10 17:16 a 31/10 09:40, 112 h) vira um episódio = o FP 'de 112 h' do publicado. Com 192 h: a idade na escalada (112 h) é < 192, não há corte; a corrida inteira (225 h) começa 7 dias (167 h) depois do início do episódio de 15/10 (que bloqueia) e 167 h < 192 h: o furo do refratário NÃO a deixa passar; os 225 h de voto, com força > 20 por 4,7 dias, ficam em SILÊNCIO. Com 144 h a corrida inteira passa pelo furo (167 >= 144), com o mesmo número de FP e mais horas (520 h fora, EXP 21/26): por isso 144 'não muda' e 192 'remove': o ganho é um DEGRAU que depende de uma idade (167 h) cair entre 144 e 192.
  JANEIRO/2026. O voto bruto de 05/01 a 11/01 (149 h; força > 20 a partir de 09/01 18:42, idade 115 h): com 96 h é cortado e o trecho de 09/01 18:42 a 11/01 04:42 (34 h) vira o FP; com 192 h não há corte e o episódio fica sob o refratário.
  FEVEREIRO/2026 (o evento de 26/02): o voto de 670 h é cortado em 25/02 19:18 (força > 20) nas DUAS configurações: o onset do trip de 26/02 nasce dentro da janela com 96 e com 192 h.
  27/02/2025: com 96 h o início do evento NASCE dentro da janela (início, lead 48,0 h); com 192 h NÃO (início 0): o alarme que cobre o trip já existia, o reanúncio na janela (idade >= 96 h e < 192 h) deixa de ocorrer. É essa a diferença 'início 6 para 5' do EXP 30 (o E192 mantém 7/8 de detecção, mas perde o INÍCIO do 27/02/2025: a operação deixa de ter notícia nova nas 48 h antes do trip).
JUÍZO COMO ESPECIALISTA (a pergunta do usuário). NÃO É COERENTE: (1) o E192 não 'reduz FP': SILENCIA alarmes fortes (força > 20 vezes o limiar alto por 4,7 dias em out/2025) e DESLIGA o reanúncio dentro da janela de 8 dias, que é a função do mecanismo; (2) o valor 192 não tem justificativa física nem operacional (o 96 é meia semana de atenção; 192 é uma semana e um dia): ele foi escolhido por uma varredura de 29 células nos mesmos eventos e o resultado é um degrau cujo patamar depende de UMA idade (167 h) e de uma segunda (115 h); (3) o episódio silenciado de out/2025 termina numa parada de 77 h e o trip de 04/11 vem 4 dias depois (EXP 27b): se for aviso, o ajuste o apaga; (4) a contagem de 'FP' infla pelo CORTE em dois: o voto de 225 h de outubro é UM alarme contínuo, que o publicado conta como um episódio de 112 h depois do corte e o E192 conta como zero; o que o usuário vê como notificações é outra coisa.
CONSEQUÊNCIA. H-E192 (EXP 30) é RETIRADA como hipótese prospectiva: não se justifica por mecanismo. Se a questão for o excesso de notificações de alarmes longos, o caminho coerente é de apresentação (agrupar o reanúncio no mesmo alarme: gestão de alarme R1 com reanúncio, validada fora da amostra, ainda não implementada) e não de silenciar escaladas. Nenhuma alteração no pacote. O consenso (EXP 25-26) é outro mecanismo (remove o FP de 15/10/2025 e o de 09/01/2026, não o de 26/10) e segue com as ressalvas dos EXP 26 e 27.
"""
from __future__ import annotations
import contextlib, copy, io, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.controle_sinais import _ci, PKG
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    S = pd.read_parquet("data_local/sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")

    def roda(idade):
        d = copy.deepcopy(det); d["escalada_idade_h"] = idade
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.prever({"detector": d}, proc)
    R96, R192 = roda(96), roda(192); ini = pd.Timestamp("2025-02-01", tz="UTC")
    for nome, R in (("96", R96), ("192", R192)):
        r = avalia(R["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=ini)
        print(f"escalada_idade_h={nome}: {r.det}/8 detec | início {r.inicio} | FP {r.fp} | por evento (início, lead):", {k[:10]: (int(x['inicio']), None if x['lead_h'] is None else round(x['lead_h'], 1)) for k, x in r.por_evento.items()})
    v0 = ((R96["canais_nivel_a"] >= det["voto_nivel_a"]) | ((R96["canais_nivel_b"] >= det["voto_nivel_b"]) & (R96["b_sp"] | R96["b_vb"]))) & R96["vigiado"]
    runs = lambda s: episodios_de_serie(s.astype(bool))
    for rot, (a0, b0) in (("out/2025", ("2025-10-10", "2025-11-05")), ("jan/2026", ("2026-01-05", "2026-01-20")), ("fev/2026", ("2026-01-25", "2026-02-27"))):
        sl = slice(pd.Timestamp(a0, tz="UTC"), pd.Timestamp(b0, tz="UTC")); print(f"\n== {rot}: voto bruto (antes da escalada, do refratário e da duração) contra o alarme final")
        for nome, s in (("voto bruto", v0), ("final 96", R96["is_anomaly"]), ("final 192", R192["is_anomaly"])):
            print(f"   {nome:10s}:", [f"{a:%m-%d %H:%M}→{b:%m-%d %H:%M} ({(b - a).total_seconds() / 3600:.0f} h)" for a, b in runs(s.loc[sl])])
        f = R96["forca"].loc[sl]; print("   força > escalada_abs (20) em:", [f"{a:%m-%d %H:%M}→{b:%m-%d %H:%M}" for a, b in runs(f > det["escalada_abs"])][:6])


if __name__ == "__main__":
    main()
