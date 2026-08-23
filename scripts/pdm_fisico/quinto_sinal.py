#!/usr/bin/env python3
"""A representacao multi-escala como 5o SINAL DO VOTO, nao como alarme somado.

De onde vem. ensemble.py mostrou que o detector de 4 sinais e o stack de features do
EXP7 erram eventos DIFERENTES (nos perdemos 2024-01-16, ele perde 2025-03-17), e que a
uniao dos dois alarmes da 9/9 com LOEO 9/9. Mas a uniao e uma forma ruim de aproveitar
isso: leva os episodios de 88 para 334 e piora o p de permutacao de 0,0003 para 0,0136,
porque somar alarmes soma cobertura, e cobertura tambem acerta por acaso.

A causa do preco esta no formato do alerta do stack, nao na informacao dele: com
sustentacao de 2 min o alerta fica ativo de 2 a 34 MINUTOS dentro da janela de 48 h de
cada evento. Sao piscadas. Subir a sustentacao para 10 min derruba a deteccao para 4/9,
entao o stack sozinho nao consegue ser ao mesmo tempo sensivel e persistente.

A hipotese deste script e que o voto resolve isso. No detector daqui, um sinal que
pisca sozinho nao alarma: e preciso DOIS sinais simultaneos sustentados por 30 min. Se
o escore do stack entrar como quinto sinal, a piscada dele so vira alarme quando
coincidir com outro sinal ja elevado -- o que deveria manter a deteccao e nao criar
episodio novo, ao contrario da uniao.

Construcao, igual a dos outros quatro sinais: EWMA de 30 min sobre o escore, limiar,
sustentacao de 30 min, voto >= 2 sobre cinco sinais. O limiar do quinto entra na escala
dos outros como z robusto contra referencia rolante de 400 h com banda de guarda -- a
mesma maquinaria de rolante.py, para nao introduzir uma escala nova sem referencia.

CUIDADO com o vazamento ja medido: z_rolante recebe a lista de falhas e apaga +-7 dias
em torno de cada uma da serie que vira referencia. Correto em producao, vazamento num
LOEO. Aqui o LOEO recalcula o z por fold com apenas os 8 eventos visiveis, como em
crest_ceticismo.py.

Comparacoes obrigatorias, todas a custo igualado:
  - contra o detector de 4 sinais no ponto de operacao;
  - contra a UNIAO (o mesmo ganho de deteccao, com o preco em episodios);
  - contra dessensibilizar os 4 sinais ate gastar as mesmas horas -- se isso tambem
    fechar 9/9, o quinto sinal nao acrescenta nada e e so sensibilidade.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from textura import alerta_n
from auto_reset import trunca
from diego_stack import alerta_de
from diego_stack_valida import seleciona

TAG = "diego_iforest_estatico"      # 8/9, LOEO 8/9, 100% de cobertura de escore
K_Q = [1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5]


def z_do_escore(s: pd.Series, stable: pd.Series, falhas: pd.Series) -> pd.Series:
    """Poe o escore do iforest na mesma escala dos outros quatro sinais."""
    F = s.where(stable).to_frame("escore")
    return RO.z_rolante(F, stable, falhas, horas_base=400, guarda_h=24, phi=0.0)["escore"]


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    stable = df["stable"].astype(bool)
    mask = mascara_pontuacao(df)
    tr = pd.Series(idx < CORTE, index=idx)
    ev_tr = falhas[falhas < CORTE]

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    nosso = alerta_2k(out, mask, K_BASE, K_VIB)

    s = pd.read_parquet(f"escore_{TAG}.parquet")["escore"].reindex(idx)
    mj = mask & s.notna()
    (_, _), p_st, sm_st, lim_st = seleciona(s, mj, tr, ev_tr, mj & tr)
    stack = alerta_de(s, mj, lim_st, sm_st)
    uniao = (nosso.fillna(False) | stack.fillna(False)) & mask

    print("calculando o z rolante do escore ...", flush=True)
    Z = z_do_escore(s, stable, falhas)
    q = Z[mask & tr]
    print(f"  z do escore no treino: p50={q.median():.2f} p95={q.quantile(.95):.2f} "
          f"p99={q.quantile(.99):.2f} max={q.max():.1f}\n", flush=True)

    def m(al):
        x = A.avalia(al, falhas, mask)
        x.update(A.permuta(al, mask, x["det"], len(falhas)))
        x["perdidos"] = sorted(set(t.strftime("%Y-%m-%d") for t in falhas) - set(x["detectados"]))
        return x

    linhas = []
    def mostra(nome, al):
        x = m(al)
        print(f"{nome:38s} {x['det']:>2}/9 {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f} {x['p']:8.4f}  "
              f"{','.join(x['perdidos']) or '-'}")
        linhas.append(dict(config=nome, det=x["det"], eps=x["episodios"], fp_mes=x["fp_mes"],
                           h_mes=x["h_fp_mes"], lead=x["lead_med"], p=x["p"],
                           perdidos=",".join(x["perdidos"])))
        return x

    print(f"{'configuracao':38s} {'det':>4} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} "
          f"{'lead':>6} {'p':>8}  perdidos")
    mostra("4 sinais (referencia)", nosso)
    mostra("4 sinais + teto 12h", trunca(nosso, 12))
    mostra("stack sozinho", stack)
    mostra("UNIAO de alarmes", uniao)
    print()
    for k in K_Q:
        mostra(f"5 sinais, k_q={k}", alerta_n(out, mask, [(Z, "30min", 3.0)], K_BASE, K_VIB, k))
    print()
    for k in K_Q:
        mostra(f"5 sinais + teto 12h, k_q={k}",
               trunca(alerta_n(out, mask, [(Z, "30min", 3.0)], K_BASE, K_VIB, k), 12))

    print("\n=== CONTROLE: dessensibilizar os 4 sinais ate gastar as mesmas horas ===")
    for kb in [1.0, 1.2, 1.4, 1.5, 1.6, 1.7]:
        for kv in [1.2, 1.6, 2.2]:
            x = m(alerta_2k(out, mask, kb, kv))
            if x["det"] == 9:
                print(f"  k_base={kb} k_vib={kv} FECHA 9/9 a {x['h_fp_mes']:.1f} h/mes")
    print("  (nenhuma linha acima = os 4 sinais nao chegam a 9/9 em nenhuma sensibilidade)")

    pd.DataFrame(linhas).to_csv("quinto_sinal.csv", index=False)

    print("\n=== LOEO do 5o sinal, com o z recalculado sem o evento escondido ===")
    print(f"{'evento fora':>12} {'k*':>5} {'8 restantes':>12} {'h/mes':>7} | {'detectado?':>11} {'lead':>6}")
    pega = []
    for ev in falhas:
        outros = falhas[falhas != ev]
        Zf = z_do_escore(s, stable, outros)
        melhor = None
        for k in K_Q:
            al = alerta_n(out, mask, [(Zf, "30min", 3.0)], K_BASE, K_VIB, k)
            x = A.avalia(al, outros, mask)
            chave = (x["det"], -x["h_fp_mes"])
            if melhor is None or chave > melhor[0]:
                melhor = (chave, k, x, al)
        (_, _), k, x8, al = melhor
        xe = A.avalia(al, pd.Series([ev]), mask)
        ok = xe["det"] == 1
        pega.append(ok)
        print(f"{ev.strftime('%Y-%m-%d'):>12} {k:5.1f} {x8['det']:>8}/8   {x8['h_fp_mes']:7.1f} | "
              f"{'SIM' if ok else 'nao':>11} {xe['lead_med'] if ok else float('nan'):6.1f}")
    print(f"\nLOEO do 5o sinal: {sum(pega)}/9    (4 sinais: 7/9 | uniao de alarmes: 9/9)")


main()
