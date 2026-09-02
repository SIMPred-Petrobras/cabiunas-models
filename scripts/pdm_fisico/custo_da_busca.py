#!/usr/bin/env python3
"""Quanto custa procurar? LOEO aninhado em funcao do TAMANHO DO ESPACO DE BUSCA.

A pergunta que decide se vale montar AutoML aqui. Busca ampla compra ajuste dentro da
amostra pagando com generalizacao; com 8 eventos e um piso de ruido de retreino de 20,7
pp, a duvida e se sobra alguma coisa. Tres pontos soltos ja sugeriam a forma da curva
(1 config -> LOEO 8/9; ~24 -> 7/9; 200 -> 6/9), mas eram grades diferentes e nao
comparaveis. Aqui a medida e feita direito.

DESENHO. Espaco completo de 1.512 configuracoes da CAMADA DE DECISAO (k_base x k_vib x
refratario x duracao minima x teto). Para cada tamanho S, sorteia R subconjuntos
aleatorios de tamanho S e roda LOEO ANINHADO em cada um: para cada evento retirado, a
busca escolhe dentro do subconjunto usando SO os outros 7, sob orcamento de custo, e o
evento retirado e usado apenas para testar.

A curva esperada e um U INVERTIDO, e as duas pontas significam coisas diferentes:
  - S pequeno: a busca nao acha boa configuracao (subajuste da SELECAO, nao do modelo);
  - S grande: acha configuracao que so vale naquela dobra (sobreajuste da selecao).
O pico diz quanto espaco de busca este problema comporta. Se a curva for plana a direita,
AutoML e seguro aqui. Se cair, o numero diz o preco.

Custo O(1) por avaliacao: cada configuracao e reduzida a (lista de episodios, deteccao
por evento) UMA vez; o LOEO depois e so contabilidade.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from auto_reset import trunca
import reduz_fp as RF

T0 = pd.Timestamp("2024-02-01 00:00", tz="UTC")     # janela valida (janela_valida.py)
JAN = pd.Timedelta(hours=48)
KB = [1.2, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0]
KV = [1.6, 2.2, 2.8, 3.5]
RS = [0, 12, 24, 48, 72, 120]
DS = [0, 60, 120]
TE = [0, 12, 24]
TAMANHOS = [1, 3, 10, 30, 100, 300, 1000, 1512]
N_SORTEIOS = 40
ORCAMENTO = 2.5          # FP/mes, o do ponto de operacao
RNG = np.random.default_rng(7)


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df) & (idx >= T0)
    alvo = list(todas[todas >= T0])
    meses = mask.sum() * 2 / 60 / 730.0
    out = roda(BRACO, df, todas)
    print(f"alvo: {len(alvo)} eventos | {mask.sum()*2/60:.0f} h pontuaveis "
          f"({meses:.1f} meses) | espaco: {len(KB)*len(KV)*len(RS)*len(DS)*len(TE)} configs\n",
          flush=True)

    # --- reduz cada configuracao a (episodios, deteccao por evento)
    EPS, DET_, chaves = [], [], []
    for kb in KB:
        for kv in KV:
            base = alerta_2k(out, mask, kb, kv) & mask
            for Rh in RS:
                aR = RF.refratario(base, Rh) if Rh else base
                for D in DS:
                    aD = RF.dur_min(aR, D) if D else aR
                    for te in TE:
                        al = trunca(aD, te) if te else aD
                        eps = A.episodios(al)
                        EPS.append(eps)
                        DET_.append(np.array([any(a <= t and b >= t - JAN for a, b in eps)
                                              for t in alvo]))
                        chaves.append((kb, kv, Rh, D, te))
        print(f"  k_base={kb} reduzido ({len(EPS)} configs)", flush=True)
    DET_ = np.array(DET_)
    NEP = np.array([len(e) for e in EPS])
    print(f"\nconfiguracoes reduzidas: {len(EPS)}", flush=True)

    # --- FP de cada config para cada subconjunto de eventos (basta variar o retirado)
    def fp_sem(i_out):
        """FP/mes de cada config quando o evento i_out sai do alvo."""
        janelas = [(t - JAN, t) for j, t in enumerate(alvo) if j != i_out]
        v = np.empty(len(EPS))
        for i, eps in enumerate(EPS):
            n = sum(1 for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in janelas))
            v[i] = n / meses
        return v

    FP = np.array([fp_sem(i) for i in range(len(alvo))])
    print("matriz de custo por dobra pronta\n", flush=True)

    def loeo(sub):
        """LOEO aninhado dentro do subconjunto `sub` de indices de configuracao."""
        ac = 0
        for i in range(len(alvo)):
            outros = [j for j in range(len(alvo)) if j != i]
            det_tr = DET_[np.ix_(sub, outros)].sum(axis=1)
            fp = FP[i][sub]
            ok = fp <= ORCAMENTO
            if not ok.any():
                continue
            # melhor deteccao no treino; empate -> menor custo
            pont = np.where(ok, det_tr * 1000 - fp, -np.inf)
            best = sub[int(np.argmax(pont))]
            ac += bool(DET_[best, i])
        return ac

    print("=" * 92)
    print(f"LOEO ANINHADO x TAMANHO DO ESPACO DE BUSCA (orcamento <= {ORCAMENTO} FP/mes)")
    print("=" * 92)
    print(f"{'S':>6} {'sorteios':>9} {'LOEO medio':>11} {'dp':>6} {'min':>5} {'max':>5}   "
          f"distribuicao")
    linhas = []
    todos = np.arange(len(EPS))
    for S in TAMANHOS:
        n = 1 if S >= len(EPS) else N_SORTEIOS
        vals = []
        for _ in range(n):
            sub = todos if S >= len(EPS) else RNG.choice(todos, S, replace=False)
            vals.append(loeo(np.sort(sub)))
        v = np.array(vals, float)
        hist = "".join(str(int((v == k).sum())) if (v == k).sum() < 10 else "+"
                       for k in range(len(alvo) + 1))
        print(f"{S:6d} {n:9d} {v.mean():10.2f}/{len(alvo)} {v.std():6.2f} "
              f"{int(v.min()):5d} {int(v.max()):5d}   {hist}  (contagem por LOEO=0..{len(alvo)})",
              flush=True)
        linhas.append(dict(S=S, n=n, media=v.mean(), dp=v.std(), mn=v.min(), mx=v.max()))
    pd.DataFrame(linhas).to_csv("custo_da_busca.csv", index=False)

    print("\n" + "=" * 92)
    print("referencia: o ponto recomendado, sem busca nenhuma")
    print("=" * 92)
    i = chaves.index((1.7, 2.2, 48, 60, 0))
    print(f"  k=1,7 k_vib=2,2 R=48h D=60min sem teto: {int(DET_[i].sum())}/{len(alvo)} "
          f"na serie toda, {NEP[i]} episodios, {FP[0][i]:.2f} FP/mes")
    print(f"  LOEO de um espaco com ele sozinho: {loeo(np.array([i]))}/{len(alvo)}")


if __name__ == "__main__":
    main()
