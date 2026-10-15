#!/usr/bin/env python3
"""EXP 01b — ATLAS, PARTE 1b: A CORRIDA TERMINAL (correção do estatístico de horizonte do EXP 01).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. Exploratório.

POR QUE EXISTE. O EXP 01 definiu o "lead" como o início da PRIMEIRA corrida sustentada (|z| >= 3 por >= 3 h seguidas) na janela de 7 d. Os gráficos mostraram que isso confunde precursor com TRANSIENTE DE COMEÇO DE JANELA: em 09/12/2025 o "lead de 163 h" de TI_0301/0317/0303
é o aquecimento depois de uma partida no começo da janela (z cai a ~0 em seguida), e em 04/11/2025 o TI_0305 já está ~6 sigma acima da referência quando a janela começa (antes de uma parada de 2 dias e de uma partida a frio). Um precursor é um desvio que CONTINUA ATÉ O FIM: o estatístico certo é a CORRIDA TERMINAL.
Também: a coluna "queda T5" do Q1 (720 min = já estava abaixo de 0,95 x a mediana no começo da janela de 12 h) mede nível, não queda; fica descartada.

O QUE SE FAZ. Mesma referência causal, mesmos 450 instantes normais (mesma semente), mesma janela [E - 7 d, E - 2 h]. Por tag: `term_h` = nº de horas válidas seguidas com |z horário| >= 3 terminando na ÚLTIMA hora da janela (0 se a última hora não é vigiada ou |z| < 3).
Percentil do evento = fração das janelas normais com `term_h` menor. Relatam-se, por evento, as tags com term_h >= 6 h e o percentil; por família (vibração, termopares, processo, mancais TI_03xx), o maior term_h; e a taxa, nas janelas normais, de terminar com corrida >= 6, 12 e 24 h (a especificidade).

EXPECTATIVAS REGISTRADAS: as corridas terminais são mais curtas que os "leads" do EXP 01 (mediana da tag de maior term_h <= 40 h); em >= 4 dos 8 eventos alguma tag tem term_h >= 6 h com percentil >= 0,95; 04/29 e 02/26/2026 não têm nenhuma;
os sensores de temperatura de mancal (TI_03xx) têm taxa de corrida terminal >= 12 h nas janelas normais menor que 5% (alta especificidade) e a vibração maior que 20% (baixa especificidade).

RESULTADO (15/10/2026) -- A 1.ª EXECUÇÃO FOI INVÁLIDA (bug da hora final incompleta, que zerava a corrida terminal só nas janelas normais; ver a emenda ao fim de `01_anatomia_trips.py`) E FOI REPETIDA. Valem estes números.
  ESPECIFICIDADE (fração das 450 janelas normais que TERMINAM com corrida >= 6 / 12 / 24 h em alguma tag da família): vibração (10 tags) 48,0 / 45,3 / 37,8%; processo (18) 32,0 / 22,2 / 16,0%; termopares de exaustão (6) 5,1 / 4,0 / 2,7%; mancais TI_0301/0303/0305/0307 (4) 1,3 / 0,9 / 0,7%.
  CORRIDA TERMINAL POR EVENTO (maior term_h; tags com term_h >= 6 h e percentil >= 0,95): 27/02: 148 h em TV_354X/Y (0,96) e 14 h em PDIT_0305 (0,99) | 17/03: nenhuma (processo 5 h) | 07/04: 79 h em PDI_0317, TV_355X/Y (0,98 a 0,99), 59 h em TC382_04 (1,00), 9 h em TC382_02/03, mancais 8 h | 11/04: 72 h em PDI_0317, TV_355X/Y, TV_351X/Y, termopares 36 h, mancais 3 h (aglomerado com 07/04) |
        29/04: 29 h em TV_353Y (0,96) e 11 h em TV_353X | 04/11: vibração 13 h (0,89 a 0,93; nenhuma >= 0,95) | 09/12: 166 h em TV_353X (1,00) e 95 h em TV_353Y (0,99), 25 h em TV_351Y | 26/02/2026: vibração 29 h (0,85 a 0,86), nenhuma >= 0,95.
  Maior term_h por evento: 148, 5, 79, 72, 29, 13, 166, 29 h (mediana 50,5 h). Eventos com alguma tag a term_h >= 6 h e percentil >= 0,95: 5 de 8 (27/02, 07/04, 11/04, 29/04, 09/12).
LEITURA. (1) A VIBRAÇÃO é a família de maior sensibilidade (term_h >= 24 h em 6 de 8 eventos) e de MENOR especificidade (37,8% das janelas normais terminam com corrida >= 24 h): é a fonte natural de falsos positivos e o que o `vb` do v2 explora. (2) Os TERMOPARES DE EXAUSTÃO e os MANCAIS TI_03 são muito específicos (<= 4% e <= 1,3% no normal), mas pouco sensíveis:
  termopares só em 07/04 (59 h) e 11/04 (36 h) (o mesmo aglomerado), mancais em 07/04 (8 h) e 11/04 (3 h). (3) 07/04 e 11/04 têm um precursor MULTIFAMÍLIA de 72 a 80 h (vibração + processo + termopares); 27/02 e 09/12 têm precursor SÓ DE VIBRAÇÃO de 148 e 166 h (z 8 a 15); 17/03, 04/11 (janela de 53% vigiada) e 26/02/2026 não têm corrida terminal distintiva.
  (4) HIPÓTESES geradas (a pré-registrar antes de testar): H1 uma CONJUNÇÃO de famílias (vibração E termopares/mancais) separa 07/04 e 11/04 do normal com especificidade ~95%, ao custo de sensibilidade 2 de 8; H2 a corrida terminal de vibração >= 72 h (não >= 6 h) separa 27/02, 07/04, 11/04 e 09/12 do normal (taxa nula a medir); H3 os eventos sem precursor nas tags (17/03, 04/11, 26/02) concentram-se em torno de partida/parada (a partida a frio de 04/11; 17/03 e 26/02 sem corrida terminal).
PLACAR: ACERTEI: >= 4 eventos com tag a term_h >= 6 h e percentil >= 0,95 (5); os mancais TI_03 têm taxa de corrida terminal >= 12 h no normal < 5% (0,9%) e a vibração > 20% (45,3%). ERREI: a mediana do maior term_h <= 40 h (deu 50,5 h); "29/04 e 26/02/2026 sem nenhuma": 29/04 tem TV_353Y a 29 h e percentil 0,96.

Uso:  PYTHONPATH=src .venv/bin/python experiments/01b_corrida_terminal.py
"""
from __future__ import annotations
import importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("exp01", Path(__file__).with_name("01_anatomia_trips.py")); E1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E1)
from nv import io as nvio
from nv.causal import carrega_registro

LIM = 3.0


def term(N, E):
    r = N.zhoras(E)
    if r is None:
        return None
    zh, frac = r
    if frac < 0.20 or zh.notna().any(axis=1).sum() < 24:
        return dict(aplicavel=False, frac=frac)
    a = zh.abs().to_numpy(); out = {}
    for j, t in enumerate(N.tags):
        run = 0
        for k in range(len(a) - 1, -1, -1):
            v = a[k, j]
            if np.isfinite(v) and v >= LIM:
                run += 1
            else:
                break
        out[t] = run
    return dict(aplicavel=True, frac=frac, term=out)


def main():
    g = nvio.carrega_grade(); N = E1.Normal(g); reg = carrega_registro(); ev = [t for t in reg if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    nulos = []
    for E, _ in E1.sorteia_nulos(N, reg):
        s = term(N, E)
        if s and s.get("aplicavel"):
            nulos.append(s["term"])
    print(f"{len(nulos)} janelas normais", flush=True)
    tags = N.tags; fam = nvio.grupos(tags)
    fam["mancal_TI03"] = [t for t in tags if any(t.endswith(x) for x in ("TI_0301", "TI_0303", "TI_0305", "TI_0307"))]
    nt = {t: np.array([x[t] for x in nulos]) for t in tags}
    print("\nESPECIFICIDADE: fração das janelas normais que TERMINAM com corrida >= 6 / 12 / 24 h, por família (alguma tag da família)")
    for nome in ("vibracao", "exaustao_tc", "processo", "mancal_TI03"):
        cols = [t for t in fam[nome] if t in nt]
        m = np.max(np.vstack([nt[t] for t in cols]), axis=0)
        print(f"   {nome:12s} ({len(cols):2d} tags): >=6 h {100 * np.mean(m >= 6):5.1f}% | >=12 h {100 * np.mean(m >= 12):5.1f}% | >=24 h {100 * np.mean(m >= 24):5.1f}%")
    print("\nCORRIDA TERMINAL EM CADA TRIP (term_h e percentil entre as janelas normais)")
    topos = []
    for T in ev:
        s = term(N, T)
        if not s or not s.get("aplicavel"):
            print(f"== {T:%Y-%m-%d}: vigilância {100 * (s or {}).get('frac', 0):.0f}%: NÃO APLICÁVEL"); topos.append(np.nan); continue
        tt = s["term"]; pc = {t: float(np.mean(nt[t] < tt[t])) for t in tags}
        com = sorted([t for t in tags if tt[t] >= 6], key=lambda t: -tt[t])
        print(f"== {T:%Y-%m-%d}: tags com term_h >= 6 h: {len(com)} | com percentil >= 0,95: {sum(pc[t] >= 0.95 for t in com)}")
        for t in com[:6]:
            print(f"     {t.replace('954005_624_', ''):12s} term_h {tt[t]:4d} | percentil {pc[t]:.2f}")
        topos.append(max(tt.values()))
        for nome in ("vibracao", "exaustao_tc", "processo", "mancal_TI03"):
            cols = [t for t in fam[nome] if t in tt]
            print(f"     família {nome:12s}: maior term_h {max(tt[t] for t in cols):3d} h")
    print(f"\nmaior term_h por evento: {topos} | mediana {np.nanmedian(topos):.1f} h")


if __name__ == "__main__":
    main()
