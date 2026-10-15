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
