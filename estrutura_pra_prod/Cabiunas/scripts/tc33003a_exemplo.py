#!/usr/bin/env python3
"""
SIMPred — Inferência de anomalias do TC-33003A (Cabiúnas).

Script FINO: aponta o bundle e os dados deste equipamento e executa os 4 passos,
todos implementados no módulo `cabiunas_inference.py`, nesta mesma pasta. Não
depende de nenhuma biblioteca interna nossa.

    python3 tc33003a_exemplo.py
    python3 tc33003a_exemplo.py --csv /caminho/para/dados.csv --dias 7

A janela de entrada é de 60 dias e só os últimos `--dias` são o resultado: o
resto é aquecimento, e não é opcional — ver o cabeçalho do módulo.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cabiunas_inference as ci

EQUIP = "TC-33003A"
EQUIP_DIR = Path(__file__).resolve().parent.parent      # .../Cabiunas
DADOS_DIR = EQUIP_DIR / "dados"
MODELOS_DIR = EQUIP_DIR / "modelos"
CSV_NOME = "data_tc33003a_raw.csv"
TRIPS = EQUIP_DIR / "registro_trips.csv"     # atualizar a cada trip novo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV de entrada (padrão: procura em dados/)")
    ap.add_argument("--modelos", default=None, help="pasta modelos/ (padrão: ../modelos)")
    ap.add_argument("--dias", type=int, default=7, help="dias finais a reportar")
    ap.add_argument("--ignorar-validade", action="store_true",
                    help="roda mesmo com bundle vencido (só para reprocessar histórico)")
    a = ap.parse_args()

    print(f"[{EQUIP}] 1/4 carregando dados...")
    csv = Path(a.csv) if a.csv else ci.achar_csv(DADOS_DIR, CSV_NOME)
    df = ci.carregar_dados(csv)
    print(f"        {len(df)} linhas  |  {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}"
          f"  ({(df.index[-1]-df.index[0]).days} d)")

    print(f"[{EQUIP}] 2/4 carregando modelo...")
    # TODOS os bundles, não só o mais recente: cada mês da janela de aquecimento
    # é pontuado pelo bundle que vigia nele. Ver `carregar_modelos` no módulo —
    # usar um bundle só aplica o PCA de hoje a dado de dois meses atrás, e o
    # resíduo explode por deriva do baseline, não por saúde da máquina.
    modelos = ci.carregar_modelos(Path(a.modelos) if a.modelos else MODELOS_DIR)
    corrente = modelos[-1]
    print(f"        {len(modelos)} bundle(s)  |  corrente: baseline até "
          f"{pd.Timestamp(corrente['modelo']['baseline_fim']):%Y-%m-%d}")
    erro = ci.checa_validade(corrente, df.index[-1])
    if erro:
        if not a.ignorar_validade:
            print(f"\nERRO: {erro}")
            return 2
        print(f"        AVISO (ignorado): {erro}")

    exigida = corrente["detector"].get("janela_entrada_dias", 60)
    tem = (df.index[-1] - df.index[0]).days
    if tem < exigida:
        print(f"        AVISO: {tem} d de entrada contra {exigida} d exigidos. "
              f"O alarme pode diferir — janela curta não dá erro, dá resultado diferente.")

    print(f"[{EQUIP}] 3/4 pre-processando...")
    # O registro de trips é do EQUIPAMENTO, não do bundle: muda quando a máquina
    # falha, não quando o modelo é retreinado. Ver `carregar_trips` no módulo.
    trips = ci.carregar_trips(TRIPS)
    if not trips:
        print(f"        AVISO: {TRIPS.name} vazio ou ausente. A referência do vb "
              f"não vai excluir degradação já conhecida, e o canal fica surdo "
              f"para a repetição dela.")
    else:
        print(f"        {len(trips)} trip(s) no registro, último "
              f"{max(trips):%Y-%m-%d}")
    proc = ci.preprocessar(modelos, df, trips=trips)
    n_vig = int(proc["mask"].sum())
    print(f"        {len(proc)} instantes de 2 min  |  {n_vig} vigiados "
          f"({100*n_vig/max(len(proc),1):.1f}%)")

    print(f"[{EQUIP}] 4/4 inferindo...")
    res = ci.prever(modelos, proc)

    # ── Só os últimos --dias são resultado; o resto foi aquecimento ──
    corte = res.index[-1] - pd.Timedelta(days=a.dias)
    saida = res.loc[res.index >= corte]
    eps = ci.resumo_episodios(saida)

    n_al = int((saida["severity"] == "alarme").sum())
    n_at = int((saida["severity"] == "atencao").sum())
    print(f"\n{EQUIP}: últimos {a.dias} d  |  {len(saida)} instantes  |  "
          f"{n_at} atenção  |  {n_al} alarme")
    if len(eps):
        print(f"\n{len(eps)} episódio(s) de alarme:")
        for _, e in eps.iterrows():
            print(f"    {e.inicio:%Y-%m-%d %H:%M} .. {e.fim:%Y-%m-%d %H:%M}  "
                  f"{e.horas:6.1f} h   força {e.forca_max:5.1f}x   "
                  f"{e.canais_max} canais")
    else:
        print("    nenhum alarme na janela reportada")

    nome = EQUIP.lower().replace("-", "") + "_inferencia.csv"
    destino = Path(__file__).parent / nome
    saida.to_csv(destino)
    print(f"\n        resultado salvo em: {destino.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
