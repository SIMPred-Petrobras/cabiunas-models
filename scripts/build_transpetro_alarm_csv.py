"""Constrói o CSV de alarme sintético por equipamento Transpetro, no schema mínimo que
io.py::load_data espera: Data da Ocorrência / Tag Alarme / Condição do Alarme.

Fonte de verdade: 1 falha catastrófica documentada por equipamento
(Mapeamento Falhas Equipamentos Críticos.xlsx). Para B-4064A, soma-se o log semanal de
disponibilidade (Histórico falha B-4064A.xlsm): Indisponível->HIHI, Restrição->HI,
dando mais contexto à camada preditiva (o evento catastrófico isolado teria N=1).

Uso:
  python scripts/build_transpetro_alarm_csv.py
"""
from pathlib import Path
import pandas as pd

MAP_XLSX = Path("../dados/drive-download-20260720T190351Z-1-001/Mapeamento Falhas Equipamentos Críticos.xlsx")
FALHA_B4064A_XLSM = Path("../dados/drive-download-20260720T190356Z-1-001/Histórico falha B-4064A.xlsm")
OUT_DIR = Path("../dados/transpetro")

# colunas de sensor por equipamento (nomes exatos das colunas nos CSVs convertidos)
SENSOR_COLS = {
    "B-4064A": ["Pressão Sucção", "Pressão Descarga", "Corrente", "Vibração Bomba LNA",
               "Temperatura Bomba LA", "Temperatura Bomba LNA", "Temperatura Motor LA",
               "Temperatura Motor LNA", "Densidade"],
    "B-90001A": ["Pressão Descarga", "Pressão Sucção", "Vibração Motor LNA Y",
                "Vibração Motor LA X", "Vibração Motor LA Y", "Vibração Bomba LA X",
                "Vibração Bomba LA Y", "Vibração Bomba LNA X", "Vibração Bomba LNA Y"],
    "B-8802B": ["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA",
               "Vibração Bomba LNA", "Temperatura Bomba LA", "Temperatura Bomba LNA",
               "Temperatura Motor LA", "Temperatura Motor LNA"],
    "B-402E": ["Pressão Sucção", "Pressão Descarga", "Corrente", "Vazão",
              "Vibração Bomba LA", "Temperatura Estator U", "Temperatura Estator V",
              "Temperatura Estator Wa", "Temperatura Estator Wb",
              "Temperatura Mancal LA Motor", "Temperatura Mancal LNA Motor",
              "Temperatura Mancal Ext. Escora LNA Bomba",
              "Temperatura Mancal Int. Escora LNA Bomba",
              "Temperatura Mancal Radial LA Bomba", "Temperatura Mancal Radial LNA Bomba"],
}

DETECCAO = {
    "B-4064A": pd.Timestamp("2024-08-30 07:58:00"),
    "B-90001A": pd.Timestamp("2021-08-28 00:00:00"),
    "B-8802B": pd.Timestamp("2022-07-06 10:00:00"),
    "B-402E": pd.Timestamp("2019-10-30 11:06:00"),
}


def rows_for_event(equip: str, ts: pd.Timestamp, cond: str) -> list[dict]:
    return [{"Data da Ocorrência": ts, "Tag Alarme": col, "Condição do Alarme": cond}
            for col in SENSOR_COLS[equip]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = rows_for_event("B-4064A", DETECCAO["B-4064A"], "HIHI")
    disp = pd.read_excel(FALHA_B4064A_XLSM, sheet_name="Export")
    disp = disp.dropna(subset=["Data"])
    cond_map = {"Indisponível": "HIHI", "Restrição": "HI"}
    for _, r in disp.iterrows():
        c = cond_map.get(r["Disponibilidade"])
        if c:
            rows += rows_for_event("B-4064A", pd.Timestamp(r["Data"]), c)
    df_b4064a = pd.DataFrame(rows).sort_values("Data da Ocorrência")
    out = OUT_DIR / "B-4064A_alarme.csv"
    df_b4064a.to_csv(out, index=False)
    print(f"B-4064A: {len(df_b4064a)} linhas -> {out} "
          f"(condições: {df_b4064a['Condição do Alarme'].value_counts().to_dict()})")

    for equip in ["B-90001A", "B-8802B", "B-402E"]:
        rows_e = rows_for_event(equip, DETECCAO[equip], "HIHI")
        df_e = pd.DataFrame(rows_e).sort_values("Data da Ocorrência")
        out_e = OUT_DIR / f"{equip}_alarme.csv"
        df_e.to_csv(out_e, index=False)
        print(f"{equip}: {len(df_e)} linhas -> {out_e} (N=1 evento, sem log semanal)")


if __name__ == "__main__":
    main()
