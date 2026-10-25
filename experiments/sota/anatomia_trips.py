"""Executor do EXP 66 (a definição e o pré-registro estão em `experiments/66_anatomia_dos_trips.py`)."""
from __future__ import annotations
import contextlib, io, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.avalia.metricas import JANELA, TMIN, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2

ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "anatomia_trips"
ALARMES = Path(os.environ.get("ALARMES_CSV", "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados/alarmes_mapeados_colunas.csv"))
EXTRAS = [pd.Timestamp("2025-02-27 17:33", tz="UTC"), pd.Timestamp("2025-03-18 11:16", tz="UTC")]
P = "954005_624_"; H6_TAREFA = "6e11bc034e0949ba8a39e1241c566357"
ANALOG = {"TI_0305": [P + "TI_0305"], "TI_0303": [P + "TI_0303"], "TI_0307": [P + "TI_0307"], "PDIT_0305": [P + "PDIT_0305"], "PI_0340": [P + "PI_0339", P + "PI_0340"],
          "PALL_6240309": [P + "PI_0339", P + "PI_0340"], "TI_0325": [P + "TI_0325"]}
LADO = {"TI_0305": 1, "TI_0303": 1, "TI_0307": 1, "PDIT_0305": 1, "PI_0340": -1, "PALL_6240309": -1, "TI_0325": -1}
MODO = {"TI_0305": "mancal radial LNA", "TI_0303": "mancal de escora inativo", "TI_0307": "mancal radial LA", "PDIT_0305": "selagem (vazamento do selo primário)",
        "PI_0340": "óleo lubrificante (pressão do header)", "PALL_6240309": "óleo lubrificante (pressão do compressor)", "TI_0325": "óleo lubrificante (temperatura do tanque)"}
loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M") if pd.notna(t) else "—"
curta = lambda tag: tag.replace(P, "")


def h6_top3():
    """No worker, o artefato da tarefa da H6; fora dele, o arquivo local (sem tocar no servidor, que pode estar inacessível)."""
    if os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task
        return pd.read_csv(Task.get_task(task_id=H6_TAREFA).artifacts["h6_top3"].get_local_copy())
    f = ROOT / "sota" / "h6_atribuicao" / "h6_top3.csv"
    return pd.read_csv(f) if f.exists() else None


def main():
    SAIDA.mkdir(parents=True, exist_ok=True); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    g = nvio.carrega_grade(); mk = nvio.mascaras(g); op = mk["ligada"]; est = mk["estavel"]; t5 = g["T5_AVG_A"].where(est)
    manut = ci._manutencoes(g); partidas = g.index[mk["partida"].to_numpy(bool)]
    d = pd.read_csv(ALARMES); d["t"] = pd.to_datetime(d["Data da Ocorrência"]).dt.tz_localize("UTC"); d = d[d["Status"] == "ACT/UNACK"].copy()
    d["tag"] = d["Tag Alarme"].map(curta); d["nivel_trip"] = d["Descrição Alarme"].astype(str).str.contains(r"Mt\.|TRIP", regex=True)
    registro = sorted(carrega_registro()); casos = [(t, "registro") for t in registro] + [(t, "fora do registro") for t in EXTRAS]
    LED = {v: carrega_controle(v) for v in ("atual", "G168", "B800")}
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        R = decide_v2(proc, det)
    H6 = h6_top3(); fichas, seqs = [], []
    for T, origem in sorted(casos):
        w = op.loc[T - pd.Timedelta(minutes=32):T + pd.Timedelta(minutes=32)]; quedas = w.index[1:][w.to_numpy()[:-1] & ~w.to_numpy()[1:]]
        Tp = min(quedas, key=lambda x: abs(x - T)) if len(quedas) else pd.NaT; ref = Tp if pd.notna(Tp) else T
        jan = d[(d.t >= ref - pd.Timedelta(minutes=15)) & (d.t <= ref + pd.Timedelta(minutes=2)) & d.nivel_trip].sort_values("t"); desloc = 0
        if jan.empty:
            larga = d[(d.t >= ref - pd.Timedelta(hours=6)) & (d.t <= ref + pd.Timedelta(hours=6)) & d.nivel_trip].sort_values("t")
            if not larga.empty:
                desloc = round((larga.t.iloc[0] - ref).total_seconds() / 3600, 2)
        fo = jan.iloc[0] if not jan.empty else None
        fo_tag = fo["tag"] if fo is not None else None
        rodando = bool(op.asof(fo["t"].floor("2min") - pd.Timedelta(minutes=2))) if fo is not None else None
        s = d[(d.t >= ref - pd.Timedelta(minutes=15)) & (d.t <= ref + pd.Timedelta(minutes=5))].sort_values("t")
        for t_, tag_, cond_, desc_, niv_ in s[["t", "tag", "Condição do Alarme", "Descrição Alarme", "nivel_trip"]].itertuples(index=False, name=None):
            seqs.append(dict(caso=loc(T), minuto=round((t_ - ref).total_seconds() / 60, 1), tag=tag_, condicao=cond_, descricao=desc_, nivel_trip=bool(niv_)))
        pre = d[(d.tag == fo_tag) & ~d.nivel_trip & (d.t >= ref - JANELA) & (d.t < ref)] if fo_tag else d.iloc[0:0]
        fic = dict(caso=loc(T), origem=origem, parada=loc(Tp), parada_menos_registro_min=round((Tp - T).total_seconds() / 60, 1) if pd.notna(Tp) else np.nan,
                   first_out=(f"{fo_tag} — {fo['Descrição Alarme']}" if fo is not None else "nenhum alarme de nível de trip"), first_out_min=round((fo["t"] - ref).total_seconds() / 60, 1) if fo is not None else np.nan,
                   first_out_com_maquina_operando=rodando, deslocamento_diario_h=desloc if jan.empty else 0.0, modo=MODO.get(fo_tag, "sem alarme de trip"),
                   pre_alarmes_48h=len(pre), primeiro_pre_alarme_h=round((ref - pre.t.min()).total_seconds() / 3600, 1) if len(pre) else np.nan)
        for tag in ANALOG.get(fo_tag, []):
            x = g[tag].where(op); base = x.loc[ref - JANELA - pd.Timedelta(days=30):ref - JANELA].where(est).dropna()
            med = float(base.median()) if len(base) else np.nan; mad = float(1.4826 * (base - med).abs().median()) if len(base) else np.nan
            v = lambda h: float(x.loc[:ref - pd.Timedelta(hours=h)].dropna().iloc[-1]) if x.loc[ref - JANELA:ref - pd.Timedelta(hours=h)].notna().any() else np.nan
            wv = x.loc[ref - JANELA:ref - pd.Timedelta(minutes=2)]; lado = LADO[fo_tag]
            fora = (lado * (wv - med) > 4 * max(mad, 1e-6)) | wv.isna()
            inicio = np.nan
            if fora.iloc[-1] if len(fora) else False:
                dentro = (~fora).to_numpy(); k = np.flatnonzero(dentro)
                ini = wv.index[k[-1] + 1] if len(k) else wv.index[0]
                inicio = round((ref - ini).total_seconds() / 3600, 1)
            nome = curta(tag)
            fic.update({f"{nome}_mediana30d": round(med, 3), f"{nome}_48h": round(v(48), 3), f"{nome}_6h": round(v(6), 3), f"{nome}_1h": round(v(1), 3), f"{nome}_10min": round(v(1 / 6), 3),
                        f"{nome}_extremo48h": round(float(wv.max() if lado > 0 else wv.min()), 3) if wv.notna().any() else np.nan, f"{nome}_excursao_h": inicio})
        rt5 = t5.loc[ref - pd.Timedelta(days=30) - JANELA:ref - JANELA].dropna(); wt5 = t5.loc[ref - pd.Timedelta(hours=2):ref].dropna()
        up = partidas[partidas < ref]; mm = [(a, b) for a, b in manut if ref - pd.Timedelta(days=7) <= b <= ref]
        volta = op.loc[ref:]; vt = volta.index[volta.to_numpy()]
        fic.update(carga_L=round(float((rt5 < wt5.median()).mean()), 2) if len(wt5) and len(rt5) > 30 else np.nan, T5_10min=round(float(g["T5_AVG_A"].asof(ref - pd.Timedelta(minutes=10))), 0),
                   h_desde_partida=round((ref - up[-1]).total_seconds() / 3600, 1) if len(up) else np.nan, manutencao_7d_antes=(f"{loc(mm[-1][0])} a {loc(mm[-1][1])}" if mm else "—"),
                   h_parada_depois=round((vt[0] - ref).total_seconds() / 3600, 1) if len(vt) else np.nan)
        for v, (al, vg) in LED.items():
            eps = [(a, b) for a, b in episodios_de_serie(al.loc[T - JANELA - pd.Timedelta(days=30):T]) if a <= T and b >= T - JANELA]
            if eps:
                a0 = eps[0][0]; fic[f"{v}_detecta"] = True; fic[f"{v}_lead_h"] = round((T - max(a0, T - JANELA)).total_seconds() / 3600, 1)
                fic[f"{v}_inicio"] = a0 >= T - JANELA; fic[f"{v}_banda"] = (a0 >= T - JANELA) and (a0 <= T - TMIN)
                if v == "atual":
                    fic["atual_nascimento"] = loc(a0); fic["atual_canais"] = "+".join(c for c in ("t", "p", "sp", "vb") if bool(R[f"a_{c}"].asof(a0)) or bool(R[f"b_{c}"].asof(a0)))
                    if H6 is not None:
                        h = H6[H6.inicio_local == loc(a0)]
                        fic["atual_top3"] = "; ".join(str(x) for x in h[["top1", "top2", "top3"]].iloc[0]) if len(h) else "—"
            else:
                fic[f"{v}_detecta"] = False
        fichas.append(fic)
    F = pd.DataFrame(fichas); S = pd.DataFrame(seqs)
    F.to_csv(SAIDA / "fichas_dos_trips.csv", index=False); S.to_csv(SAIDA / "sequencia_de_alarmes.csv", index=False)
    from nv.avalia import avalia
    alvo = [t for t in registro if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    v2 = {v: (avalia(al, vg, alvo, inicio_periodo=pd.Timestamp("2025-02-01", tz="UTC")).det, avalia(al, vg, alvo, inicio_periodo=pd.Timestamp("2025-02-01", tz="UTC")).fp) for v, (al, vg) in LED.items()}
    v1 = F[(F.origem == "registro") & (F.caso.str[6:10].isin(["2025", "2026"]))].first_out_min.notna().all()
    print(f"V1 — first-out na janela para todos os trips de 2025-2026: {'PASSA' if v1 else 'FALHA'} | V2 — ledgers (det, FP): {v2}", flush=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
    for f in fichas:
        print("\n" + "=" * 110 + f"\nCASO {f['caso']} (local) — {f['origem']} — MODO: {f['modo']}", flush=True)
        for k, v in f.items():
            if k not in ("caso", "origem", "modo"):
                print(f"   {k:34s} {v}", flush=True)
        sq = S[S.caso == f["caso"]]
        if len(sq):
            print("   sequência [parada − 15 min, + 5 min]:", flush=True)
            for r in sq.itertuples():
                print(f"      {r.minuto:+6.1f} min  {'TRIP ' if r.nivel_trip else '     '}{r.tag:22s} {r.condicao:5s} {r.descricao}", flush=True)
    print(f"\nRESUMO DOS MODOS: {F.modo.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
