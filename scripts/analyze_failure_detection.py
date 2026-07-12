#!/usr/bin/env python3
"""Análise comparativa de detecção de falha: modelo por-sensor (univariado)
vs multivariado (multisensor).

Para cada equipamento e cada conjunto, mede se as anomalias de PONTO caíram
perto da(s) data(s) de falha documentada(s):
  - anom_48h    : pontos anômalos na janela ±2 dias da falha (janela de avaliação)
  - anom_pre10  : pontos anômalos nos 10 dias ANTES da falha (precursor)
  - lead_time   : dias entre o 1º ponto anômalo no pré-10d e a falha
  - rate_day    : taxa global de anomalias por dia (contexto de ruído)

Classificação por equipamento:
  BOM     -> detectou na janela ±48h da falha
  PARCIAL -> só precursor nos 10 dias antes (fora de ±48h)
  FRACO   -> nenhuma anomalia em ±10 dias da falha

Multivariado é baixado do ClearML (só os artefatos leves) e fica em
resultados/raw_multivar_v2/. Por-sensor usa os CSVs já coletados em
resultados/Uni_sensor/ (dados pesados ficam fora de analysis/, que é só
relatórios/scripts).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.cnn1d_ae.pipeline import parse_failure_dates

PERSENSOR_DIR = Path("resultados/Uni_sensor")
MULTIVAR_DIR = Path("resultados/raw_multivar_v2")
OUT_MD = Path("analysis/COMPARACAO_persensor_vs_multivar.md")

MULTIVAR_TASKS = {
    "B-0302C": "2c5e3091a9354b639d7dc8944d9729a3",
    "B-24001B": "5642c558c9f44fd1b81dcafdbc1386e3",
    "B-3403C": "a8f4fe6110c446b39300f2203d10df79",
    "B-402E": "1a801a18ede944a18d4fea53356f8286",
    "B-4064A": "62aab58491c04d93a95af28f291a4417",
    "B-4703.24001B": "1b19ab0104f34759919648431e5d1629",
    "B-5401A": "c4ea20d303ae4aaa95ca5cdfce5e2a5f",
    "B-5501B": "d00bfa2a82124cfea5336f9fb8c837aa",
    "B-6511502A": "9e7b6c0ad3e74c008193d069e56dc4e3",
    "B-8801C": "08735c4f5da14dfcbef800893188c08c",
    "B-8802B": "59c0cc4561ee4de0bb72120cb9446c19",
    "B-90001A": "b64ce118703f4a2daa3d6076906b9cb1",
}

WANT_SUFFIXES = ("point_anomalies_all.csv", "calibration_report.json", "run_config.json")


def _ensure_clearml_config() -> None:
    if not os.getenv("CLEARML_CONFIG_FILE"):
        local = Path.cwd() / "clearml.conf"
        if local.is_file():
            os.environ["CLEARML_CONFIG_FILE"] = str(local)


def _get_local_copy_retry(art, attempts: int = 3):
    for _ in range(attempts):
        try:
            p = art.get_local_copy()
        except Exception:
            p = None
        if p:
            return p
    return None


def download_multivar(eq: str, tid: str) -> Path:
    """Baixa SOMENTE os artefatos do modelo do GRUPO (prefixo == nome do grupo).

    A task multivariada também sobe point_anomalies de modelos univariados
    extras (subproduto) — usar o arquivo errado infla os números. O grupo é o
    artefato cujo prefixo é o nome do grupo (== EQUIPMENT_ID por convenção)."""
    import shutil
    eq_dir = MULTIVAR_DIR / eq
    csv_dir = eq_dir / "csv"
    if all((csv_dir / s.split("/")[-1]).exists()
           for s in ("point_anomalies_all.csv", "calibration_report.json", "run_config.json")):
        return eq_dir

    from clearml import Task
    task = Task.get_task(task_id=tid)

    # Descobre o prefixo do grupo: os que têm calibration_report com chave "group".
    group_prefix = None
    prefixes = {n.rsplit("/csv/", 1)[0] for n in task.artifacts if "/csv/" in n}
    # Convenção: nome do grupo == EQUIPMENT_ID.
    if eq in prefixes:
        group_prefix = eq
    else:
        # Fallback: abre os calibration_report e escolhe o que tem "group".
        for pfx in prefixes:
            key = f"{pfx}/csv/calibration_report.json"
            if key in task.artifacts:
                local = _get_local_copy_retry(task.artifacts[key])
                if local and '"group"' in Path(local).read_text(encoding="utf-8"):
                    group_prefix = pfx
                    break
    if group_prefix is None:
        print(f"[WARN] {eq}: prefixo do grupo não identificado; prefixos={sorted(prefixes)}")
        return eq_dir

    csv_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("point_anomalies_all.csv", "calibration_report.json", "run_config.json"):
        name = f"{group_prefix}/csv/{suffix}"
        if name not in task.artifacts:
            print(f"[WARN] {eq}: artefato ausente {name}")
            continue
        local = _get_local_copy_retry(task.artifacts[name])
        if not local:
            print(f"[WARN] {eq}: download falhou {name}")
            continue
        shutil.copy2(local, csv_dir / suffix)
    print(f"  {eq}: grupo='{group_prefix}'")
    return eq_dir


def _find(eq_dir: Path, fname: str) -> Path | None:
    for p in eq_dir.rglob(fname):
        return p
    return None


def _load_points(eq_dir: Path) -> pd.Series | None:
    p = _find(eq_dir, "point_anomalies_all.csv")
    if not p:
        return None
    df = pd.read_csv(p)
    tcol = df.columns[0]
    idx = pd.to_datetime(df[tcol], errors="coerce")
    s = pd.Series(df["is_anom_point"].values, index=idx).dropna()
    return s[s.index.notna()]


def _failure_dates(eq_dir: Path) -> list:
    rc = _find(eq_dir, "run_config.json")
    if rc:
        cfg = json.loads(rc.read_text(encoding="utf-8"))
        return parse_failure_dates(cfg.get("FAILURE_DATE", ""))
    return []


def _calib(eq_dir: Path) -> dict:
    c = _find(eq_dir, "calibration_report.json")
    return json.loads(c.read_text(encoding="utf-8")) if c else {}


def analyze(eq: str, eq_dir: Path) -> dict:
    pts = _load_points(eq_dir)
    fails = _failure_dates(eq_dir)
    calib = _calib(eq_dir)
    res = {
        "equip": eq,
        "total_anom": int(pts.sum()) if pts is not None else None,
        "rate_day": calib.get("anomaly_rate_points_per_day"),
        "threshold": calib.get("threshold"),
        "n_fail": len(fails),
        "anom_48h": 0, "anom_pre10": 0, "lead_days": None,
        "classe": "SEM_DADOS",
    }
    if pts is None or not fails:
        return res

    anom_times = pts.index[pts.values == 1]
    best_lead = None
    for f in fails:
        w48 = anom_times[(anom_times >= f - pd.Timedelta(days=2)) & (anom_times <= f + pd.Timedelta(days=2))]
        pre = anom_times[(anom_times >= f - pd.Timedelta(days=10)) & (anom_times < f)]
        res["anom_48h"] += len(w48)
        res["anom_pre10"] += len(pre)
        if len(pre) > 0:
            lead = (f - pre.min()).total_seconds() / 86400.0
            best_lead = lead if best_lead is None else max(best_lead, lead)
    res["lead_days"] = round(best_lead, 1) if best_lead is not None else None

    if res["anom_48h"] > 0:
        res["classe"] = "BOM"
    elif res["anom_pre10"] > 0:
        res["classe"] = "PARCIAL"
    else:
        res["classe"] = "FRACO"
    return res


def _fmt(v):
    return "—" if v is None else v


def main() -> None:
    _ensure_clearml_config()
    equipes = list(MULTIVAR_TASKS.keys())

    print("[1/2] Baixando dados multivariados (leves)...")
    for eq, tid in MULTIVAR_TASKS.items():
        try:
            download_multivar(eq, tid)
            print(f"  ok {eq}")
        except Exception as e:
            print(f"  [WARN] {eq}: {e}")

    print("[2/2] Analisando ambos os conjuntos...")
    ps = {eq: analyze(eq, PERSENSOR_DIR / eq) for eq in equipes}
    mv = {eq: analyze(eq, MULTIVAR_DIR / eq) for eq in equipes}

    def table(title, data):
        lines = [f"### {title}", "",
                 "| Equip | Classe | Anom ±48h | Anom pré-10d | Lead (dias) | Anom/dia | Total anom |",
                 "|---|---|---|---|---|---|---|"]
        for eq in equipes:
            r = data[eq]
            lines.append(
                f"| `{eq}` | **{r['classe']}** | {_fmt(r['anom_48h'])} | {_fmt(r['anom_pre10'])} | "
                f"{_fmt(r['lead_days'])} | {_fmt(r['rate_day'])} | {_fmt(r['total_anom'])} |")
        return "\n".join(lines)

    def counts(data):
        c = {"BOM": 0, "PARCIAL": 0, "FRACO": 0, "SEM_DADOS": 0}
        for r in data.values():
            c[r["classe"]] = c.get(r["classe"], 0) + 1
        return c

    cps, cmv = counts(ps), counts(mv)

    md = [
        "# Comparação: modelo por-sensor (univariado) vs multivariado (multisensor)",
        "",
        "Métrica central: as **anomalias de ponto** caíram perto da falha documentada?",
        "- **BOM** = detectou na janela ±48h da falha · **PARCIAL** = só precursor nos 10 dias antes · **FRACO** = nada em ±10 dias.",
        "",
        f"**Placar por-sensor:** BOM={cps['BOM']} · PARCIAL={cps['PARCIAL']} · FRACO={cps['FRACO']}",
        "",
        f"**Placar multivariado:** BOM={cmv['BOM']} · PARCIAL={cmv['PARCIAL']} · FRACO={cmv['FRACO']}",
        "",
        table("Por-sensor (univariado)", ps),
        "",
        table("Multivariado (multisensor)", mv),
        "",
        "### Comparação lado a lado (classe)",
        "",
        "| Equip | Por-sensor | Multivariado | Vencedor |",
        "|---|---|---|---|",
    ]
    rank = {"BOM": 3, "PARCIAL": 2, "FRACO": 1, "SEM_DADOS": 0}
    for eq in equipes:
        a, b = ps[eq]["classe"], mv[eq]["classe"]
        win = "empate" if rank[a] == rank[b] else ("por-sensor" if rank[a] > rank[b] else "multivariado")
        md.append(f"| `{eq}` | {a} | {b} | {win} |")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\n[OK] Relatório salvo em {OUT_MD}")


if __name__ == "__main__":
    main()
