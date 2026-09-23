#!/usr/bin/env python3
"""SIMPred / Cabiúnas — monta a pasta do TC-33003A pronta para subir ao Drive.

POR QUÊ EXISTE. A pasta de trabalho tem 1,8 GB, e só ~130 MB disso é
entregável: as outras três pastas de `dados/` são a matéria-prima em xlsx que
usamos para montar o CSV. Arrastar tudo infla a entrega catorze vezes e deixa
quem recebe sem saber qual arquivo é a entrada.

O QUE ELE FAZ. Copia para uma pasta nova, deixando a original intacta:

  · metadata.csv e registro_trips.csv
  · dados/<periodo>/ — só o CSV consolidado
  · modelos/ — os bundles, SEM os .pkl (ver abaixo)
  · documentos/ — os relatórios da entrega + a documentação de integração
  · scripts/ — o módulo, o exemplo, o retreino e a manutenção
  · README.md na raiz do equipamento

A cópia sai em `<destino>/Cabiunas/TC-33003A/`, com o nível do equipamento que a
convenção exige e a nossa pasta de trabalho não tem.

OS .pkl FICAM DE FORA. A recomendação do SIMPred é não entregar modelo em
código fechado, e pickle é fechado mesmo sendo sklearn puro: não se lê num
diff, exige a versão certa da biblioteca e executa código ao carregar. Os
`<familia>_transformacao.json` têm os mesmos números — conferido bit a bit — e
a inferência já lê deles. Os .pkl continuam no repositório; se a engenharia
pedir, use --com-pkl.

O QUE NÃO VAI. __pycache__, _pdf/ (fonte de diagramação), os xlsx brutos, e os
relatórios anteriores ao detector v2 — citar número velho numa pasta de entrega
é pior que não ter o documento.

    python3 monta_entrega_drive.py                    # <repo>/../_upload_drive/
    python3 monta_entrega_drive.py --destino /tmp/x   # outro lugar
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

EQUIP_DIR = Path(__file__).resolve().parent.parent          # .../Cabiunas
REPO = EQUIP_DIR.parents[1]                                 # .../cabiunas-models

# A convenção do SIMPred tem DOIS níveis: frente / equipamento. A nossa pasta de
# trabalho tem só um -- `Cabiunas/` faz o papel do equipamento -- e subir assim
# deixaria tudo um nível acima do que a Transpetro e a Constellation usam, sem
# lugar para o próximo turbocompressor de Cabiúnas. A cópia repõe o nível.
FRENTE = "Cabiunas"
EQUIPAMENTO = "TC-33003A"        # como está no metadata.csv e nos 16 bundles

# Relatórios que descrevem O QUE ESTÁ SENDO ENTREGUE. Os de agosto ficam fora:
# são anteriores ao gatilho de dois níveis e citam outro ponto de operação.
RELATORIOS = {
    "INTEGRACAO_DASHBOARD_TC33003A.pdf": "INTEGRACAO_DASHBOARD.pdf",
    "ENTREGA_TC33003A.pdf": "ENTREGA_TC33003A.pdf",
    "DETECTOR_V2.pdf": "DETECTOR_V2.pdf",
    "PIPELINE_TC33003A.pdf": "PIPELINE_TC33003A.pdf",
    "RELATORIO_PRODUCAO_TC33003A.pdf": "RELATORIO_PRODUCAO_TC33003A.pdf",
}

SCRIPTS = ("cabiunas_inference.py", "tc33003a_exemplo.py", "constroi_bundle.py",
           "migra_bundle_json.py", "monta_entrega_drive.py",
           "requirements.txt", "README.md")

BUNDLE_JSON = ("detector.json", "modelo.json", "normalizacao.json",
               "spread_mancal.json",
               "temperatura_transformacao.json", "pressao_transformacao.json")


def mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=None,
                    help="onde criar <frente>/<equipamento> (padrão: <repo>/../_upload_drive)")
    ap.add_argument("--com-pkl", action="store_true",
                    help="inclui os .pkl dos bundles (padrão: só os JSON)")
    ap.add_argument("--forcar", action="store_true", help="sobrescreve o destino")
    a = ap.parse_args()

    raiz = Path(a.destino) if a.destino else REPO.parent / "_upload_drive"
    frente = raiz / FRENTE
    dest = frente / EQUIPAMENTO
    if frente.exists():
        if not a.forcar:
            print(f"ERRO: {frente} já existe. Use --forcar para sobrescrever.")
            return 2
        shutil.rmtree(frente)
    dest.mkdir(parents=True)

    avisos: list[str] = []

    # ── raiz ────────────────────────────────────────────────────────────────
    for nome in ("README.md", "metadata.csv", "registro_trips.csv"):
        o = EQUIP_DIR / nome
        if o.exists():
            shutil.copy2(o, dest / nome)
        else:
            avisos.append(f"{nome} não existe em {EQUIP_DIR}")

    # ── dados: só as pastas <ini>_<fim> com o CSV consolidado ───────────────
    n_csv = 0
    for per in sorted((EQUIP_DIR / "dados").glob("[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]")):
        csvs = sorted(per.glob("data_*_raw.csv"))
        if not csvs:
            continue
        (dest / "dados" / per.name).mkdir(parents=True, exist_ok=True)
        for c in csvs:
            shutil.copy2(c, dest / "dados" / per.name / c.name)
            n_csv += 1
    if not n_csv:
        avisos.append("nenhum data_*_raw.csv em dados/<ini>_<fim>/ — a pasta vai sem entrada")

    # ── modelos: os JSON; os .pkl só a pedido ───────────────────────────────
    n_bundle, sem_json = 0, []
    for b in sorted((EQUIP_DIR / "modelos").glob("model_*_PCA4SINAIS")):
        if not b.is_dir():
            continue
        alvo = dest / "modelos" / b.name
        alvo.mkdir(parents=True)
        faltando = [j for j in BUNDLE_JSON if not (b / j).exists()]
        if faltando:
            sem_json.append(f"{b.name}: falta {', '.join(faltando)}")
        for j in BUNDLE_JSON:
            if (b / j).exists():
                shutil.copy2(b / j, alvo / j)
        if a.com_pkl:
            for p in b.glob("*.pkl"):
                shutil.copy2(p, alvo / p.name)
        n_bundle += 1
    if sem_json:
        avisos += sem_json
        avisos.append("rode migra_bundle_json.py antes de entregar")
    if not n_bundle:
        avisos.append("nenhum bundle em modelos/")

    # ── documentos ──────────────────────────────────────────────────────────
    docs = dest / "documentos"; docs.mkdir()
    for f in sorted((EQUIP_DIR / "documentos").glob("*")):
        if f.is_file():
            shutil.copy2(f, docs / f.name)
    md = EQUIP_DIR / "scripts" / "INTEGRACAO_DASHBOARD.md"
    if md.exists():
        shutil.copy2(md, docs / "INTEGRACAO_DASHBOARD.md")
    for origem, nome in RELATORIOS.items():
        o = REPO / origem
        if o.exists():
            shutil.copy2(o, docs / nome)
        else:
            avisos.append(f"relatório ausente: {origem}")

    # ── scripts ─────────────────────────────────────────────────────────────
    sc = dest / "scripts"; sc.mkdir()
    for nome in SCRIPTS:
        o = EQUIP_DIR / "scripts" / nome
        if o.exists():
            shutil.copy2(o, sc / nome)
        else:
            avisos.append(f"script ausente: {nome}")

    # ── relatório ───────────────────────────────────────────────────────────
    print(f"-> {frente}   (suba a pasta {FRENTE}/ inteira)\n")
    print(f"   {FRENTE}/{EQUIPAMENTO}/")
    for sub in ("README.md", "metadata.csv", "registro_trips.csv",
                "dados", "modelos", "documentos", "scripts"):
        p = dest / sub
        if p.exists():
            n = f"{len(list(p.rglob('*'))):>4d} itens" if p.is_dir() else "     arquivo"
            print(f"   {sub:<20s} {n}   {mb(p):8.1f} MB")
    print(f"\n   {'TOTAL':<20s} {'':>10s}   {mb(dest):8.1f} MB")
    print(f"   {n_bundle} bundles"
          f"{' com .pkl' if a.com_pkl else ' (só JSON — sem código fechado)'}, "
          f"{n_csv} CSV de dados")

    if avisos:
        print("\n   AVISOS:")
        for w in avisos:
            print(f"     · {w}")
    print("\n   Confira rodando a partir da cópia antes de subir:")
    print(f"     cd {dest / 'scripts'} && python3 tc33003a_exemplo.py --dias 7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
