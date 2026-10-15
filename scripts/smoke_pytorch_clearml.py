#!/usr/bin/env python3
"""Teste de fumaça do PyTorch num worker do ClearML (fila Cica): versão, GPU e um treino mínimo do AE da nova frente.

Local (sem ClearML):      PYTHONPATH=src python scripts/smoke_pytorch_clearml.py
Enfileirar (PUBLICA):     PYTHONPATH=src python scripts/smoke_pytorch_clearml.py --enqueue --queue NOME [--image IMAGEM]
O enfileiramento cria uma tarefa no servidor ClearML; só rodar com o usuário ciente da fila e da imagem.
"""
import argparse, time


def teste():
    import numpy as np, torch
    from nv.modelos.ae_torch import ModeloAE
    info = dict(torch=torch.__version__, cuda=torch.cuda.is_available(), dispositivos=torch.cuda.device_count(), threads=torch.get_num_threads())
    rng = np.random.default_rng(0); X = rng.normal(size=(5000, 6)); t0 = time.time()
    m = ModeloAE(6, 0, semente=0).fit(X, np.zeros((5000, 0)), epocas=3); s = m.score(X[:100], np.zeros((100, 0)))
    info.update(segundos=round(time.time() - t0, 2), escore_mediano=float(np.nanmedian(s)))
    print(info, flush=True); return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--enqueue", action="store_true"); ap.add_argument("--queue", default=None); ap.add_argument("--image", default="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    a = ap.parse_args()
    if a.enqueue:
        from clearml import Task
        if not a.queue:
            raise SystemExit("--queue é obrigatório com --enqueue")
        t = Task.init(project_name="cabiunas-nova-visao", task_name="smoke_pytorch"); t.set_base_docker(a.image); t.set_packages(["torch", "numpy", "scikit-learn", "pandas"]); t.execute_remotely(queue_name=a.queue)
    info = teste()
    if a.enqueue:
        from clearml import Task
        Task.current_task().get_logger().report_text(str(info))
