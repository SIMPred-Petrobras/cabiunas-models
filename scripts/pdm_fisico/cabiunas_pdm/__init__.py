"""Replicacao MINIMA do pacote `cabiunas_pdm`, restaurada da branch do Francisco.

POR QUE ESTE PACOTE EXISTE AQUI. O original vivia num diretorio temporario
(`/tmp/.../scratchpad/pdm/src`) que foi apagado, e com ele `piso_fisico.py`
parou de rodar -- justamente o script que GERA o `piso_fisico_cache.npz`, de onde
saem os quatro sinais. O cache existia mas nao era regeneravel: a maior lacuna de
reprodutibilidade do detector.

A fonte foi recuperada de `origin/feat/pdm-deteccao-4sinais:src/cabiunas_pdm/`, e
aqui estao replicados APENAS os simbolos que os nossos scripts consomem:

    config    GRID · TEMPERATURE_TAGS · PRESSURE_TAGS · VIBRATION_TAGS · SENSOR_TAGS
    detector  SUSTAIN · THR_FAM · THR_SPREAD · BLACKOUT · FIT_POINTS
              _spread_mancal · _sustained
    scoring   MultivariateScorer

Copia fiel, nao reimplementacao -- e o que garante que o cache regenerado bata
com o publicado. Mesmo criterio do `publica_clearml.py`, que ja replicava as
constantes para nao depender de nada fora do repositorio.

Se precisar de algo mais do pacote original, tire da branch dele em vez de
reescrever de memoria.
"""
