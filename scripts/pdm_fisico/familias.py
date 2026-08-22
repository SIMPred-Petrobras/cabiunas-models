#!/usr/bin/env python3
"""Familias fisicas e regra de votacao.

Duas licoes opostas tem de conviver:
  - a autopsia mostra evento que aparece em UM canal so (PDI_0302 em 29/04,
    vib_351 em 11/04). Exigir consenso mata esses.
  - o maximo sobre 25 canais fica elevado quase sempre, porque basta um canal
    heterocedastico. Foi o que deu 150 h/mes de alarme falso.
O meio-termo e o maximo DENTRO da familia (que preserva o canal solitario, ja
que os irmaos medem a mesma coisa) e voto ENTRE familias (que exige que o
desvio seja de um mecanismo, nao de um instrumento).
"""
FAM = {
    "mancal":    ["spread_0301", "spread_0303", "spread_0305", "spread_0307", "dT_manc_oleo"],
    "vibracao":  ["vib_351X", "vib_351Y", "vib_352X", "vib_352Y", "vib_353X",
                  "vib_353Y", "vib_354X", "vib_354Y", "vib_355X", "vib_355Y"],
    "selagem":   ["selagem"],
    "oleo":      ["PI_0307", "PI_0308", "PDI_0301", "PDI_0302", "PDI_0317", "PDI_0338"],
    "combustao": ["T5_spread", "t5r__0"],
    "processo":  ["gas_0315"],
}
CONJ = {
    "todas":      list(FAM),
    "maquina":    ["mancal", "vibracao", "selagem", "oleo"],
    "mecanica":   ["mancal", "vibracao"],
}
