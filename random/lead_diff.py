import numpy as np

import pomocne_metode_analiza
from pomocne_metode_analiza import process_image_data, full_thing, render_element_grid, ELEMENT_MAP

diff_map = {
    "Pb": {"name": "Pb La - Pb Lb", "kev": 0.0, "cmap": "seismic"}
}



lead_analysis_map = {
    "Pb_a": {"name": "Lead La", "kev": 10.55, "cmap": "Purples"},
    "Pb_b": {"name": "Lead Lb", "kev": 12.61, "cmap": "Purples"}
}



# 2. Process once (Faster I/O)
cube, keys, w, h = process_image_data("Resources/aurora-antico1-prova1/10264", elemenent_map=lead_analysis_map)
render_element_grid(cube, keys,lead_analysis_map, w, h, figname="lead alpha, beta, prova1/10264")

# 3. Perform the subtraction on the NumPy slices
# keys[0] is Pb_a, keys[1] is Pb_b
diff_data = cube[0:1] - cube[1:2]


render_element_grid(diff_data, ["Pb"], diff_map, width=w, height=h, figname= "lead alpha-beta, prova1 10264 ",savename="rezultati/best_j/lead_diff_prova1_10264_alpha-beta.png")

better_main.render_comparisons(strips = [cube, cube, diff_data],
                                element_keys=[keys, keys, ["Pb"]],
                               element_maps = [lead_analysis_map, lead_analysis_map, diff_map],
                               figname = "lead diff prova1 10264",
                               savename = "rezultati/best_j/lead_diff_prova1_10264"
                               )