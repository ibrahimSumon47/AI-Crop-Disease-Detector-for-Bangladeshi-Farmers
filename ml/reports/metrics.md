# Model Metrics (test split)

*Model:* `best_model.keras` 
*Data:* `data/processed/test`  (14641 images)  
*Generated:* 2026-10-22T08:02:01+00:00

All numbers below are computed from the actual run — not estimates.

**Overall accuracy: 42.80%  |  Macro F1: 0.299**

| class | precision | recall | F1 | support | mean confidence |
| --- | --- | --- | --- | --- | --- |
| cauliflower_black_rot | 0.664 | 0.482 | 0.558 | 164 | 0.378 |
| cauliflower_healthy | 0.393 | 0.610 | 0.478 | 141 | 0.384 |
| cauliflower_insect_holes | 0.266 | 0.219 | 0.240 | 96 | 0.382 |
| chilli_bacterial_spot | 0.105 | 0.167 | 0.129 | 24 | 0.269 |
| chilli_cercospora_leaf_spot | 0.194 | 0.481 | 0.277 | 27 | 0.416 |
| chilli_curl_virus | 0.615 | 0.750 | 0.676 | 64 | 0.593 |
| chilli_healthy | 0.439 | 0.420 | 0.430 | 69 | 0.314 |
| chilli_nutrient_deficiency | 0.275 | 0.209 | 0.237 | 67 | 0.308 |
| chilli_white_spot | 0.182 | 0.400 | 0.250 | 30 | 0.341 |
| cucumber_anthracnose | 0.000 | 0.000 | 0.000 | 24 | 0.320 |
| cucumber_bacterial_wilt | 0.105 | 0.167 | 0.129 | 24 | 0.374 |
| cucumber_belly_rot | 0.133 | 0.417 | 0.202 | 24 | 0.316 |
| cucumber_downy_mildew | 0.000 | 0.000 | 0.000 | 24 | 0.317 |
| cucumber_gummy_stem_blight | 0.091 | 0.083 | 0.087 | 24 | 0.264 |
| cucumber_healthy | 0.016 | 0.021 | 0.018 | 48 | 0.380 |
| cucumber_pythium_fruit_rot | 0.188 | 0.115 | 0.143 | 26 | 0.291 |
| grape_bacterial_leaf_spot | 0.000 | 0.000 | 0.000 | 15 | 0.432 |
| grape_black_rot | 0.351 | 0.166 | 0.225 | 531 | 0.451 |
| grape_downy_mildew | 0.465 | 0.834 | 0.598 | 145 | 0.589 |
| grape_esca | 0.390 | 0.873 | 0.539 | 623 | 0.629 |
| grape_healthy | 1.000 | 0.079 | 0.147 | 379 | 0.414 |
| grape_leaf_blight | 0.493 | 0.355 | 0.412 | 485 | 0.457 |
| grape_powdery_mildew | 0.196 | 0.443 | 0.271 | 61 | 0.393 |
| maize_common_rust | 0.695 | 0.831 | 0.757 | 537 | 0.642 |
| maize_gray_leaf_spot | 0.382 | 0.442 | 0.410 | 231 | 0.443 |
| maize_healthy | 0.597 | 0.629 | 0.613 | 523 | 0.579 |
| maize_lethal_necrosis | 0.850 | 0.296 | 0.439 | 1745 | 0.445 |
| maize_northern_leaf_blight | 0.475 | 0.534 | 0.503 | 444 | 0.454 |
| maize_streak_virus | 0.564 | 0.830 | 0.672 | 1386 | 0.748 |
| mango_anthracnose | 0.242 | 0.640 | 0.352 | 75 | 0.347 |
| mango_bacterial_canker | 0.500 | 0.093 | 0.157 | 75 | 0.330 |
| mango_cutting_weevil | 0.628 | 0.947 | 0.755 | 75 | 0.843 |
| mango_die_back | 0.722 | 0.173 | 0.280 | 75 | 0.399 |
| mango_gall_midge | 0.500 | 0.533 | 0.516 | 75 | 0.367 |
| mango_healthy | 0.512 | 0.880 | 0.647 | 75 | 0.593 |
| mango_powdery_mildew | 0.480 | 0.480 | 0.480 | 75 | 0.407 |
| mango_sooty_mould | 0.366 | 0.200 | 0.259 | 75 | 0.351 |
| okra_alternaria_leaf_spot | 0.429 | 0.061 | 0.107 | 49 | 0.208 |
| okra_cercospora_leaf_spot | 0.625 | 0.312 | 0.417 | 32 | 0.242 |
| okra_downy_mildew | 0.250 | 0.030 | 0.054 | 33 | 0.237 |
| okra_healthy | 0.368 | 0.149 | 0.212 | 47 | 0.270 |
| okra_leaf_curl_virus | 0.000 | 0.000 | 0.000 | 32 | 0.241 |
| okra_phyllosticta_leaf_spot | 0.375 | 0.088 | 0.143 | 34 | 0.237 |
| papaya_anthracnose | 0.033 | 0.185 | 0.056 | 54 | 0.362 |
| papaya_bacterial_spot | 0.076 | 0.449 | 0.130 | 69 | 0.382 |
| papaya_curl | 0.113 | 0.250 | 0.155 | 88 | 0.415 |
| papaya_healthy | 0.065 | 0.514 | 0.115 | 35 | 0.363 |
| papaya_ring_spot | 0.090 | 0.212 | 0.127 | 80 | 0.373 |
| potato_early_blight | 0.660 | 0.316 | 0.427 | 450 | 0.424 |
| potato_healthy | 0.087 | 0.522 | 0.148 | 69 | 0.471 |
| potato_late_blight | 0.383 | 0.298 | 0.335 | 450 | 0.358 |
| rice_bacterial_leaf_blight | 0.584 | 0.382 | 0.462 | 301 | 0.395 |
| rice_blast | 0.261 | 0.250 | 0.255 | 216 | 0.342 |
| rice_brown_spot | 0.342 | 0.044 | 0.078 | 294 | 0.400 |
| rice_healthy | 0.168 | 0.921 | 0.285 | 38 | 0.722 |
| rice_tungro | 0.533 | 0.058 | 0.105 | 276 | 0.299 |
| tomato_bacterial_spot | 0.820 | 0.147 | 0.250 | 958 | 0.268 |
| tomato_early_blight | 0.247 | 0.162 | 0.196 | 450 | 0.332 |
| tomato_healthy | 0.438 | 0.641 | 0.521 | 716 | 0.583 |
| tomato_late_blight | 0.496 | 0.465 | 0.480 | 860 | 0.386 |
| tomato_leaf_mold | 0.279 | 0.331 | 0.303 | 429 | 0.338 |
