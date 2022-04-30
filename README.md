# SPARNN
Spatial parallel autoreservior neural networks

For "Predicting sea surface temperature based on a parallel autoreservoir computing approach with short-term measured data", the code is in the SPARNN_for_SST.
 
For "Short-term data based spatial parallel autoreservoir computing on spatiotemporally chaotic system prediction"
SPARNN code:
(1) Main_ARNN.m is for table 1-4 of the paper.
(2) Main_ARNN_revise.m, Main_ARNN_revise_bursslator.m and Main_ARNN_revise_SH.m are for table 5-7.
(partly rewrite from https://github.com/RPcb/ARNN)

data generate: contains the programs to solve the Kuramoto-Sivashinsky model(refer to https://github.com/E-Renshaw/kuramoto-sivashinsky), the Brusselator model and Swift-Hohenberg model(refer to https://github.com/tanumoydhar/1D-Swift-Hohenberg-equation)

comparison methods: contains the programs of CNN, LSTM, BiLSTM, Random Forest, CatBoost, LightBoost, XGBoost.


# Citing the project
If you use the code, we kindly ask that you cite the paper using the following reference:

Y. Wang and S. Liu, "Predicting sea surface temperature based on a parallel autoreservoir computing approach with short-term measured data," in IEEE Geoscience and Remote Sensing Letters, doi: https://doi.org/10.1109/LGRS.2022.3167408

Y. Wang and S. Liu, "Short-term data-based spatial parallel autoreservoir computing on spatiotemporally chaotic system prediction," in Neural Computing & Applications , doi: https://doi.org/10.1007/s00521-021-06854-2
