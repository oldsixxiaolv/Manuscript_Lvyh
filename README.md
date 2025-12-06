**Stage-dependent relationships between lightning and storm structures**

Creator: Lv yihang  ;  Institude: Lanzhou University  ;  Supervisor：Wu Xueke

**Introduction**

  		I am glad to share the codes of our research with you here in GitHub. Our research is to solve the problem that the lightning data assimilation right now doesn't explicitly take the different life stages of thunderstorm into account. 
		We re-examine and explore invaluable 16-year observations from TRMM satellite, considering the structural characteristics of thunderstorms at different stages of their life cycle. We have found that tropical Africa is the ideal region on Earth in which to study the relationship between lightning and the convective structural parameters of thunderstorms, due to its high density of lightning and thunderstorms, and the intensity and depth of its thunderstorms.Statistical analysis indicates that TRMM-observed thunderstorm snapshots can be classified into different life stages based on evolutionary characteristics of the thunderstorms. We have developed a method for identifying thunderstorm life stages using convective precipitation ratio and radar structural features, including 20, 30, and 40 dBZ echo top heights and horizontal scales. The results align well with the characteristics of convective evolution across different stages of thunderstorm development. Accordingly, stage-dependent relationships between lightning and convective structural characteristics have been reconstructed for the first time using TRMM snapshot observations. These differ significantly from relationships observed without stage distinction.
	 	This study for the first time explicitly classifies the life stage of thunderstorm and rebuilds the stage-dependent relationships between lightning flash rates and thunderstorm convective structures, the results are expected to offer novel insights to establish more precise nudging functions through adding the discriminant parameters of thunderstorm life stages and greatly improve the simulation capabilities of thunderstorm with the function of maximum echo top height, echo volume parameters and lightning flash rates appended. However, although the variables chosen for cluster analysis effectively correspond to the characteristics of the classified thunderstorms at each stage, allowing for variations in convective parameters and lightning flash rates across different stages, due to the unsupervised nature of the K-means clustering algorithm, it is not possible to identify thunderstorms absolutely and correctly. Consequently, future studies aim to optimize the classification methods and boundaries. Additionally, the internal structure of thunderstorm cloud is quite complex, and multiple meteorological factors affect its development, as a result, lightning is influenced by various parameters rather than several simple variables. Therefore, a multivariate parameterization approach for thunderstorm lightning across thunderstorm different stages combining artificial intelligence will be studied in future.
	 	The codes for analyses and visualizations are open here for readers. Your precious advice will be highly appreicated. If you would like to contribute to our future research, please contact me through email (320220903211@lzu.edu.cn).
 					
**Insturction**

		1.The collated database can be found in the Reference, the duqu_last.xlsx is used for the first clustering (We used SPSS for K-means clustering), julei_x.xlsx and output_julei.xlsx are the first and second original data for clustering (both K-means), the stage1/2/3.xlsx represent the Pre-mature, mature and Post-mature stage thunderstorms respectively. 
		2.The files with **dismissed** in their names are not for the thesis right now, so the number of the fig is not absolute correspondent to thesis.
		3.These programs need to be modified before you run it. (The path of xlsx and hdf files)

**Acknowledgements**

		I am grateful to my mentor Wu Xueke. As an undergraduate student, my research skills still need improvement. He is very patient and responsible for me, I hope to work more efficiently with him in research in the future.

**Reference**

Lv yihang. (2025). Thunderstorms stage identification [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16662144
