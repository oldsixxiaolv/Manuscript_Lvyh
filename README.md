## ***New Insights from TRMM Legacy: Stage-Dependent Relationships Between Lightning and Convective Structure in Thunderstorms***

*Creator: Lyu yihang  ;  Institude: Lanzhou University  ;  Supervisor：Wu Xueke*

### **Introduction**

​    I am glad to share the codes of our research with you here in GitHub. Our research is to solve the problem that the lightning data assimilation right now doesn't explicitly take the different life stages of thunderstorm into account. 
​    Reliable relationship between lightning and convective parameters is fundamental to better understanding thunderstorm physics mechinism and improving severe storm simulation. However, this relationship varies significantly across storm's lifecycle. To address this, we developed an improved machine learning-based K-means clustering method using echo structure parameters from the 16-year Tropical Rainfall Measuring Mission (TRMM) Precipitation Radar (PR) observations. This method effectively classifies snapshot thunderstorms from non-sun-synchronous satellites by combining evolving convective precipitation ratios and structural characteristics. The majority (81%) of thunderstorms, classified as Compact Storms (CSs), were further subdivided into three distinct lifecycle stages: Pre-Mature, Mature, and Post-Mature. These derived stage-dependent relationships align well with the known evolution characteristics of thunderstorm.  It is found that despite comparable lightning flash rates, Pre-Mature thunderstorms exhibit smaller horizontal scales yet higher lightning generation efficiency, whereas Post-Mature thunderstorms demonstrate the lowest lightning efficiency but larger scales. Furthermore, using parameters of the 40-dBZ intense echo core as an example, the relationships between lightning and convective structures across thunderstorm lifecycle stages are examined. The results reveal that, although the correlation coefficients may be similar across stages, the parameters of the linear fitting equations can differ significantly. These different relationships indicate distinct underlying physical mechanisms governing electrification in each stages of thunderstorm. 
​    By explicitly classifying thunderstorm life stages and rebuilding the stage-dependent lightning-convective structure relationships, this study provides insights for refining model nudging functions through stage-discriminant parameters (Qie et al., 2014b) and for improving simulations using key storm descriptors. Furthermore, to address the complex, variable, regional, and multivariate nature of thunderstorm electrification, developing an improved stage-dependent parameterization scheme using AI (e.g. machine learning) is valuable for advancing severe storm forecasting and storm warning. It is important to note that this study is confined to the tropical continental region of Africa as a case study area. However, the methodology developed here can be replicated for thunderstorm investigations in other regions worldwide, thereby contributing to a more comprehensive global understanding of thunderstorms.
**The codes for analyses and visualizations are open here for readers. Your precious advice will be highly appreicated. If you would like to contribute to our future research, please contact me through email (320220903211@lzu.edu.cn or lvyihang200411@outlook.com).**
 					

### **Insturction**

​    1.The collated database can be found in the reference.
​    2.The files with **dismissed** in their names are not for the thesis right now, and the number of the fig is not absolute correspondent to thesis.
​    3.These programs need to be modified before you run it. (The path of hdf files which you can download through the database in the reference)

### **Acknowledgements**

​    I am grateful to my mentor Wu Xueke. As an undergraduate student, my research skills still need improvement. He is very patient and responsible for me, I hope to work more efficiently with him in research in the future.

### **Reference**

​    **Lyu, . yihang . (2025). Thunderstorms stage identification (2.0) [Data set]. Lyu yihang. https://doi.org/10.5281/zenodo.18041880**