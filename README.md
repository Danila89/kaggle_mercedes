# kaggle_mercedes
The 11th place solution of Mercedes-Benz Greener Manufacturing competition on Kaggle
The descriprion is here: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36242

The presentation of the solution in Yandex here (Russian audio, English subtitles): https://youtu.be/0qHXNeuNOAE

Also my blog post about the solution is available (Russian language): https://habrahabr.ru/company/ods/blog/336168/

List of files:
1. cluster_target_encoder.py - the class for generating the cluster labels. See: https://www.kaggle.com/daniel89/mercedes-cars-clustering/
2. sep_estimator.py - the class which fits four independent estimators. When predicting it splits the DataFrame by 'labels' columns and use the corresponding estimator to predict targets for corresponding 'labels' values.
3. solution_11_place.ipynb - the main script
