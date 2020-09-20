import pandas as pd
from sklearn.decomposition import PCA


def reduce_pca(data, n_components:int):
    id_c = data.pop('id')
    class_c = data.pop('class')
    pca = PCA(n_components)
    data = pca.fit_transform(data)
    data = pd.DataFrame(data)
    data['id'] = id_c
    data['class'] = class_c
    return data
