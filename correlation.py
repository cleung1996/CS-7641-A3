from ucimlrepo import fetch_ucirepo
import seaborn
from sklearn.preprocessing import LabelEncoder

from scipy.io import arff

import pandas as pd

import warnings
warnings.filterwarnings('ignore')
def main():
    data = arff.loadarff('./Rice_Cammeo_Osmancik.arff')
    train = pd.DataFrame(data[0])

    print(f"The number of cases that are Cammeo: {train[train['Class'] == b'Cammeo'].size / train.size * 100}")
    print(f"The number of rice cases: {train['Class'].size}")

    train.loc[train['Class'] == b'Cammeo', 'Class'] = 0
    train.loc[train['Class'] == b'Osmancik', 'Class'] = 1
    corr = train.corr()

    rice_heatmap = seaborn.heatmap(corr, annot=True)
    rice_heatmap.set_yticklabels(rice_heatmap.get_yticklabels(), size = 5, rotation=30)
    rice_heatmap.set_xticklabels(rice_heatmap.get_xticklabels(), size = 5, rotation=30)
    rice_heatmap.set_title('Rice Heatmap - Correlation of Features')
    figure = rice_heatmap.get_figure()
    figure.savefig('rice_corr_heatmap.png', dpi=400)
    figure.clf()

    mushroom = fetch_ucirepo(id=73)
    mushroom_data = mushroom.data

    X = mushroom_data.features
    y = mushroom_data.targets

    mappings = list()
    encoder = LabelEncoder()

    for column in range(len(X.columns)):
        X[X.columns[column]] = encoder.fit_transform(X[X.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)

    print(f"The number of cases that are poisonous: {y[y['poisonous'] == 'p'].size / y.size * 100}")
    print(f"The number of mushroom cases: {y.size}")

    y[y == 'p'] = 1
    y[y == 'e'] = 0
    y = y.astype(int)

    mushroom_data = pd.concat([X,y], axis=1)
    print(mushroom_data['veil-type'])
    mushroom_data=mushroom_data.drop(['veil-type'], axis=1)
    corr = mushroom_data.corr()

    mushroom_heatmap = seaborn.heatmap(corr, annot=True, annot_kws={"size": 3})
    mushroom_heatmap.set_yticklabels(mushroom_heatmap.get_yticklabels(), size=3)
    mushroom_heatmap.set_xticklabels(mushroom_heatmap.get_xticklabels(), size=3, rotation =60)
    mushroom_heatmap.set_title('Mushroom Heatmap - Correlation of Features')
    test = mushroom_heatmap.get_figure()
    test.savefig('mushroom_corr_heatmap.png', dpi=400)
    test.clf()


if __name__ == '__main__':
    main()

