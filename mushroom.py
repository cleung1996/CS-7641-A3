from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import scipy

import scipy.stats
from sklearn.random_projection import GaussianRandomProjection
from sklearn import mixture
from sklearn.mixture import GaussianMixture


import time
from ucimlrepo import fetch_ucirepo

import warnings
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings('ignore')


def main():
    mushroom = fetch_ucirepo(id=73)
    mushroom_data = mushroom.data

    X = mushroom_data.features
    y = mushroom_data.targets.poisonous

    mappings = list()
    encoder = LabelEncoder()

    for column in range(len(X.columns)):
        X[X.columns[column]] = encoder.fit_transform(X[X.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)

    y[y == 'p'] = 1
    y[y == 'e'] = 0
    y = y.astype(int)

    # Pre process data onto the same scale
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=620)

    clusters = np.arange(1,30,1)
    distortions = []

    for cluster in clusters:
        res = KMeans(n_clusters=cluster, random_state=620)
        res.fit(X)
        distortions.append(res.inertia_)

    inertia = np.array(distortions)
    plt.plot(clusters, distortions)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('Mushroom - Inertia vs Clusters')
    plt.grid()
    plt.savefig('MUSHROOM_KMeans_Inertia.png')
    plt.show()


    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html analyze silhouette scores
    # cluster size = 4 #provided the tapered results. Use this to calculate the Silhouette
    best_cluster = 6
    fig, (ax1) = plt.subplots(1, 1)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (best_cluster + 1) * 10])

    clusterer = KMeans(n_clusters=best_cluster, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        best_cluster,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(best_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / best_cluster)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Mushroom - Silhouette Scores with %d Clusters"
        % best_cluster,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig("MUSHROOM_KMeans_Silhouette.png")
    plt.show()


    final_cluster = KMeans(n_clusters=4, random_state=620).fit(X)
    print(f"Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"AMI Score: {AMI_score}")


    ####### EM

    clusters = np.arange(1,10,1)
    distortions = []

    for cluster in clusters:
        res = GaussianMixture(n_components=cluster, random_state=620)
        res.fit(X)
        distortions.append(res.bic(X))

    distortions = np.array(distortions)
    plt.plot(clusters, distortions)
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.title('Mushroom - EM - Bic vs Number of Components')
    plt.grid()
    plt.savefig('MUSHROOM_EM_Bic.png')
    plt.show()

    best_cluster = 8
    fig, (ax1) = plt.subplots(1, 1)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (best_cluster + 1) * 10])

    clusterer = mixture.GaussianMixture(n_components=best_cluster)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        best_cluster,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(best_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / best_cluster)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.suptitle(
        "Mushroom - EM - Silhouette Scores with %d Clusters"
        % best_cluster,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig("MUSHROOM_EM_Silhouette.png")
    plt.show()


    final_cluster = GaussianMixture(n_components=3, random_state=620).fit(X)
    silhouette_score_value = silhouette_score(X, final_cluster.predict(X))
    print(f"Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.predict(X))
    print(f"AMI Score: {AMI_score}")

    # ################# PCA

    pca = PCA().fit(X)
    plt.figure()
    ranges = np.arange(1, pca.explained_variance_ratio_.size + 1)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(ranges, cumulative_variance, label='Cumulative Variance')
    plt.xlabel('Component')
    plt.ylabel('Variance')
    plt.title('Mushroom - PCA - Variance vs. Component')
    plt.legend()
    plt.grid()
    plt.savefig('MUSHROOM_PCA_Var.png')
    plt.show()

    # Can reduce from 7 to 3 comfortably
    n = 3
    pca = PCA(n_components=n).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=y, cmap='plasma', edgecolor='k', s=30)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Mushroom - PCA: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('MUSHROOM_PCA_Result.png')
    plt.show()

    # ## ICA


    kurtosis = []
    for i in range(1,23):
        X_ICA = FastICA(n_components=i)
        X_ICA = X_ICA.fit_transform(X)
        kurto = scipy.stats.kurtosis(X_ICA)
        kurtosis.append(np.mean(kurto)/i)
    kurtosis = np.array(kurtosis)
    plt.plot(ranges, kurtosis)
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Mean Kurtosis')
    plt.title('Mushroom - Mean Kurtosis vs Number of Components')
    plt.grid()
    plt.savefig('MUSHROOM_ICA_Kur.png')
    plt.show()

    n = 3
    ica = FastICA(n_components=n).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(ica[:, 0], ica[:, 1], ica[:, 2], c=y, cmap='plasma', edgecolor='k', s=30)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Mushroom - ICA: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('MUSHROOM_ICA_Result.png')
    plt.show()

    # ### RP

    n = 3
    rp = GaussianRandomProjection(n_components=n).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], c=y, cmap='plasma', edgecolor='k', s=30)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Mushroom - RP: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('MUSHROOM_RP_Result.png')
    plt.show()

    ## PCA with KMeans

    pca = PCA().fit(X)
    x_pca = pca.transform(X)
    clusters = np.arange(1,20,1)
    distortions = []
    silhouettes = []
    iterations = 10


    for cluster in clusters:
        res = KMeans(n_clusters=cluster, random_state=620)
        res.fit(x_pca)
        cluster_labels = res.predict(x_pca)
        distortions.append(res.inertia_)
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_pca, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, distortions)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('Mushroom (KMeans with PCA) - Inertia vs Clusters')
    plt.grid()
    plt.savefig('MUSHROOM_KMeans_PCA_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (KMeans with PCA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.savefig('MUSHROOM_KMeans_PCA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=10, random_state=620).fit(x_pca)
    print(f"Mushroom: KMeans/PCA- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Mushroom: KMeans/PCA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Mushroom: KMeans/PCA - AMI Score: {AMI_score}")


    ## ICA with KMeans

    ica = FastICA().fit(X)
    x_ica = ica.transform(X)
    clusters = np.arange(1,20,1)
    distortions = []
    silhouettes = []
    iterations = 10

    for cluster in clusters:
        res = KMeans(n_clusters=cluster, random_state=620)
        res.fit(x_ica)
        cluster_labels = res.predict(x_ica)
        distortions.append(res.inertia_)
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_ica, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, distortions)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('Mushroom (KMeans with ICA) - Inertia vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_KMeans_ICA_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (KMeans with ICA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_KMeans_ICA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=19, random_state=620).fit(x_ica)
    print(f"Mushroom: KMeans/ICA- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Mushroom: KMeans/ICA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Mushroom: KMeans/ICA - AMI Score: {AMI_score}")

    ## RP with KMeans

    x_rp = GaussianRandomProjection(n_components=3).fit_transform(X)
    clusters = np.arange(1,20,1)
    distortions = []
    silhouettes = []

    for cluster in clusters:
        res = KMeans(n_clusters=cluster, random_state=620)
        res.fit(x_rp)
        cluster_labels = res.predict(x_rp)
        distortions.append(res.inertia_)
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_rp, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, distortions)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('Mushroom (KMeans with RP) - Inertia vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_KMeans_RP_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (KMeans with RP) - Silhouette Scores vs Number of Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_KMeans_RP_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=3, random_state=620).fit(x_rp)
    print(f"Mushroom: KMeans/RP- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Mushroom: KMeans/RP - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Mushroom: KMeans/RP - AMI Score: {AMI_score}")

    ## PCA with EM
    pca = PCA().fit(X)
    x_pca = pca.transform(X)
    clusters = np.arange(1,20,1)
    bic = []
    silhouettes = []

    for cluster in clusters:
        res = GaussianMixture(n_components=cluster, random_state=620)
        res.fit(x_pca)
        cluster_labels = res.predict(x_pca)
        bic.append(res.bic(x_pca))
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_pca, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, bic)
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.title('Mushroom (EM with PCA) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_EM_PCA_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (EM with PCA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_EM_PCA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=10, random_state=620).fit(x_pca)
    test = final_cluster.predict(x_pca)
    print(f"Mushroom: EM/PCA- BIC/Distortion Score: {final_cluster.bic(x_pca)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Mushroom: EM/PCA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Mushroom: EM/PCA - AMI Score: {AMI_score}")


    ########### ICA with EM
    ica = FastICA().fit(X)
    x_ica = ica.transform(X)
    clusters = np.arange(1,20,1)
    bic = []
    silhouettes = []

    for cluster in clusters:
        res = GaussianMixture(n_components=cluster, random_state=620)
        res.fit(x_ica)
        cluster_labels = res.predict(x_ica)
        bic.append(res.bic(x_ica))
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_ica, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, bic)
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')
    plt.title('Mushroom (EM with ICA) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_EM_ICA_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (EM with ICA) - Silhouette Scores vs Clusters')
    plt.xticks(np.arange(2, 20, step=1))
    plt.grid()
    plt.savefig('MUSHROOM_EM_ICA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=19, random_state=620)
    final_cluster.fit(x_ica)
    test = final_cluster.predict(x_ica)
    print(f"Mushroom: EM/ICA- BIC/Distortion Score: {final_cluster.bic(x_ica)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Mushroom: EM/ICA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Mushroom: EM/ICA - AMI Score: {AMI_score}")

    ########## RP with EM
    x_rp = GaussianRandomProjection(n_components=3).fit_transform(X)
    clusters = np.arange(1,20,1)
    bic = []
    silhouettes = []

    for cluster in clusters:
        res = GaussianMixture(n_components=cluster, random_state=620)
        res.fit(x_rp)
        cluster_labels = res.predict(x_rp)
        bic.append(res.bic(x_rp))
        if cluster > 1:
            silhouette_cluster = silhouette_score(x_rp, cluster_labels)
            silhouettes.append(silhouette_cluster)
        else:
            silhouettes.append(np.nan)

    plt.plot(clusters, bic)
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')
    plt.title('Mushroom (EM with RP) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('MUSHROOM_EM_RP_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Mushroom (EM with RP) - Silhouette Scores vs Clusters')
    plt.xticks(np.arange(2, 20, step=1))
    plt.grid()
    plt.savefig('MUSHROOM_EM_RP_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=3, random_state=620)
    final_cluster.fit(x_rp)
    test = final_cluster.predict(x_rp)
    print(f"Mushroom: EM/RP- BIC/Distortion Score: {final_cluster.bic(x_rp)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Mushroom: EM/RP - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Mushroom: EM/RP - AMI Score: {AMI_score}")

if __name__ == "__main__":
    main()