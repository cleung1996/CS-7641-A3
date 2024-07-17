from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples


from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

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


def main():
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
    rice_data = rice_cammeo_and_osmancik.data

    X = rice_data.features
    y = rice_data.targets.Class
    y = y.replace({'Cammeo': 0, 'Osmancik': 1})
    print("% Osmancik Cases:", y[y == 1].size / y.size * 100)

    # Pre process data onto the same scale
    X = preprocessing.scale(X)

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
    plt.title('Rice - Inertia vs Clusters')
    plt.grid()
    plt.savefig('RICE_KMeans_Inertia.png')
    plt.show()


    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html analyze silhouette scores
    ## cluster size = 4 provided the tapered results. Use this to calculate the Silhouette
    best_cluster = 4
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

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

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / best_cluster)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Rice - Silhouette Scores with %d Clusters"
        % best_cluster,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig("RICE_KMeans_Silhouette.png")
    plt.show()


    final_cluster = KMeans(n_clusters=4, random_state=620).fit(X)
    print(f"Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"AMI Score: {AMI_score}")


    # ######## EM

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
    plt.title('Rice - EM - Bic vs Number of Components')
    plt.grid()
    plt.savefig('RICE_EM_Bic.png')
    plt.show()

    best_cluster = 4
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
        "Rice - EM - Silhouette Scores with %d Clusters"
        % best_cluster,
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig("RICE_EM_Silhouette.png")
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
    plt.title('Rice - PCA - Variance vs. Component')
    plt.legend()
    plt.grid()
    plt.savefig('RICE_PCA_Var.png')
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
    ax.set_title('Rice - PCA: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('RICE_PCA_Result.png')
    plt.show()

    # ## ICA


    kurtosis = []
    for i in range(1,8):
        X_ICA = FastICA(n_components=i)
        X_ICA = X_ICA.fit_transform(X)
        kurto = scipy.stats.kurtosis(X_ICA)
        kurtosis.append(np.mean(kurto)/i)
    kurtosis = np.array(kurtosis)
    plt.plot(ranges, kurtosis)
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Mean Kurtosis')
    plt.title('Rice - Mean Kurtosis vs Number of Components')
    plt.grid()
    plt.savefig('RICE_ICA_Kur.png')
    plt.show()

    n = 3
    ica = FastICA(n_components=n).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(ica[:, 0], ica[:, 1], ica[:, 2], c=y, cmap='plasma', edgecolor='k', s=30)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Rice - ICA: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('RICE_ICA_Result.png')
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
    ax.set_title('Rice - RP: 3 Dimensions')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.savefig('RICE_RP_Result.png')
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
    plt.title('Rice (KMeans with PCA) - Inertia vs Clusters')
    plt.grid()
    plt.savefig('RICE_KMeans_PCA_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (KMeans with PCA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.savefig('RICE_KMeans_PCA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=2, random_state=620).fit(x_pca)
    print(f"Rice: KMeans/PCA- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Rice: KMeans/PCA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Rice: KMeans/PCA - AMI Score: {AMI_score}")


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
    plt.title('Rice (KMeans with ICA) - Inertia vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_KMeans_ICA_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (KMeans with ICA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_KMeans_ICA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=6, random_state=620).fit(x_ica)
    print(f"Rice: KMeans/ICA- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Rice: KMeans/ICA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Rice: KMeans/ICA - AMI Score: {AMI_score}")

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
    plt.title('Rice (KMeans with RP) - Inertia vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_KMeans_RP_Inertia.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (KMeans with RP) - Silhouette Scores vs Number of Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_KMeans_RP_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = KMeans(n_clusters=2, random_state=620).fit(x_rp)
    print(f"Rice: KMeans/RP- Inertia/Distortion Score: {final_cluster.inertia_}")
    silhouette_score_value = silhouette_score(X, final_cluster.labels_)
    print(f"Rice: KMeans/RP - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, final_cluster.labels_)
    print(f"Rice: KMeans/RP - AMI Score: {AMI_score}")

    # ## PCA with EM
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
    plt.title('Rice (EM with PCA) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_EM_PCA_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (EM with PCA) - Silhouette Scores vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_EM_PCA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=2, random_state=620).fit(x_pca)
    test = final_cluster.predict(x_pca)
    print(f"Rice: EM/PCA- BIC/Distortion Score: {final_cluster.bic(x_pca)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Rice: EM/PCA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Rice: EM/PCA - AMI Score: {AMI_score}")


    ############ ICA with EM
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
    plt.title('Rice (EM with ICA) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_EM_ICA_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (EM with ICA) - Silhouette Scores vs Clusters')
    plt.xticks(np.arange(2, 20, step=1))
    plt.grid()
    plt.savefig('RICE_EM_ICA_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=2, random_state=620)
    final_cluster.fit(x_ica)
    test = final_cluster.predict(x_ica)
    print(f"Rice: EM/ICA- BIC/Distortion Score: {final_cluster.bic(x_ica)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Rice: EM/ICA - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Rice: EM/ICA - AMI Score: {AMI_score}")

    ########### RP with EM
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
    plt.title('Rice (EM with RP) - BIC vs Clusters')
    plt.grid()
    plt.xticks(np.arange(2, 20, step=1))
    plt.savefig('RICE_EM_RP_BIC.png')
    plt.show()
    plt.close()

    plt.plot(clusters, silhouettes)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Rice (EM with RP) - Silhouette Scores vs Clusters')
    plt.xticks(np.arange(2, 20, step=1))
    plt.grid()
    plt.savefig('RICE_EM_RP_Silhouette.png')
    plt.show()
    plt.close()

    final_cluster = GaussianMixture(n_components=2, random_state=620)
    final_cluster.fit(x_rp)
    test = final_cluster.predict(x_rp)
    print(f"Rice: EM/RP- BIC/Distortion Score: {final_cluster.bic(x_rp)}")
    silhouette_score_value = silhouette_score(X, test)
    print(f"Rice: EM/RP - Silhouette Score: {silhouette_score_value}")
    AMI_score = adjusted_mutual_info_score(y, test)
    print(f"Rice: EM/RP - AMI Score: {AMI_score}")

    #######################################################################################

    ###### NN with Dim Reduction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=620)

    cnn = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000)
    parameter_range = np.logspace(-6, 3, 10)
    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="alpha",
                                                   param_range=parameter_range, cv=5)

    # Plot validation curve 1 - Accuracy Vs Alpha
    plot_validation_curves(train_results, test_results,
                           title="Rice - Neural Network - Validation Curve (Accuracy vs Alpha)", x_label="Alpha",
                           y_label="Accuracy", save_name='rice_NN_validation_1_alpha.png',
                           parameter_range=parameter_range, is_log=True)

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="learning_rate_init",
                                                   param_range=parameter_range, cv=5)

    # Plot validation curve 2 - Accuracy Vs Learning Rate
    plot_validation_curves(train_results, test_results,
                           title="Rice - Neural Network - Validation Curve (Accuracy vs Learning Rate)",
                           x_label="Learning Rate", y_label="Accuracy",
                           save_name='rice_NN_validation_2_learning_rate.png', parameter_range=parameter_range,
                           is_log=True)

    # Plot validation curve 3 - Accuracy vs Num Nodes

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="hidden_layer_sizes",
                                                   param_range=np.arange(2, 20, 2), cv=5)

    plot_validation_curves(train_results=train_results, test_results=test_results,
                           title="Rice - Neural Network - Validation Curve (Accuracy vs. # Nodes (Layers kept at 1))",
                           x_label="Num Nodes (in Single Layer)", y_label="Accuracy",
                           save_name="rice_NN_validation_3_nodes.png", parameter_range=np.arange(2, 20, 2),
                           is_log=False)

    # Plot validation curve 4 - Accuracy vs Activation
    total_train_results = []
    total_test_results = []

    p_range = ["relu", "logistic", "tanh"]

    nn_relu = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="relu")
    nn_relu.fit(X_train, y_train)
    y_predict_train = nn_relu.predict(X_train)
    y_predict_test = nn_relu.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_logistic = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="logistic")
    nn_logistic.fit(X_train, y_train)
    y_predict_train = nn_logistic.predict(X_train)
    y_predict_test = nn_logistic.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_tanh = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="tanh")
    nn_tanh.fit(X_train, y_train)
    y_predict_train = nn_tanh.predict(X_train)
    y_predict_test = nn_tanh.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    train_results_mean = np.array(total_train_results)
    test_results_mean = np.array(total_test_results)

    plt.figure()
    X_axis = np.arange(len(p_range))

    plt.bar(X_axis - 0.2, train_results_mean, 0.4)
    plt.bar(X_axis + 0.2, test_results_mean, 0.4)
    plt.xticks(X_axis, p_range)

    for i in range(len(p_range)):
        plt.text(i - 0.2, train_results_mean[i], str(round(train_results_mean[i], 3)), ha='center', va='bottom')
        plt.text(i + 0.2, test_results_mean[i], str(round(test_results_mean[i], 3)), ha='center', va='bottom')

    plt.legend(['Train Results', 'Test Results'], loc='best')
    plt.title('Rice - Neural Network - Validation: Accuracy vs Activation Func', fontsize=10)
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('rice_NN_validation_4_activation.png')
    plt.show()

    # # Rice - Neural Network Optimized Parameters: {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 16, 'learning_rate_init': 0.01}
    cnn_learn = MLPClassifier(hidden_layer_sizes=(16, ), random_state=620, max_iter=3000, learning_rate_init=0.01, alpha=0.001, activation='tanh')
    train_sizes_abs, train_results, test_results = learning_curve(cnn_learn, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)
    param_range = np.linspace(0.1, 1.0, 10) * 100


    start_time = time.time()
    cnn_learn.fit(X_train, y_train)
    end_time = time.time()

    print("RICE - Benchmark Time:", str(end_time- start_time))


    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                         title="Rice - Neural Network - Learning Curve (Accuracy vs % Trained)",
                         save_name="NN_rice_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy",
                         parameter_range=param_range)

    # ##### PCA
    pca = PCA(n_components= 2).fit(X)
    X_pca = pca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
    cnn = MLPClassifier(random_state=620, max_iter=2000)
    parameter_range = np.logspace(-6, 3, 10)

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="alpha",
                                                   param_range=parameter_range, cv=5)

    # Plot validation curve 1 - Accuracy Vs Alpha
    plot_validation_curves(train_results, test_results,
                           title="Rice - PCA - Neural Network - Validation Curve (Accuracy vs Alpha)", x_label="Alpha",
                           y_label="Accuracy", save_name='rice_NN_PCA_validation_1_alpha.png',
                           parameter_range=parameter_range, is_log=True)

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="learning_rate_init",
                                                   param_range=parameter_range, cv=5)

    # Plot validation curve 2 - Accuracy Vs Learning Rate
    plot_validation_curves(train_results, test_results,
                           title="Rice - PCA - Neural Network - Validation Curve (Accuracy vs Learning Rate)",
                           x_label="Learning Rate", y_label="Accuracy",
                           save_name='rice_NN_PCA_validation_2_learning_rate.png', parameter_range=parameter_range,
                           is_log=True)

    # Plot validation curve 3 - Accuracy vs Num Nodes

    train_results, test_results = validation_curve(cnn, X_train, y_train, param_name="hidden_layer_sizes",
                                                   param_range=np.arange(2, 20, 2), cv=5)

    plot_validation_curves(train_results=train_results, test_results=test_results,
                           title="Rice - PCA - Neural Network - Validation Curve (Accuracy vs. # Nodes (Layers kept at 1))",
                           x_label="Num Nodes (in Single Layer)", y_label="Accuracy",
                           save_name="rice_NN_PCA_validation_3_nodes.png", parameter_range=np.arange(2, 20, 2),
                           is_log=False)

    # Plot validation curve 4 - Accuracy vs Activation
    total_train_results = []
    total_test_results = []

    p_range = ["relu", "logistic", "tanh"]

    nn_relu = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="relu")
    nn_relu.fit(X_train, y_train)
    y_predict_train = nn_relu.predict(X_train)
    y_predict_test = nn_relu.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_logistic = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="logistic")
    nn_logistic.fit(X_train, y_train)
    y_predict_train = nn_logistic.predict(X_train)
    y_predict_test = nn_logistic.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    nn_tanh = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=620, max_iter=3000, activation="tanh")
    nn_tanh.fit(X_train, y_train)
    y_predict_train = nn_tanh.predict(X_train)
    y_predict_test = nn_tanh.predict(X_test)
    total_train_results.append(accuracy_score(y_predict_train, y_train))
    total_test_results.append(accuracy_score(y_test, y_predict_test))

    train_results_mean = np.array(total_train_results)
    test_results_mean = np.array(total_test_results)

    plt.figure()
    X_axis = np.arange(len(p_range))

    plt.bar(X_axis - 0.2, train_results_mean, 0.4)
    plt.bar(X_axis + 0.2, test_results_mean, 0.4)
    plt.xticks(X_axis, p_range)

    for i in range(len(p_range)):
        plt.text(i - 0.2, train_results_mean[i], str(round(train_results_mean[i], 3)), ha='center', va='bottom')
        plt.text(i + 0.2, test_results_mean[i], str(round(test_results_mean[i], 3)), ha='center', va='bottom')

    plt.legend(['Train Results', 'Test Results'], loc='best')
    plt.title('Rice - PCA - Neural Network - Validation: Accuracy vs Activation Func', fontsize=10)
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('rice_NN_validation_4_activation.png')
    plt.show()

    param_grid = {'alpha': np.logspace(-4,3,8), 'learning_rate_init': np.logspace(-5,1,7), 'activation': ['relu', 'tanh'], 'hidden_layer_sizes':np.arange(2, 20, 2)}

    # cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)
    # print("RICE - PCA - Best params for NN:",cnn_optimized.best_params_)
    # RICE - PCA - Best params for NN: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': 2, 'learning_rate_init': 0.1}

    cnn_optimized = MLPClassifier(random_state=620, max_iter=2000, activation ='tanh', alpha=0.01, hidden_layer_sizes=(2,), learning_rate_init= 0.1)
    start_time = time.time()
    cnn_optimized.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time-start_time
    print(f"Train Time: {time_train}")

    start_time = time.time()
    classifier_accuracy = accuracy_score(y_test, cnn_optimized.predict(X_test))
    end_time = time.time()
    time_infer = end_time-start_time
    print("RICE - PCA - Best params best accuracy:", classifier_accuracy)
    print("Time to infer:",time_infer)

    train_sizes_abs, train_results, test_results = learning_curve(cnn_optimized, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)

    param_range = np.linspace(0.1,1.0,10)*100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results, title="Rice - PCA - Neural Network - Learning Curve (Accuracy vs % Trained)", save_name="NN_PCA_rice_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy", parameter_range=param_range)

    # # ##### ICA

    ica = FastICA(n_components=2, max_iter=10000).fit(X)
    X_ica = ica.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_ica, y, test_size=0.2)
    param_grid = {'alpha': np.logspace(-4,3,8), 'learning_rate_init': np.logspace(-5,1,7), 'activation': ['relu', 'tanh'], 'hidden_layer_sizes':np.arange(2, 20, 2)}
    cnn_optimized = MLPClassifier(random_state=620, max_iter=2000, activation ='relu', alpha=1.0, hidden_layer_sizes=(4,), learning_rate_init= 0.001)

    # cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)
    # RICE - ICA - Best params for NN: {'activation': 'relu', 'alpha': 1.0, 'hidden_layer_sizes': 4, 'learning_rate_init': 0.001}


    start_time = time.time()
    cnn_optimized.fit(X_train, y_train)
    end_time = time.time()

    # print("RICE - ICA - Best params for NN:",cnn_optimized.best_params_)

    time_train = end_time-start_time
    print(f"Train Time: {time_train}")

    start_time = time.time()
    classifier_accuracy = accuracy_score(y_test, cnn_optimized.predict(X_test))
    end_time = time.time()
    time_infer = end_time-start_time
    print("RICE - ICA - Best params best accuracy:", classifier_accuracy)
    print("Time to infer:",time_infer)

    train_sizes_abs, train_results, test_results = learning_curve(cnn_optimized, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10), cv=5)

    param_range = np.linspace(0.1,1.0,10)*100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results, title="Rice - ICA - Neural Network - Learning Curve (Accuracy vs % Trained)", save_name="NN_ICA_rice_learning_curve_score.png", x_label="Percent Trained", y_label="Accuracy", parameter_range=param_range)

    # ##### RP

    x_rp = GaussianRandomProjection(n_components=2).fit_transform(X)
    clusters = np.arange(1,20,1)
    X_train, X_test, y_train, y_test = train_test_split(x_rp, y, test_size=0.2)
    param_grid = {'alpha': np.logspace(-4, 3, 8), 'learning_rate_init': np.logspace(-5, 1, 7),
                  'activation': ['relu', 'tanh'], 'hidden_layer_sizes': np.arange(2, 20, 2)}

    cnn_optimized = MLPClassifier(random_state=620, max_iter=2000, activation='relu', alpha=0.1,
                                  hidden_layer_sizes=(16,), learning_rate_init=0.01)

    # cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)
    # RICE - RP - Best params for NN: {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 16, 'learning_rate_init': 0.01}
    start_time = time.time()
    cnn_optimized.fit(X_train, y_train)
    end_time = time.time()
    # print("RICE - RP - Best params for NN:",cnn_optimized.best_params_)

    time_train = end_time - start_time
    print(f"Train Time: {time_train}")

    start_time = time.time()
    classifier_accuracy = accuracy_score(y_test, cnn_optimized.predict(X_test))
    end_time = time.time()
    time_infer = end_time - start_time
    print("RICE - RP - Best params best accuracy:", classifier_accuracy)
    print("Time to infer:", time_infer)

    train_sizes_abs, train_results, test_results = learning_curve(cnn_optimized, X_train, y_train,
                                                                  train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    param_range = np.linspace(0.1, 1.0, 10) * 100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                         title="Rice - RP - Neural Network - Learning Curve (Accuracy vs % Trained)",
                         save_name="NN_RP_rice_learning_curve_score.png", x_label="Percent Trained",
                         y_label="Accuracy", parameter_range=param_range)

    y_predictions = cnn_optimized.predict(X_test)
    scoring_accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy Score: {scoring_accuracy}")

    # ########## KMeans Clustering with NN

    start = time.time()
    kmm = KMeans(n_clusters=2, random_state=620)
    kmm.fit(X)
    x_kmeans = kmm.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(x_kmeans, y, test_size=0.2,
                                                                                    random_state=620)

    time_taken = time.time() - start
    print(f"Rice - KNN NN - Training Time: {str(time_taken)}")

    plt.figure()
    plt.hist(kmm.labels_, bins=np.arange(0,3) - 0.5, rwidth=0.75, zorder=2)
    plt.ylabel('Number of Samples')
    plt.xlabel("Cluster Number")
    plt.xticks(np.array([0,1]))
    plt.title('Rice - KMeans NN - Histogram of Frequencies')
    plt.grid()
    plt.show()
    plt.savefig('RICE_NN_KMeans_Histogram.png')

    cnn_optimized = MLPClassifier(random_state=42, max_iter=2000, learning_rate_init=0.001, activation='tanh', hidden_layer_sizes=(18,), alpha=0.0001)
    param_grid = {'alpha': np.logspace(-4, 3, 8), 'learning_rate_init': np.logspace(-5, 1, 7),'activation': ['relu', 'tanh'], 'hidden_layer_sizes': np.arange(2, 20, 2)}
    # Best params for neural network: {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': 18, 'learning_rate_init': 0.001}
    # cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)

    start_time = time.time()
    cnn_optimized.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time
    print(f"Time to train: {time_train}")

    train_sizes_abs, train_results, test_results = learning_curve(cnn_optimized, X_train, y_train,
                                                                  train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    param_range = np.linspace(0.1, 1.0, 10) * 100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                         title="Rice - KMeans - Neural Network - Learning Curve (Accuracy vs % Trained)",
                         save_name="NN_KMeans_rice_learning_curve_score.png", x_label="Percent Trained",
                         y_label="Accuracy", parameter_range=param_range)

    y_predictions = cnn_optimized.predict(X_test)
    scoring_accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy Score: {scoring_accuracy}")

 # ########## EM Clustering with NN

    start = time.time()
    gmm = mixture.GaussianMixture(n_components=2)
    gmm.fit(X)
    x_em = gmm.predict_proba(X)
    X_train, X_test, y_train, y_test = train_test_split(x_em, y, test_size=0.2,
                                                                                    random_state=620)

    # time_taken = time.time() - start
    # print(f"Rice - EM NN - Training Time: {str(time_taken)}")

    plt.figure()
    plt.hist(x_em, bins=np.arange(0,3) - 0.5, rwidth=0.75, zorder=2)
    plt.ylabel('Number of Samples')
    plt.xlabel("Cluster Number")
    plt.xticks(np.array([0,1]))
    plt.title('Rice - EM NN - Histogram of Frequencies')
    plt.grid()
    plt.show()
    plt.savefig('RICE_NN_EM_Histogram.png')

    cnn = MLPClassifier(random_state=620, max_iter=2000)
    cnn_optimized = MLPClassifier(random_state=620, max_iter=2000, learning_rate_init=0.0001, activation='tanh', hidden_layer_sizes=(18,), alpha=0.01)
    param_grid = {'alpha': np.logspace(-4, 3, 8), 'learning_rate_init': np.logspace(-5, 1, 7),'activation': ['relu', 'tanh'], 'hidden_layer_sizes': np.arange(2, 20, 2)}
    # RICE - EM - Best params for NN: {'activation': 'tanh', 'alpha': 100.0, 'hidden_layer_sizes': 18, 'learning_rate_init': 1e-05}

    # cnn_optimized = GridSearchCV(cnn, param_grid=param_grid, cv=5)
    print(cnn_optimized)

    start_time = time.time()
    cnn_optimized.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time
    print(f"Time to train: {time_train}")
    # print("RICE - EM - Best params for NN:",cnn_optimized.best_params_)

    train_sizes_abs, train_results, test_results = learning_curve(cnn_optimized, X_train, y_train,
                                                                  train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    y_predictions = cnn_optimized.predict(X_test)
    scoring_accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy Score: {scoring_accuracy}")

    param_range = np.linspace(0.1, 1.0, 10) * 100
    plot_learning_curves(x_range=param_range, train_results=train_results, test_results=test_results,
                         title="Rice - EM - Neural Network - Learning Curve (Accuracy vs % Trained)",
                         save_name="NN_EM_rice_learning_curve_score.png", x_label="Percent Trained",
                         y_label="Accuracy", parameter_range=param_range)




def plot_validation_curves(train_results, test_results, title, x_label, y_label, save_name, parameter_range, is_log):

    #Find mean
    train_results_mean = np.mean(train_results, axis=1)
    test_results_mean = np.mean(test_results, axis=1)

    plt.figure()
    #Plot Mean
    if is_log:
        plt.semilogx(parameter_range, train_results_mean)
        plt.semilogx(parameter_range, test_results_mean)
    else:
        plt.plot(parameter_range, train_results_mean)
        plt.plot(parameter_range, test_results_mean)

    #Plot STD
    train_results_std = np.std(train_results, axis=1)
    test_results_std = np.std(test_results, axis=1)
    plt.fill_between(parameter_range, train_results_mean-train_results_std, train_results_mean + train_results_std, alpha=0.4)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)

    plt.legend(['Train Results', 'Test Results'])
    plt.title(title, fontsize=10)
    plt.ylim(0.4,1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return

def plot_learning_curves(x_range, train_results, test_results, title, x_label,y_label, save_name, parameter_range):
    #Find mean
    test_results_mean = np.mean(test_results, axis=1)
    train_results_mean = np.mean(train_results, axis=1)

    plt.figure()
    #Plot MEAN
    plt.plot(x_range, train_results_mean)
    plt.plot(parameter_range, test_results_mean)

    #Plot STD
    test_results_std = np.std(test_results, axis=1)
    train_results_std = np.std(train_results, axis=1)
    plt.fill_between(parameter_range, train_results_mean - train_results_std, train_results_mean + train_results_std, alpha=0.4)
    plt.fill_between(parameter_range, test_results_mean - test_results_std, test_results_mean + test_results_std, alpha=0.4)

    plt.legend(['Train Results', 'Test Results'])
    plt.title(title, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0.7,1)
    plt.grid()
    plt.savefig(save_name)
    plt.show()
    return



if __name__ == '__main__':
    main()