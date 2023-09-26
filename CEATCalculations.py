from extractCeatCategories import getCeatWords
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
from scipy.stats import norm
import numpy as np
import pickle


def associate(w, A, B):
    return (
        cosine_similarity(w.reshape(1, -1), A).mean()
        - cosine_similarity(w.reshape(1, -1), B).mean()
    )


def difference(X, Y, A, B):
    return np.sum([associate(X[i, :], A, B) for i in range(X.shape[0])]) - np.sum(
        [associate(Y[i, :], A, B) for i in range(Y.shape[0])]
    )


def EffectSize(X, Y, A, B):
    delta_mean = np.mean(
        [associate(X[i, :], A, B) for i in range(X.shape[0])]
    ) - np.mean([associate(Y[i, :], A, B) for i in range(Y.shape[0])])

    XY = np.concatenate((X, Y), axis=0)
    s = [associate(XY[i, :], A, B) for i in range(XY.shape[0])]

    std_dev = np.std(s, ddof=1)
    var = std_dev**2

    return delta_mean / std_dev, var


def CEAT_DataGeneration(
    weatGroup: list,
    embeddingsFileName: str,
    nSample=10000,
    seed=32,
    model="bert",
    experimentType="random",
    save=False,
):
    np.random.seed(seed=seed)
    embeddingsDict = pickle.load(open(embeddingsFileName, "rb"))

    effectSizeArray = np.array([], dtype=np.float32)
    varianceArray = np.array([], dtype=np.float32)

    XSet = weatGroup[0]
    YSet = weatGroup[1]
    ASet = weatGroup[2]
    BSet = weatGroup[3]

    for i in range(nSample):
        X = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in XSet
            ]
        )
        Y = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in YSet
            ]
        )
        A = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in ASet
            ]
        )
        B = np.array(
            [
                embeddingsDict[word][np.random.randint(0, len(embeddingsDict[word]))]
                for word in BSet
            ]
        )

        # print(f"X Shape: {X.shape}")
        # print(f"Y Shape: {Y.shape}")
        # print(f"A Shape: {A.shape}")
        # print(f"B Shape: {B.shape}")
        # print()

        effectSize, variance = EffectSize(X, Y, A, B)

        effectSizeArray = np.append(effectSizeArray, effectSize)
        varianceArray = np.append(varianceArray, variance)

    if save:
        pickle.dump(effectSizeArray, open("es_" + model + ".pickle", "wb"))
        pickle.dump(variance, open("var_" + model + ".pickle", "wb"))

    return effectSizeArray, varianceArray


def CEAT_MetaAnalysis(
    effectSizeArray, V, nSample=10000
):  # effectSizeArray and V are numpy array
    # inverse Variance
    W = 1 / V
    Q = np.sum(W * (effectSizeArray**2)) - (
        (np.sum(W * effectSizeArray) ** 2) / np.sum(W)
    )

    df = nSample - 1

    if Q >= df:
        C = np.sum(W) - np.sum(W**2) / np.sum(W)
        sigma_square_btn = (Q - df) / C
    else:
        sigma_square_btn = 0

    # sigma_square_btn is the between-sample variance
    # V is the in-sample variance
    # v is the weight assigned to each weight, where v = 1/(V + sigma_square_btn)

    v = 1 / (V + sigma_square_btn)

    # calculate the combined effect size
    # CES -> Combined Effect Size
    CES = np.sum(v * effectSizeArray) / np.sum(v)

    # calculate the Standard Error of the CES
    SE_CES = np.sqrt(1 / np.sum(v))

    # calculate the p-value. use scipy.stats.norm.sf -> Survival function
    # Also equivalent to 1 - cdf
    # According to paper, it should be a 2-tailed p value, but the implementation shows single tailed.??
    p_value = norm.sf((CES / SE_CES), loc=0, scale=1)

    return CES, p_value


if __name__ == "__main__":
    categoryDefinition, ceatData = getCeatWords()

    models = ["BanglaBert_Generator"]

    for model in models:
        print(f"Model In Use: {model}")
        for testIndex, ceatGroup in enumerate(ceatData):
            print(categoryDefinition[testIndex]["Category Name"])
            print("target: ", categoryDefinition[testIndex]["target(s)"])
            print("attribute: ", categoryDefinition[testIndex]["attribute(s)"])

            effectSizeArray, varianceArray = CEAT_DataGeneration(
                ceatGroup,
                embeddingsFileName="./embeddings/embeddings.pkl",
                nSample=10,
                seed=32,
                model=model,
                experimentType="random",
                save=False,
            )
            pes, p_value = CEAT_MetaAnalysis(
                effectSizeArray, varianceArray, nSample=10000
            )

            print(f"Combined Effect Size: {pes}")
            print(f"p-value: {p_value}")


# def sample_statistics(X, Y, A, B, num=100):
#     XY = np.concatenate((X, Y), axis=0)

#     def inner_1(XY, A, B):
#         X_test_idx = np.random.choice(XY.shape[0], X.shape[0], replace=False)
#         Y_test_idx = np.setdiff1d(list(range(XY.shape[0])), X_test_idx)
#         X_test = XY[X_test_idx, :]
#         Y_test = XY[Y_test_idx, :]
#         return difference(X_test, Y_test, A, B)

#     s = [inner_1(XY, A, B) for i in range(num)]

#     return np.mean(s), np.std(s, ddof=1)


# def p_value(X, Y, A, B, num=100):
#     m, s = sample_statistics(X, Y, A, B, num)
#     d = difference(X, Y, A, B)
#     p = 1 - scipy.stats.norm.cdf(d, loc=m, scale=s)
#     return p
