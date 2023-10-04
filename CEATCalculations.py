from extractCeatCategories import getCeatWords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
import numpy as np
import pickle
import os
import sys
from prettytable import PrettyTable


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
    embeddingsDict,
    nSample=10000,
    model="bert",
    save=False,
):
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


def writeDataValue(model, data, sentenceLengths):
    table = PrettyTable()
    headers = ["CEAT Type", "Data Value"]
    headers.extend([f"Length: {lenString}" for lenString in sentenceLengths])

    table.field_names = headers

    print(headers)

    for category, value in data.items():
        row = [
            f"{category}\n{value['target']}\n{value['attribute']}",
            "\nCES:\nP-Value:",
        ]
        row.extend(
            [
                f"\n{value[lenString]['CES']}\n{value[lenString]['p']}"
                for lenString in sentenceLengths
            ]
        )
        table.add_row(row)

    # print(table)
    with open(f"./results/{model}_ceat_results.txt", "w") as f:
        f.write(table.get_string())
        f.close()


if __name__ == "__main__":
    categoryDefinition, ceatData = getCeatWords()

    embeddingsMapper = {
        "BanglaBert_Generator": "embeddings_len_%s.pkl",
        "BanglaBert_Discriminator": "embeddings_bbdisc_len_%s.pkl",
        "Muril_Base": "embeddings_murilB_len_%s.pkl",
        "XLM_Roberta_Base": "embeddings_xlmRB_len_%s.pkl",
    }

    sentenceLengths = ["9", "25", "75", "all"]
    seed = 32
    np.random.seed(seed=seed)
    experimentType = "random"
    if len(sys.argv) >= 2 and sys.argv[1] == "-exp" and sys.argv[2] == "fixed":
        experimentType = "fixed"

    print(f"Experiment Type: {experimentType}")

    for model in embeddingsMapper:
        print(f"Model In Use: {model}")
        if experimentType == "fixed":
            np.random.seed(seed=seed)

        embeddingsFileFormat = embeddingsMapper[model]
        data = {}
        """
        data is a container to hold values for each model in the following format:
        {
            sentenceLength: {
                category: {
                    target: [],
                    attribute: [],
                    CES: [],
                    p: []
                }
            }
        }
        """
        for category in categoryDefinition:
            data[category["Category Name"]] = {
                "target": category["target(s)"],
                "attribute": category["attribute(s)"],
            }

        availableLengths = []

        for lenString in sentenceLengths:
            embeddingsFileName = embeddingsFileFormat % lenString
            print(f"Embeddings File Name: {embeddingsFileName}")
            embeddingsFilePath = os.path.join("./embeddings", model, embeddingsFileName)
            if not os.path.exists(embeddingsFilePath):
                print("File not found")
                continue
            availableLengths.append(lenString)
            embeddingsDict = pickle.load(open(embeddingsFilePath, "rb"))
            for testIndex, ceatGroup in enumerate(ceatData):
                # print(categoryDefinition[testIndex]["Category Name"])
                # print("target: ", categoryDefinition[testIndex]["target(s)"])
                # print("attribute: ", categoryDefinition[testIndex]["attribute(s)"])

                effectSizeArray, varianceArray = CEAT_DataGeneration(
                    ceatGroup,
                    embeddingsDict,
                    nSample=10,
                    model=model,
                )
                pes, p_value = CEAT_MetaAnalysis(
                    effectSizeArray, varianceArray, nSample=10
                )

                # print(f"Combined Effect Size: {pes}")
                # print(f"p-value: {p_value}")
                data[categoryDefinition[testIndex]["Category Name"]][lenString] = {
                    "CES": pes,
                    "p": p_value,
                }

        # print(data)
        writeDataValue(model=model, data=data, sentenceLengths=availableLengths)


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
