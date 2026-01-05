from sklearn.linear_model import LogisticRegression
from descriptors import *
from sklearn.metrics import ConfusionMatrixDisplay

if __name__ == "__main__":
    imgs, labels = load_image()
    data_hist = histogram(imgs)

    print("Histogram")
    lr = LogisticRegression(random_state=SEED)
    acc, conf, labels = process_train_test(lr, data_hist, labels)
    print(f"Accuracy: {acc}")
    print(f"Confusion matrix: \n{conf}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=lr.classes_)
    disp.plot()
    plt.show()

    print("\nLBP")
    lr = LogisticRegression(random_state=SEED)
    data_lbp = LBP(imgs)
    acc, conf, labels = process_train_test(lr, data_lbp, labels)
    print(f"Accuracy: {acc}")
    print(f"Confusion matrix: \n{conf}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=lr.classes_)
    disp.plot()
    plt.show()

    print("\nFusion")
    lr = LogisticRegression(random_state=SEED)
    fusion = np.hstack((data_hist, data_lbp))
    acc, conf, labels = process_train_test(lr, fusion, labels)
    print(f"Accuracy: {acc}")
    print(f"Confusion matrix: \n{conf}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=lr.classes_)
    disp.plot()

    print("\nGeometric split")
    data_geo = geometric_split(imgs, histogram, nb_split=5)
    acc, conf, labels = process_train_test(lr, data_geo, labels)
    print(f"Accuracy: {acc}")
    print(f"Confusion matrix: \n{conf}")
