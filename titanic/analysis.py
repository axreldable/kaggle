import matplotlib.pyplot as plt
import pandas as pd


def feature_importance(model, feature_names):
    forest_importances = pd.Series(model.feature_importances_, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
