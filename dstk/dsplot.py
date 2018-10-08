from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def plot_calc_featimp(X, y, feature_names=None, n_jobs=-1, num_max_show=50):
    if feature_names is None:
        try:
            feature_names=X.columns
        except AttributeError:
            raise ValueError("Please provide feature_names= or use data with X.columns")

    clf=RandomForestClassifier(n_estimators=100, max_features="auto", n_jobs=n_jobs)
    clf.fit(X, y)

    ax = plot_featimp(clf, feature_names, num_max_show=num_max_show)
    ax.set_title("Random Forest feature importances")


def plot_featimp(clf, feature_names, ax=None, num_max_show=50):
    if ax is None:
        fig, ax=plt.subplots(figsize=(10, len(feature_names)*0.3))

    feat_imps = sorted(zip(feature_names, clf.feature_importances_), key=itemgetter(1), reverse=True)

    feat_imps = feat_imps[:num_max_show]
    num_show = len(feat_imps)

    yticks=list(range(num_show))
    show_feat_names=[feat_name for feat_name, _ in feat_imps]
    if len(feature_names)>num_max_show:
        yticks.append(-1)
        show_feat_names.append(f"... {len(feature_names)-num_max_show} more")


    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    for i, (feat_name, imp) in enumerate(feat_imps):
        y_pos=num_show - i - 1
        ax.plot([0, imp], [y_pos, y_pos], color=color)
        ax.scatter([imp], [y_pos], color=color)

    ax.set_yticks(yticks)
    ax.set_yticklabels(show_feat_names, fontdict={"weight": "bold"})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid("on", axis="x", linestyle=":")
    ax.set_xticklabels([])
    ax.set_ylim((min(yticks)-0.5, max(yticks)+0.5))

    return ax
