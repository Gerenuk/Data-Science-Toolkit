def calc_feat_directions(feats, pca_model, rotation_angle):
    """
    Used for the PCA mapping to calculate which directions the features would correspond to in the PCA
    """
    result={}
    
    rotation = make_rotation_matrix(rotation_angle)
    
    for i, feat in enumerate(feats):
        vec=np.array([[0]*len(feats)])
        vec[0, i]=1
        feat_vec = pca_model.transform(vec)[0]
        
        result[feat] = np.array(feat_vec) @ rotation
        
    return result


def plot_feat_dir(ax, feat_directions, feat_dir_scaling=1):
    """
    Plot feature directions of PCA onto the PCA plot
    """
    for feat, feat_dir in feat_directions.items():
        feat_dir = feat_dir * feat_dir_scaling

        arrow_color = "k"
        if "[mm]" in feat:
            arrow_color = "green"

        ax.arrow(
            0, 0, feat_dir[0], feat_dir[1], color=arrow_color, head_width=0.03, zorder=11
        )
        ax.text(feat_dir[0], feat_dir[1], feat, zorder=11)
        
        
def make_rotation_matrix(rotation_angle):
    """
    Returns a rotation matrix for some angle. Mainly used to rotate the PCA dimension such that Power points exactly up.
    """
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c, -s), (s, c)))

    return rotation
    
    
def add_pca(dd, pca_feats, min_power=30, rotation_angle=55):
    """
    Add a PCA column to the data
    """
    pca_model = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=123))

    pca_model.fit(dd.loc[dd[power_feat] > min_power, pca_feats])

    mapped = pca_model.transform(dd[pca_feats])

    rotation = make_rotation_matrix(rotation_angle)

    mapped_rot = mapped @ rotation

    dd["pca1"] = mapped_rot[:, 0]
    dd["pca2"] = mapped_rot[:, 1]

    return pca_model


def plot_pca_map(dd_plot):
    """
    Function to plot the map of PCA
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    
    dd_plot["pca1-smooth"] = dd_plot["pca1"].rolling(10).mean()
    dd_plot["pca2-smooth"] = dd_plot["pca2"].rolling(10).mean()

    plot_traj(ax, "pca1-smooth", "pca2-smooth", dd_plot)

    #plot_feat_dir(ax, feat_directions)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    
    ax.add_artist(mpl.patches.Polygon([(-4, 2.5), (4, 2.5), (2, -2), (4, -5), (1, -5)], facecolor="aliceblue"))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 3)
    
    ax.set_xlabel("Mechanical values")
    ax.set_ylabel("Load-related values")
                 
    return ax