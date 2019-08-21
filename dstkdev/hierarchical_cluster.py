link=linkage(dd.T, metric="correlation", method="complete")

# Remap distances (1-corr) to resolve small values better
link2=link.copy()
link2[:,2]=np.log10(link2[:,2]+1e-10)+11

distance_threshold=8

fig, ax = plt.subplots(figsize=(20, 80))
denres = dendrogram(
    link2,
    labels=dd.columns,
    orientation="right",
    ax=ax,
    color_threshold=distance_threshold,
    leaf_font_size=8,
    distance_sort=True,
    above_threshold_color="lightgray",
)


# Determine threshold for clustering
sns.distplot(link2[:,2], bins=40);


cols=dd.columns
links=pipe(link2, lambda x:enumerate(x, len(cols)), cfilter(lambda x:x[1][2]<=distance_threshold), dict)

used_idxs=set()
clusters=[]

def cluster(idx):
    used_idxs.add(idx)
    result=set()
    elem1, elem2, score, _ = links[idx]
    for elem in [elem1, elem2]:
        if elem<len(cols):
            result.add(elem)
        else:
            result.update(cluster(elem))
    return result
    

for link_idx in sorted(links.keys(), reverse=True):
    if link_idx in used_idxs:
        continue
        
    clusters.append(cluster(link_idx))
    
clusters=[{cols[x] for x in clus} for clus in clusters]
clusters