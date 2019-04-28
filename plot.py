import pickle
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
features = pickle.load(open('features','rb'))
labels = pickle.load(open('labels0','rb'))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
X_2d = tsne.fit_transform(features)
for i, c in zip(range(7), colors):
    plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=str(i))

plt.legend()
plt.savefig('0.pdf')
