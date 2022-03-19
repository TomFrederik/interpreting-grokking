import matplotlib.pyplot as plt
import numpy as np

data = np.load('7_related_knn_sorted.npy')

k = 7
topk = np.array([323, 297, 243, 448,  98,  93,  20, 223, 325, 486,  92,  13, 238, 129,
         327, 315, 265, 139, 108, 275, 343, 286, 356, 125,  39, 228, 335, 506,
           1, 460, 501, 186,  46, 220, 303,  60,  94, 305, 368, 433, 268, 405,
          95, 252, 427, 337, 185, 455, 281, 133, 359, 113, 316, 170,  74, 363,
         319, 104, 163, 339, 158, 293, 498, 438,  17, 345, 211, 491, 244, 311])[:k]

top10_data = data[:,:10]

agreement = []

for j, d in enumerate(top10_data):
    agreement.append(np.zeros(len(topk)))
    for i, c in enumerate(topk):
        if c in d:
            agreement[-1][i] += 1
            if c == 20:
                print(j)
agreement = np.array(agreement)
print(agreement.mean(axis=0))

plt.figure()
plt.bar(np.arange(len(topk)), agreement.mean(axis=0))
plt.xticks(np.arange(len(topk)), list(map(lambda x: str(x), topk)))
plt.xlabel('Neuron ID')
plt.ylabel("Share of top-10")
plt.savefig('7_related_knn_sorted.png')

a = [((7*97)*i + 7 * j) % (97**2) for i in range(14) for j in range(14)]
a = list(map(lambda x: (x//97, x%97), a))
print(a[11])
print(a[45])
print(a[75])
print(a[142])