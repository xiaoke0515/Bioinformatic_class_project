import numpy as np
import matplotlib.pyplot as plt
a = np.array([[97.3, 97.3, 96.5], [94.6, 89.9, 90.9]])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.bar([0.7, 1.7, 2.7], a[0] / 100, width=0.3, label='Train')
ax.bar([1.3, 2.3, 3.3], a[1] / 100, width=0.3, label='Test')
ax.set_xlabel('diminsion')
ax.set_ylabel('accuracy')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['50', '20', '10'])
ax.set_ylim(0.85, 1)
plt.legend()
plt.savefig('./figure/accuracy.pdf')
plt.show()
exit()
