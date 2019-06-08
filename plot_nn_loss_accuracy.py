import numpy as np
import pickle
import matplotlib.pyplot as plt

picklefile = open ('./NN_Checkpoints/losses', 'rb')
[losses, accuracyes, losses_test, accuracyes_test] = pickle.load (picklefile)

plt.plot (range (len(losses)), losses, label='Train set')
plt.plot (range (len(losses_test)), losses_test, label='Test set')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig('figure/loss_nn.pdf')
plt.show()

plt.plot (range (len(accuracyes)), accuracyes, label='Train set')
plt.plot (range (len(accuracyes_test)), accuracyes_test, label='Test set')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('figure/accuracy_nn.pdf')
plt.show()
