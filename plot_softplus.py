import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.linspace(-5, 5), np.log(1 + np.exp(np.linspace(-5, 5))))
plt.show()