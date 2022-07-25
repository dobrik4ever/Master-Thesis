from src import Simulator
import matplotlib.pyplot as plt

sim = Simulator(shape=(400,400))

sim.run()

plt.imshow(sim.image); plt.show()