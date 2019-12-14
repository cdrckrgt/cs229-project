import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

average_costs = {0: 3.832, 0.1: 3.9903, 0.2: 4.4033, 0.3: 4.51155, 0.4: 4.39586, 0.5: 4.4975, 1: 4.56318, 1.5: 4.583, 2.0: 4.6282, 2.5: 4.58888, 3.0: 4.616}
near_collisions = {0: 0.5315, 0.1: 0.350625, 0.2: 0.0715, 0.3: 0.033, 0.4: 0.01325, 0.5: 0.00575, 1: 0.00475, 1.5: 0.00125, 2.0: 0.005, 2.5: 0.006875, 3.0: 0.002}
tracking_errors = {0: 9.5022, 0.1: 12.91472, 0.2: 18.52953, 0.3: 21.34, 0.4: 27.69675, 0.5: 24.43565, 1: 31.840106, 1.5: 26.02932, 2.0: 31.09662, 2.5: 27.9634, 3.0: 28.683}


plt.plot(average_costs.keys(), average_costs.values(), 'o')
trend = np.polyfit(list(average_costs.keys()), list(average_costs.values()), 2)
poly = np.poly1d(trend)
plt.plot(average_costs.keys(), poly(list(average_costs.keys())))
plt.title(r'$\lambda$ vs. Average Cost')
plt.xlabel(r'$\lambda$')
plt.ylabel('Average Cost')
plt.show()
plt.clf()

plt.plot(tracking_errors.values(), near_collisions.values(), 'o')
trend = np.polyfit(list(tracking_errors.values()), list(near_collisions.values()), 2)
poly = np.poly1d(trend)
plt.plot(sorted(tracking_errors.values()), poly(sorted(list(tracking_errors.values()))))
plt.title('Tracking Error vs. Near Collision Rate')
plt.xlabel('Tracking Error (m)')
plt.ylabel('Near Collision Rate')
plt.show()
plt.clf()

import tikzplotlib
tikzplotlib.save('plots.tex')
