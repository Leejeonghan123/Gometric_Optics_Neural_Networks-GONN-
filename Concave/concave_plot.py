import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Title = ['loss.txt', 'inter.txt', 'theta.txt']

### Epoch vs Loss
total_loss = np.loadtxt('Concave/'+Title[0], delimiter=',')
# plt.figure(figsize=(5,4))
plt.rcParams['font.family'] = 'Times New Roman'
for loss, c in zip(total_loss, ['r', 'g', 'b', 'y']):
    plt.plot(loss, color=c, alpha=0.8, linewidth=2.)
plt.xlim([0, 15])
plt.xticks(list(range(0,30,4)), fontsize=15)
plt.yticks(fontsize=15)
plt.yscale('log')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20, labelpad=-3)
plt.gca().set_yticks([10**(-4*i) for i in range(5)]) 
plt.tight_layout()
plt.savefig('Loss_epoch.jpg', dpi=400)
plt.show()


### L2_norm (intersection_point)
exact_inter = np.loadtxt('Concave/'+ Title[1], delimiter=',')
pred_inter = np.loadtxt('Concave/Exact_intersection_point.txt', delimiter=',')
accuracy_inter = [format(np.linalg.norm(pred-exact,2)/np.linalg.norm(exact,2), '.2e') for pred, exact in zip(pred_inter, exact_inter)]

### L2_norm (new_theta)
exact_theta = np.loadtxt('Concave/'+ Title[2], delimiter=',')
pred_theta = np.loadtxt('Concave/Exact_new_theta.txt', delimiter=',')
accuracy_theta = [format(np.linalg.norm([pred-exact],2)/np.linalg.norm([exact],2), '.2e') for pred, exact in zip(pred_theta, exact_theta)]
print(pred_theta)

data = {
    'Model': ['model1', 'model2', 'model3', 'model4'],
    'L2_norm_intersection_point': accuracy_inter,
    'L2_norm_new_theta': accuracy_theta
}
df = pd.DataFrame(data).set_index('Model')

print(df)


