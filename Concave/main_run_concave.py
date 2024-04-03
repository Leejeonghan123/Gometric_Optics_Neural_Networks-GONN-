import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from concave_utils import *

R_set = 0.5

class Ray_solver:
    def __init__(self, x_train):
        self.x_train = x_train

        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.Ray = self.Single_Ray()
        self.loss = []
        self.theta, self.inter = None, None

    def Single_Ray(self):
        X_in = tf.keras.Input(4) # x,y,theta,psi
        X_ = spherical_to_vec()(X_in)
        hidden1 = tf.keras.layers.Dense(10,
                        activation=tf.keras.activations.get('relu'),
                        kernel_initializer='glorot_normal')(X_)
        hidden2 = tf.keras.layers.Dense(10,
                        activation=tf.keras.activations.get('relu'),
                        kernel_initializer='glorot_normal')(hidden1)
        t_instance = tf.keras.layers.Dense(1)(hidden2)
        prop_in = tf.concat((X_,t_instance),axis=-1)
        prop_out = prop()(prop_in)
        intersection_loss = intersection()(prop_out)
        snell_out = snell()(prop_in)
        ray_out = ray_sum()([X_, snell_out, intersection_loss])
        model = tf.keras.Model(X_in, [t_instance, intersection_loss, ray_out, prop_out])
        return model     

    def train(self, num_epochs=300):
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                tape.watch(self.Ray.trainable_weights)
                pred = self.Ray(self.x_train)
                total_loss = tf.reduce_mean(pred[1])+tf.math.maximum(-pred[3][0][0],0)
                if epoch % 10 == 0:   
                    self.loss.append(total_loss)    
            g = tape.gradient(total_loss, self.Ray.trainable_weights)
            print('Epoch:  Loss: ', epoch, total_loss.numpy())
            self.optim.apply_gradients(zip(g, self.Ray.trainable_weights))
        self.theta = np.arctan2(pred[2][0][2], pred[2][0][3])
        self.inter = [pred[3][0][0], pred[3][0][1]]
        
    def Ray_Plot(self, save_file=None):
        t_pred, intersection_loss, ray_out, _ = self.Ray.predict(self.x_train)
        x_test = tf.concat((spherical_to_vec()(self.x_train), t_pred), axis=-1) 
        intersection_point = prop()(x_test)
        print('Intersection at', intersection_point)
        
        ray2_t = tf.constant([[1.]])
        ray_1 = np.concatenate((self.x_train[:, :2], intersection_point[:, :2]), axis=0)
        ray2_test = tf.concat((ray_out, ray2_t), axis=-1)
        ray2_point = prop()(ray2_test)
        ray_2 = np.concatenate((ray_out[:, :2], ray2_point[:, :2]), axis=0)
    
        self.ray12 = np.concatenate((ray_1, ray_2), axis=0)

        plt.plot(ray_1[:, 0], ray_1[:, 1], 'r')
        plt.plot(ray_2[:, 0], ray_2[:, 1], 'k')
        
        if save_file:  
            np.savetxt('Concave/Pred_/'+ save_file, self.ray12, delimiter=',')
            print(f"Saved ray data to {save_file}")

y_pos = np.linspace(-0.18, 0.18, 5)
x_train = np.array([[-1., y, np.pi/2, 0.] for y in y_pos])

angle = np.linspace(3*np.pi/2., 5*np.pi/2, 100)

x_circle = R_set*np.cos(angle)
y_circle = R_set*np.sin(angle)

plt.figure(figsize=(6,5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.xlabel('z_position', fontsize=13)
plt.ylabel('y_position', fontsize=13)
plt.plot(x_circle,y_circle,'b')

total_loss = []
new_theta = []
intersection_point = []

for i, x_ in enumerate(x_train):
    x_ = np.array([x_])
    ray = Ray_solver(x_)
    ray.train() 
    ray.Ray_Plot(save_file=f'ray_result_{i}.txt')
    total_loss.append(ray.loss) 
    new_theta.append(ray.theta)
    intersection_point.append(ray.inter)
plt.show()

filenames = ['loss', 'theta', 'inter']
data_lists = [total_loss, new_theta, intersection_point]

for name, data in zip(filenames, data_lists):
    np.savetxt(f'Concave/{name}.txt', data, delimiter=',')

