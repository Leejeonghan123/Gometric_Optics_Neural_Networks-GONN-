import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import newton

class Ray(object):
    def __init__(self, x_0, y_0, theta, wavelength=500):
        self.x_0 = x_0
        self.y_0 = y_0
        self.theta = theta
        self.dt = 0
        self.v_0 = np.array([x_0, y_0])
        self.paths = [[np.copy(self.v_0), np.copy(self.theta)]]
        self.end_ts = [float('Inf')]
        self.k = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.wavelength = wavelength

    def update_after_intersect(self, t_end, new_theta, end_beam=False):
        self.v_0 += self.k * t_end
        self.x_0 = self.v_0[0]
        self.y_0 = self.v_0[1]
        self.update_theta(new_theta)
        next_t = t_end + self.dt
        self.dt = next_t
        self.end_ts[-1] = next_t
        self.paths.append([np.copy(self.v_0), np.copy(self.theta)])
        if not end_beam:
            self.end_ts.append(float('Inf'))
    
    def update_theta(self, new_theta):
        self.theta = new_theta
        self.k = np.array([np.cos(self.theta), np.sin(self.theta)])
        
    def get_xy(self, delta_t):    
        vv = self.v_0 + self.k * delta_t
        return vv[0], vv[1]
    
    def estimate_t(self, xp:float):
        t = (xp - self.v_0[0]) / self.k[0]
        return t

    def render(self, ax: Axes, time_of_flights, color='C0'):
        v_e = self.v_0 + time_of_flights * self.k
        v_for_plots = np.vstack((self.v_0, v_e))
        xs = v_for_plots[:,0]
        ys = v_for_plots[:,1]
        ax.plot(xs, ys, color=color)
    
    def get_k_from_theta(self, theta):
        k = np.array([np.cos(theta), np.sin(theta)])
        return k
    
    def render_all(self, ax, time_of_flights, save_file=None, color_before_refraction='salmon', color_after_refraction='royalblue'):
        prev_t = 0
        for idx in range(len(self.end_ts)):
            v_0, theta = self.paths[idx]
            end_t = self.end_ts[idx]
            k = self.get_k_from_theta(theta)
            if time_of_flights > end_t:
                v_e = v_0 + (end_t - prev_t) * k
            else:
                v_e = v_0 + (time_of_flights - prev_t) * k
            v_for_plots = np.vstack((v_0, v_e))
            if save_file:
                result = np.array([self.paths[0][0],v_0,v_0,v_e]).reshape(-1,2)
                np.savetxt('Convex/True_/'+ save_file, result , delimiter=',')
            xs = v_for_plots[:,0]
            ys = v_for_plots[:,1]
            prev_t = end_t
            if idx == len(self.end_ts) - 1:  
                plot_color = color_before_refraction
            else:  
                plot_color = color_after_refraction
            ax.plot(xs, ys, color=plot_color, linewidth=1.)
        
class SphericalSurface(object):
    def __init__(self, x_0, R=0.5, center=[0.,0.], 
                 record_rays=True, material_nr=1.3, end_beam=False):

        self.x_0 = x_0
        self.x_c = center[0]
        self.y_c = center[1]
        self.R = R
        self.ray_bins = []
        self.record_rays = record_rays
        self.y_min = -R
        self.y_max = R
        self.n_r = material_nr
        self.end_beam = end_beam
        self.new_theta = None
    
    def get_surface_xr(self, y):
        x = self.x_c - np.sqrt(np.square(self.R) - np.square(y-self.y_c))
        return x
    
    def spherical_lens_prime(self, y):
        return -(-y + self.y_c)/np.sqrt(np.square(self.R) - np.square(y - self.y_c))
            
    def ray_param_eq(self, t, theta, x_0, y_0):
        return -t*np.cos(theta) - x_0 + self.x_c - np.sqrt(np.square(self.R) - np.square(t*np.sin(theta) + y_0 - self.y_c))
        
    def ray_param_eq_prime(self, t, theta, x_0, y_0):
        return -np.cos(theta) + (t*np.sin(theta) + y_0 - self.y_c)*np.sin(theta) / np.sqrt(np.square(self.R) - np.square(t*np.sin(theta) + y_0 - self.y_c))
        
    def add_rays_into_bins(self, x, y):
        self.ray_bins.append((x, y))
    
    def get_tangent_vec(self, yp, normalize=True):
        xp_p = self.spherical_lens_prime(yp)
        tangent_vec = np.array([xp_p, 1])
        if normalize:
            tangent_vec = tangent_vec / np.linalg.norm((tangent_vec[0], tangent_vec[1]))
        return tangent_vec
    
    def get_norm_vec(self, yp):
        tangent_vec = self.get_tangent_vec(yp, normalize=True)
        normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])
        return normal_vec
    
    def get_refraction(self, yp, ray: Ray, prev_n=1.):
        r_vec = ray.k
        normal_vec = -self.get_norm_vec(yp)
        cos_I = normal_vec[0] * ray.k[0] + normal_vec[1] * ray.k[1]
        sin_I = np.sqrt(1 - np.square(cos_I))
        sin_Ip = prev_n * sin_I / self.n_r
        cos_Ip = np.sqrt(1 - np.square(sin_Ip))
        delta_r = ((prev_n/self.n_r)-1)*r_vec + (cos_Ip-(prev_n/self.n_r)*cos_I)*normal_vec
        next_r = r_vec + delta_r
        return next_r
    
    def intersect(self, ray: Ray, t_min=0, t_max=10):
        t_min_p_1 = ray.estimate_t(self.get_surface_xr(self.R))
        t_min_p_2 = ray.estimate_t(self.x_0)
        t_min_p = min(t_min_p_1, t_min_p_2)
        t_end = newton(self.ray_param_eq, (t_min_p + t_max) / 2,
                      fprime=self.ray_param_eq_prime,
                      args=(ray.theta, ray.x_0, ray.y_0))
        x_end, y_end = ray.get_xy(t_end)
        if (y_end <= self.y_max) and (y_end >= self.y_min):
            if self.record_rays:
                self.add_rays_into_bins(x_end, y_end)
            next_r = self.get_refraction(y_end, ray)
            self.new_theta = np.arctan2(next_r[1], next_r[0])
            ray.update_after_intersect(t_end, new_theta=self.new_theta,
                                      end_beam=self.end_beam)
        
    def render(self, ax, point_num=1200):
        rs = np.linspace(-self.R, self.R, point_num)
        xs = self.get_surface_xr(rs)
        ax.plot(xs, rs, color='black', linewidth=1.0)
        ax.set_xlabel('z-position', fontsize=13)
        ax.set_ylabel('y-position', fontsize=13)

y_pos = np.linspace(-0.18, 0.18, 5)
x_train = [[-1., y, 0.] for y in y_pos] # x,y,theta      
lens_R = 0.5

new_theta = []

plt.rcParams['font.family'] = 'Times New Roman'
fig,ax = plt.subplots(figsize=(5,4))
asf = SphericalSurface(x_0=0, R = lens_R, material_nr=1.3)
asf.render(ax)
ray_wavelength=500

for i, input in enumerate(x_train):
    ray = Ray(input[0], input[1], input[2], ray_wavelength)
    asf.intersect(ray, t_min=0, t_max=2.5)
    ray.render_all(ax,time_of_flights=2.5, save_file=f'ray_tool_{i}.txt')
    if i != 2:
        new_theta.append(asf.new_theta)
asf.ray_bins.pop(2)
plt.show()

### l2 norm (Accuracy)
np.savetxt('Convex/Exact_intersection_point.txt', asf.ray_bins, delimiter=',')
print('\nSaved Intersection Point')
np.savetxt('Convex/Exact_new_theta.txt', np.array(new_theta).reshape(-1))
print('\nSaved New Theta')
