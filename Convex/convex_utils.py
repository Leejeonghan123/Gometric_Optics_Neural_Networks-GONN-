import tensorflow as tf

R_set = 0.5

class spherical_to_vec(tf.keras.layers.Layer):
    def __init__(self):
        super(spherical_to_vec, self).__init__()        
    def call(self, inputs):
        x, y, theta, psi = tf.split(inputs, 4, axis=1)
        r_x = tf.sin(theta)*tf.cos(psi)       
        r_y = tf.sin(theta)*tf.sin(psi)
        return tf.concat((x, y, r_x, r_y), axis=1)

class prop(tf.keras.layers.Layer):
    def __init__(self):
        super(prop, self).__init__()        
    def call(self, inputs):
        x, y, r_x, r_y, t = tf.split(inputs, 5, axis=1)
        x_prop = x + r_x*t        
        y_prop = y + r_y*t
        return tf.concat((x_prop, y_prop), axis=1)
        
class intersection(tf.keras.layers.Layer):
    def __init__(self, 
        center=[0.,0.], 
        R=R_set, 
        train_center=False, 
        train_R=False,
        y_u=0.2,
        y_l=-0.2
        ):        
        super(intersection, self).__init__()        
        self.center = center
        self.R = R
        self.train_center = train_center
        self.train_R = train_R
        self.y_u = y_u
        self.y_l = y_l
    def build(self,input_shape):
        self.center = tf.Variable(self.center, trainable=self.train_center)
        self.R = tf.Variable(self.R, trainable= self.train_R)
    def call(self, inputs):
        inter_ = inputs - self.center
        inter_ = tf.reduce_sum(tf.square(inter_),axis=-1)
        inter_ = inter_ - tf.square(self.R)                  
        x_pos, y_pos =tf.split(inputs, 2, axis=-1)
        cond1 = tf.math.maximum(-self.y_u+y_pos, 0.)
        cond2 = tf.math.maximum(-y_pos+self.y_l, 0.)
        return tf.square(inter_) + cond1 + cond2
                     
class snell(tf.keras.layers.Layer):
    def __init__(self, n21=1.3, center=[0.,0.]):
        super(snell, self).__init__()
        self.n21 = n21
        self.center = tf.constant(center)
    def call(self, inputs):
        x, y, r_x, r_y, t = tf.split(inputs, 5, axis=1)
        dx = r_x*t        
        dy = r_y*t
        r_vec = tf.concat((r_x, r_y), axis=1)   
        normal = -(tf.concat((x+dx, y+dy),axis=1) - self.center)
        normal = normal / tf.sqrt(tf.reduce_sum(tf.square(normal)))
        cos_theta1 = tf.reduce_sum(tf.multiply(normal, tf.concat((r_x,r_y), axis=1)),axis=-1)
        sin_theta1 = tf.sqrt(1-tf.square(cos_theta1))
        sin_theta2 = (1/self.n21)*sin_theta1
        cos_theta2 = tf.sqrt(1-tf.square(sin_theta2))
        delta_r = ((1/self.n21)-1)*r_vec + (cos_theta2-(1/self.n21)*cos_theta1)*normal

        #cos_theta2 = tf.sqrt(1- (1/self.n21)**2*(1-tf.square(cos_theta1)) )
        #r_vec_refrac= (1/self.n21)*r_vec + ((1/self.n21)*cos_theta1 - cos_theta2)*normal
        return tf.concat((dx,dy,delta_r), axis=1)
    
class ray_sum(tf.keras.layers.Layer):
    def __init__(self, tol=1e-6):
        super(ray_sum, self).__init__()
        self.tol = tol
    def call(self, inputs):
        r_in, r_add, intersection = inputs
        return tf.cond(intersection<self.tol, lambda: tf.add(r_in, r_add), lambda: r_in)
                 