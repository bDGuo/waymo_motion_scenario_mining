# Function to identify whether two rectangulars interacted 
# from cmath import cos
# from selectors import EpollSelector
from abc import ABC
from cmath import nan
from configparser import Interpolation
import tensorflow as tf

import numpy as np
import pandas as pd

class rect_object(ABC):
    
    def __init__(self,state):
        self.id = state['id']
        self.type = state['type']
        self.x = state['x'] # [num_object:1,num_timestep:91]
        self.y = state['y']
        self.z = state['z']
        self.bbox_yaw = state['bbox_yaw']
        self.length = state['length']
        self.width = state['width']
        self.height = state['height']
        self.vel_yaw = state['vel_yaw']
        self.velocity_x = state['velocity_x']
        self.velocity_y = state['velocity_y']
        self.tag = {}
        self.validity = state['validity']

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = - tf.sin(tf.cast(theta, tf.float32)) * x + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)

    def data_preprocessing(self):
        # interpolating the invalid data
        mask = tf.where(tf.squeeze(self.validity)!=1).numpy()
        if len(mask)==0:
            return 0
        self.x = self.interpolation(self.x,mask)
        self.y = self.interpolation(self.y,mask)
        self.bbox_yaw = self.interpolation(self.bbox_yaw,mask)
        self.length = self.interpolation(self.length,mask)
        self.width = self.interpolation(self.width,mask)
        self.vel_yaw = self.interpolation(self.vel_yaw,mask)
        self.velocity_x = self.interpolation(self.velocity_x,mask)
        self.velocity_y = self.interpolation(self.velocity_y,mask)

    
    def interpolation(self,data,mask):
        temp = data.numpy().astype(np.float32)
        temp = temp.squeeze()
        if mask[0]==0:
            value_indice = tf.where(tf.squeeze(self.validity)==1).numpy()[0]
            temp[0]= temp[value_indice]
            mask = np.delete(mask,0)
        mask = mask.squeeze()
        temp[mask] = nan
        temp_pd = pd.DataFrame(temp)
        temp = temp_pd.interpolate().values.T
        return tf.convert_to_tensor(temp)



class rect_interaction(ABC):
    
    def __init__(self,rect1, rect2):
        '''
        cx,cy: cordinates of center
        '''
        self.r1={'cx':rect1.x,'cy':rect1.y,
                'l':rect1.length, 'w': rect1.width,
                'theta': rect1.bbox_yaw,
                'v_yaw': rect1.vel_yaw,
                'v_x': rect1.velocity_x, 
                'v_y':rect1.velocity_y} #[num_object=1,num_steps]
        
        self.r2={'cx':rect2.x,'cy':rect2.y,
                'l':rect2.length, 'w': rect2.width,
                'theta': rect2.bbox_yaw,
                'v_yaw': rect2.vel_yaw,
                'v_x': rect2.velocity_x,
                'v_y':rect2.velocity_y} #[num_object=1,num_steps]


    def rect_relation(self,ttc=3.0,sampling_fq=2):
        '''
        two intersected rectangulars             --->    1
        ttc: time-to-collision in seconds, default=3s, with 2 Hz sampling frequency
        else                                     --->    0
        ----------------------------------------------------------------------------
        VAR:
        
        '''
        # Augmenting center cordinates and heading angle with last sampled v_x, v_y and v_yaw
        if ttc * sampling_fq == 0:
            
            r1 = self.r1
            r2 = self.r2
            return self.rect_intersection(r1,r2) | self.rect_intersection(r2,r1)
        else:
            r1 = self.r1
            r2 = self.r2
            r1_,r2_ = self.ttc_esti_center(r1,r2,ttc,sampling_fq)
            relation = (self.rect_intersection(r1_,r2_) | self.rect_intersection(r2_,r1_))
            return relation

            
    def ttc_esti_center(self,r1,r2,ttc=3.0,sampling_fq=2):


        ttc_cx,ttc_cy,ttc_theta = r1['cx'],r1['cy'],r1['theta']
        r1_l,r1_w = r1['l'], r1['w']

        for i in range(1,ttc*sampling_fq+1):
            ttc_cx = tf.concat([ttc_cx,r1['cx']+i/sampling_fq*r1['v_x']],0)
            ttc_cy = tf.concat([ttc_cy,r1['cy']+i/sampling_fq*r1['v_y']],0)
            ttc_theta = tf.concat([ttc_theta,r1['theta']+i/sampling_fq*r1['v_yaw']],0)
            r1['l'] = tf.concat([r1['l'],r1_l],0)
            r1['w'] = tf.concat([r1['w'],r1_w],0)
        r1['cx'] = ttc_cx
        r1['cy'] = ttc_cy
        r1['theta'] = ttc_theta

        ttc_cx,ttc_cy,ttc_theta = r2['cx'],r2['cy'],r2['theta']
        r2_l,r2_w = r2['l'], r2['w']
        for i in range(1,ttc*sampling_fq+1):
            ttc_cx = tf.concat([ttc_cx,r2['cx']+i/sampling_fq*r2['v_x']],0)
            ttc_cy = tf.concat([ttc_cy,r2['cy']+i/sampling_fq*r2['v_y']],0)
            ttc_theta = tf.concat([ttc_theta,r2['theta']+i/sampling_fq*r2['v_yaw']],0)
            r2['l'] = tf.concat([r2['l'],r2_l],0)
            r2['w'] = tf.concat([r2['w'],r2_w],0)
        r2['cx'] = ttc_cx
        r2['cy'] = ttc_cy
        r2['theta'] = ttc_theta

        return r1, r2


    def rect_intersection(self,r1,r2,ind=0) -> bool:
        '''
        if anyone of the four vertices of r2 fall in r1, they are intersected
        c1...c4x(y) are the cordinates of four vertices
        ''' 
        # rotate r1 and r2 cordinate with heading angle of r1
        (r1_cx, r1_cy) = self.cordinate_rotate(r1['cx'],r1['cy'],r1['theta'])
        (r2_cx, r2_cy) = self.cordinate_rotate(r2['cx'],r2['cy'],r1['theta'])
        r2_theta = r2['theta'] - r1['theta']

        r1_l,r1_w = r1['l'], r1['w']
        r2_l,r2_w = r2['l'], r2['w']

        (r2_c1x,r2_c1y) = self.cordinate_rotate(r2_l/2, -r2_w/2,-r2_theta)
        (r2_c2x,r2_c2y) = self.cordinate_rotate(r2_l/2, r2_w/2,-r2_theta) 
        (r2_c3x,r2_c3y) = self.cordinate_rotate(-r2_l/2, -r2_w/2,-r2_theta) 
        (r2_c4x,r2_c4y) = self.cordinate_rotate(-r2_l/2, r2_w/2,-r2_theta) 
        r2_c1x += r2_cx
        r2_c1y += r2_cy
        r2_c2x += r2_cx
        r2_c2y += r2_cy
        r2_c3x += r2_cx
        r2_c3y += r2_cy
        r2_c4x += r2_cx
        r2_c4y += r2_cy

        
        # ll : lower limit, hl: higher limit
        x_ll = r1_cx - r1_l/2
        x_hl = r1_cx + r1_l/2
        y_ll = r1_cy - r1_w/2
        y_hl = r1_cy + r1_w/2

        c1 = tf.less_equal(x_ll,r2_c1x) & tf.less_equal(r2_c1x,x_hl) \
            & tf.less_equal(y_ll,r2_c1y) & tf.less_equal(r2_c1y,y_hl)
        c2 = tf.less_equal(x_ll,r2_c2x) & tf.less_equal(r2_c2x,x_hl) \
            & tf.less_equal(y_ll,r2_c2y) & tf.less_equal(r2_c2y,y_hl)
        c3 = tf.less_equal(x_ll,r2_c3x) & tf.less_equal(r2_c3x,x_hl) \
            & tf.less_equal(y_ll,r2_c3y) & tf.less_equal(r2_c3y,y_hl)
        c4 = tf.less_equal(x_ll,r2_c4x) & tf.less_equal(r2_c4x,x_hl) \
            & tf.less_equal(y_ll,r2_c4y) & tf.less_equal(r2_c4y,y_hl)
        
        c = np.sum((c1 | c2 | c3 | c4).numpy(),0)
        c = tf.convert_to_tensor(c,dtype=tf.bool)

        return c

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = - tf.sin(tf.cast(theta, tf.float32)) * x + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)

