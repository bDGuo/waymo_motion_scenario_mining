from abc import ABC
import tensorflow as tf


# This class is deprecated. We will use shapely to compute the area of intersection. 
class rect_interaction(ABC):
    def __init__(self,rect1, rect2):
        '''
        cx,cy: coordinates of center
        '''
        self.r1={'cx':rect1.kinematics['x'],'cy':rect1.kinematics['y'],
                'l':rect1.kinematics['length'], 'w': rect1.kinematics['width'],
                'theta': rect1.kinematics['bbox_yaw'],
                'v_yaw': rect1.kinematics['vel_yaw'],
                'v_x': rect1.kinematics['velocity_x'], 
                'v_y':rect1.kinematics['velocity_y']} #[num_object=1,num_steps]
        
        self.r2={'cx':rect2.kinematics['x'],'cy':rect2.kinematics['y'],
                'l':rect2.kinematics['length'], 'w': rect2.kinematics['width'],
                'theta': rect2.kinematics['bbox_yaw'],
                'v_yaw': rect2.kinematics['vel_yaw'],
                'v_x': rect2.kinematics['velocity_x'], 
                'v_y':rect2.kinematics['velocity_y']} #[num_object=1,num_steps]


    def rect_relation(self,ttc=3,sampling_fq=2):
        '''
        two intersected rectangular             --->    1
        ttc: time-to-collision in seconds, default=3s, with 2 Hz sampling frequency
        else                                     --->    0
        ----------------------------------------------------------------------------
        '''
        # Augmenting center coordinates and heading angle with last sampled v_x, v_y and v_yaw
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
            
    def ttc_esti_center(self,r1,r2,ttc=3,sampling_fq=2):
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
        """
        if anyone of the four vertices of r2 fall in r1, they are intersected
        c1...c4x(y) are the coordinates of four vertices
        crossing test https://blog.csdn.net/s0rose/article/details/78831570
        """
        # rotate r1 and r2 coordinate with heading angle of r1
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

        c1 = tf.logical_and(tf.logical_and(tf.less_equal(x_ll,r2_c1x),tf.less_equal(r2_c1x,x_hl)) \
            ,tf.logical_and(tf.less_equal(y_ll,r2_c1y),tf.less_equal(r2_c1y,y_hl)))
        c2 = tf.logical_and(tf.logical_and(tf.less_equal(x_ll,r2_c2x),tf.less_equal(r2_c2x,x_hl)) \
            ,tf.logical_and(tf.less_equal(y_ll,r2_c2y),tf.less_equal(r2_c2y,y_hl)))
        c3 = tf.logical_and(tf.logical_and(tf.less_equal(x_ll,r2_c3x),tf.less_equal(r2_c3x,x_hl)) \
            ,tf.logical_and(tf.less_equal(y_ll,r2_c3y),tf.less_equal(r2_c3y,y_hl)))
        c4 = tf.logical_and(tf.logical_and(tf.less_equal(x_ll,r2_c4x),tf.less_equal(r2_c4x,x_hl)) \
            ,tf.logical_and(tf.less_equal(y_ll,r2_c4y),tf.less_equal(r2_c4y,y_hl)))
        c = tf.logical_or(tf.logical_or(c1,c2),tf.logical_or(c3,c4))
        return bool(c)

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the coordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = tf.sin(tf.cast(theta, tf.float32)) * (-x) + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)
