# Function to identify whether two rectangulars interacted 

from abc import ABC
from cmath import nan
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
from math import cos, pi, sin
from shapely.geometry import LineString,Polygon,Point,MultiPolygon
from shapely.ops import unary_union
import warnings
from logger.logger import *


class rect_object(ABC):
    
    def __init__(self,state):
        self.id = state['id']
        self.type = state['type']
        # [num_object:1,num_timestep:91]
        self.kinematics = {
            'x' : state['x'],
            'y' : state['y'],
        'bbox_yaw': state['bbox_yaw'], #
        'length' : state['length'],
        'width' : state['width'],
        'vel_yaw' : state['vel_yaw'], # The yaw angle of each object's velocity vector at each time step.[google]
        'velocity_x' : state['velocity_x'],
        'velocity_y' : state['velocity_y']
        }
        self.tag = {}
        self.validity = state['validity']
        self.validity_ratio = 1
        self.expanded_multipolygon = []
        self.expanded_polygon = []
        self.expanded_bboxes = []

    def cordinate_rotate(self,x,y,theta):
        '''
        rotate the cordinate system with r1's heading angle as x_ axis

        x' = cos (theta) * x + sin (theta) * y
        y' = -sin(theta) * x + cos (theta) * y
        '''
        x_ = tf.cos(tf.cast(theta, tf.float32)) * x + tf.sin(tf.cast(theta, tf.float32)) * y
        y_ = tf.sin(tf.cast(theta, tf.float32)) * (-x) + tf.cos(tf.cast(theta, tf.float32)) * y
        return (x_,y_)

    def data_preprocessing(self,interp:bool=True,spline:bool=False,moving_average:bool=False):
        # interpolating the invalid data
        # find where data is invalid
        mask = np.where(self.validity.numpy().squeeze()!=1)[0] # [91,]
        valid = np.where(self.validity.numpy().squeeze()==1)[0] # [91,]
        valid_length = len(valid)
        appearance_start = valid[0]
        appearance_end = valid[-1]
        appearance_length = appearance_end-appearance_start+1
        # resize yaw angle in (0,2*pi) counter clockwise
        self.kinematics['bbox_yaw'] = self.__set_angel(self.kinematics['bbox_yaw'])
        self.kinematics['vel_yaw'] = self.__set_angel(self.kinematics['vel_yaw'])
        if len(mask)==0:
            return 1
        validity_proportion = valid_length / appearance_length
        if validity_proportion < 0.5:
            logger.warning(f"Valid data proportion too small. Valid/total={validity_proportion:.2f}<50%.")
        self.validity_ratio = validity_proportion 


        if interp:
            for key in self.kinematics:
                self.kinematics[key] = self.__interpolation(self.kinematics[key],mask,valid)
        if spline:
            pass
        if moving_average:
            pass
        else:
            pass
        return f"{validity_proportion:.2f}"

    def clean_abnormal_velocity(self,data,valid,t_s,max_acc:float=0.7):
        # return [time_steps,]
        temp = data.squeeze()[valid[0]:valid[-1]+1]
        temp_shift = np.insert(temp[:-1],0,0)
        normal_temp = temp.copy()
        abnormal_temp_indice = np.where(np.abs(temp-temp_shift)>(max_acc*t_s))[0][1:]
        for i in abnormal_temp_indice:
            normal_temp[i] = np.sign(temp[i] - temp[i-1]) *max_acc*t_s + normal_temp[i-1]
        data[valid[0]:valid[-1]+1] = normal_temp
        return data

    # this method is moved to data_preprocessing.py as a seperate function
    # A tensor version of splining
    def __Univariate_spline(self,data,valid,k=3,smoothing_factor=None):
        """
        Univariate spline for the noisy data
        ---------------------------------
        Output:[time_steps,]
        """
        temp = data.squeeze().copy()
        if len(valid) <=k :
            return temp,0
        assert len(valid)==len(temp[valid]),f"x and y are in different length."
        univariate_spliner = UnivariateSpline(valid,temp[valid],k=k,s=smoothing_factor)
        time_x = np.arange(len(temp))
        result = univariate_spliner(time_x)
        result = np.array(result,dtype=np.float32)
        result[:valid[0]] = np.nan
        result[valid[-1]+1:] = np.nan
        knots = univariate_spliner.get_knots()
        return tf.convert_to_tensor(result,dtype=tf.float32),knots

    def __simple_moving_average(self,data,mask,valid,kernel_length):
        filtered_data = np.convolve(data[valid[0]:valid[-1]+1],np.ones(kernel_length))\
            [:(valid[-1]-valid[0]+1)] / kernel_length
        # bias correction
        sum_start = 0
        sum_end = 0
        for i in range(kernel_length-1):
            sum_start += filtered_data[i]
            filtered_data[i] = sum_start/ (i+1)
            sum_end += filtered_data[-i]
            filtered_data[-i] = sum_end / (i+1)
        data[valid[0]:valid[-1]+1] = filtered_data.copy()
        data[0,:valid[0]] = np.nan
        data[0,valid[-1]+1:] = np.nan
        return tf.convert_to_tensor(data,dtype=tf.float32)
    
    def __set_angel(self,angle):
        '''
        set the angle in (0,2*pi)
        '''

        return (angle+100*pi) % (2*pi)

    def __interpolation(self,data,mask,valid,VELOCITY:bool=False):
        result = data.numpy().astype(np.float32)
        result = result.squeeze()
        result[mask] = np.nan
        temp_pd = pd.DataFrame(result[valid[0]:valid[-1]+1])
        # by default [np.nan,np.nan,1,np.nan,3,np.nan]-> 
                # [np.nan,np.nan,1,2,3,3]
        temp = temp_pd.interpolate().values.T.squeeze()
        result[valid[0]:valid[-1]+1] = temp
        return tf.convert_to_tensor(result,dtype=tf.float32)
    
    def __instant_polygon(self,x:float,y:float,yaw_angle:float,length:float,width:float):
        """
        actor polygon at every time step
        """
        diagnol = np.sqrt(length**2+width**2)
        polygon_coordinates = [
            (x+length/2,y-width/2),
            (x+length/2,y+width/2),
            (x-length/2,y+width/2),
            (x-length/2,y-width/2)
        ]
        for i in range(4):
            new_x = cos(yaw_angle)*(polygon_coordinates[i][0]-x) - sin(yaw_angle)*(polygon_coordinates[i][1]-y) + x
            new_y = sin(yaw_angle)*(polygon_coordinates[i][0]-x) + cos(yaw_angle)*(polygon_coordinates[i][1]-y) + y
            polygon_coordinates[i] = (new_x,new_y)
        return Polygon(polygon_coordinates)
    
    def get_validity_range(self):
        valid = np.where(self.validity.numpy().squeeze()==1)[0] # [91,]
        return valid[0],valid[-1]

    def polygon_set(self):
        """
        return a set of polygon at every time step
        """
        polygon_set = []
        x_ = self.kinematics['x'].numpy().squeeze()
        y_ = self.kinematics['y'].numpy().squeeze()
        yaw_ = self.kinematics['bbox_yaw'].numpy().squeeze()
        length_ = self.kinematics['length'].numpy().squeeze()
        width_ = self.kinematics['width'].numpy().squeeze()
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                polygon_set.append(Polygon([(0,0),(0,0),(0,0),(0,0)]))
            else:
                polygon_set.append(self.__instant_polygon(x_[i],y_[i],yaw_[i],length_[i],width_[i]))
        return polygon_set

    def expanded_polygon_set(self,TTC:int=3,sampling_fq:int=10):
        self.expanded_multipolygon = []
        expanded_polygon_set = []
        x_ = self.kinematics['x'].numpy().squeeze()
        y_ = self.kinematics['y'].numpy().squeeze()
        yaw_ = self.kinematics['bbox_yaw'].numpy().squeeze()
        length_ = self.kinematics['length'].numpy().squeeze()
        width_ = self.kinematics['width'].numpy().squeeze()
        vx_ = self.kinematics['velocity_x'].numpy().squeeze()
        vy_ = self.kinematics['velocity_y'].numpy().squeeze()
        vyaw_ = self.kinematics['vel_yaw'].numpy().squeeze()
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                expanded_polygon_set.append(Polygon([(0,0),(0,0),(0,0),(0,0)]))
                self.expanded_multipolygon.append([Polygon([(0,0),(0,0),(0,0),(0,0)])])
            else:
                expanded_polygon = []
                expanded_all_polygon = []
                for j in range(1,TTC*sampling_fq):
                    new_x_ = x_[i] + j/sampling_fq * vx_[i]
                    new_y_ = y_[i] + j/sampling_fq * vy_[i]
                    new_yaw_ = yaw_[i] + j/sampling_fq * vyaw_[i]
                    expanded_polygon.append(self.__instant_polygon(new_x_,new_y_,new_yaw_,length_[i],width_[i]))
                    expanded_all_polygon.append(expanded_polygon[j-1])
                # expanded_polygon_set.append(MultiPolygon(expanded_polygon))
                expanded_polygon_set.append(unary_union(expanded_polygon).buffer(0.01))
                self.expanded_multipolygon.append(expanded_all_polygon)
        return expanded_polygon_set

    def expanded_bbox_list(self,expand:float=2.0):
        expanded_bbox_list = []
        x_ = self.kinematics['x'].numpy().squeeze()
        y_ = self.kinematics['y'].numpy().squeeze()
        yaw_ = self.kinematics['bbox_yaw'].numpy().squeeze()
        length_ = self.kinematics['length'].numpy().squeeze()
        width_ = self.kinematics['width'].numpy().squeeze()
        mask_ = np.where(np.isnan(x_))[0]
        for i in range(len(x_)):
            # invalid time step is set to polygon with 0 area
            if i in mask_:
                expanded_bbox_list.append(Polygon([(0,0),(0,0),(0,0),(0,0)]))
                self.expanded_bboxes.append(Polygon([(0,0),(0,0),(0,0),(0,0)]))
            else:
                expanded_bbox = self.__instant_polygon(x_[i],y_[i] ,yaw_[i],expand*length_[i],expand*width_[i])
                expanded_bbox_list.append(expanded_bbox)
                self.expanded_bboxes.append(expanded_bbox)
        return expanded_bbox_list








