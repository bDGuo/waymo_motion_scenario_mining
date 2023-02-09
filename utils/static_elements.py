
import numpy as np
from typing import Dict,Tuple
from shapely.geometry import LineString,Polygon,Point,MultiPolygon

class StaticElementsWaymo:
    """
    The class is an abstract class for static elements in different datasets.
    to represent all the static elements in 2-d array.
    Including (refered to Waymo Motion, to be filled with other datasets datum:12.10.2022)
    1 - freeway
    2 - surface street
    3 - bike lane
    6 - broken single white
    9 - broken single yellow
    10 - broken double yellow
    18 - Crosswalk
    19 - Speed bump
    others - unknown
    Another tag for the static elements is whether it is controlled by a light or not.
    0 - not controlled by a light
    1 - arrow stop
    2 - arrow caution
    3 - arrow go
    4 - stop
    5 - caution
    6 - go
    7 - flashing stop
    8 - flashing caution
    The other tag for the static elements is the lane id.
    --------------------------------------------------------------
    Init: packed as a dict
    road graph samples:
    sample cordinates, sample lane type, sample lane id
    traffic lights:
    traffic light state, lane id controlled by the traffic light
    --------------------------------------------------------------
    
    """
    def __init__(self,original_data_roadgragh:Dict,original_data_light:Dict) -> None:
        """
        original_data_roadgragh: dict     the original data from the dataset
        {'roadgraph_samples_xyz', 'roadgraph_samples_type', 'roadgraph_samples_lane_id',"roadgraph_dir_xyz"}

        original_data_light: dict         the original data from the dataset
        [time_steps, num_lights]
        {'traffic_light_state', 'traffic_light_lane_id','traffic_light_valid'}

        NOTICE: all time of traffic light data should be concatenated 
        """
        self.original_data_roadgragh = original_data_roadgragh
        self.original_data_light = original_data_light
        self.view_port = {}
        # following are the lane type set in the Waymo Motion Dataset
        self.lane_type = {'freeway':1,'surface_street':2,'bike_lane':3,
        'brokenSingleWhite':6,'brokenSingleYellow':9,'brokenDoubleYellow':10}
        self.lane_width ={'freeway':3.5,'surface_street':3.5,'bike_lane':1.5,'brokenSingleWhite':0.2,'brokenSingleYellow':0.2,'brokenDoubleYellow':0.2}
        self.lane = {'freeway':[],'surface_street':[],'bike_lane':[],'brokenSingleWhite':[],'brokenSingleYellow':[],'brokenDoubleYellow':[]}
        self.lane_id = {'freeway':[],'surface_street':[],'bike_lane':[],'brokenSingleWhite':[],'brokenSingleYellow':[],'brokenDoubleYellow':[]}

        # cross walk have no educated width, since their points are not sampled at 0.5m and they are just arcs of polygons
        self.other_object_type = {'cross_walk':18,'speed_bump':19}
        self.other_object = {'cross_walk':[],'speed_bump':[]}
        self.controlled_lane = {'controlled_lane_polygon':[]}
        self.controlled_lane_id = []
        self.traffic_lights = {}

    def __call__(self):
        pass
        
    def __set_view_port(self)->Tuple:
        """
        get the view port of the scene
        """
        # [num_samples, 1]
        all_y = self.original_data_roadgragh['roadgraph_samples_xyz'][:,1].numpy()
        all_x = self.original_data_roadgragh['roadgraph_samples_xyz'][:,0].numpy()
        center_y = (np.max(all_y) - np.min(all_y)) / 2
        center_x = (np.max(all_x) - np.min(all_x)) / 2
        range_y = np.ptp(all_y)
        range_x = np.ptp(all_x)
        width = max(range_y, range_x)
        self.view_port['center_x'] = center_x
        self.view_port['center_y'] = center_y
        self.view_port['width'] = width
        return center_x,center_y,width
    
    def __get_view_port(self):
        """
        set the view port of the scene
        """
        if 'center_x' in self.view_port and 'center_y' in self.view_port and 'width' in self.view_port:
            return self.view_port['center_x'],self.view_port['center_y'],self.view_port['width']
        else:
            return self.__set_view_port()
    
    def __set_view_edge(self)->Tuple:
        """
        get the view edge of the scene
        """
        center_x,center_y,width = self.__get_view_port()
        self.view_port['left_edge'] = center_x - width / 2
        self.view_port['right_edge'] = center_x + width / 2
        self.view_port['top_edge'] = center_y + width / 2
        self.view_port['bottom_edge'] = center_y - width / 2
        return center_x - width/2, center_x + width/2, center_y - width/2, center_y + width/2

    def __get_view_edge(self):
        """
        set the view edge of the scene
        """
        if 'left_edge' in self.view_port and 'right_edge' in self.view_port and 'top_edge' in self.view_port and 'bottom_edge' in self.view_port:
            return self.view_port['left_edge'],self.view_port['right_edge'],self.view_port['top_edge'],self.view_port['bottom_edge']
        else:
            return self.__set_view_edge()
    
    def get_lane(self,key):
        """
        get the lane of the scene
        """
        return self.lane[key]


    def get_other_object(self,key):
        """
        get the other_object of the scene
        """
        return self.other_object[key]
    

    def get_controlled_lane(self):
        """
        get the controlled_lane of the scene
        """
        return self.controlled_lane['controlled_lane_polygon']
    
    def create_polygon_set(self):
        # [num_points, 3] float32.
        roadgraph_xyz = self.original_data_roadgragh['roadgraph_xyz']
        roadgraph_type = self.original_data_roadgragh['roadgraph_type']
        roadgraph_lane_id = self.original_data_roadgragh['roadgraph_lane_id']
        roadgraph_dir_xyz = self.original_data_roadgragh['roadgraph_dir_xyz']
        traffic_lights_id = self.original_data_light['traffic_lights_id']
        traffic_lights_valid_status = self.original_data_light['traffic_lights_valid']
        controlled_lanes_id = np.unique(traffic_lights_id[traffic_lights_valid_status==1])
        # self.controlled_lanes_id = controlled_lanes_id.tolist()
        # create the lane polygon set
        self.__create_lane_polygon_set(roadgraph_type,roadgraph_xyz,roadgraph_dir_xyz,roadgraph_lane_id,controlled_lanes_id)
        # create the other object polygon set
        self.__create_other_object_polygon_set(roadgraph_type,roadgraph_xyz,roadgraph_dir_xyz)
        # create the traffic light dict
        self.__reducing_traffic_lights_dim()

    def __create_lane_polygon_set(self,roadgraph_type,roadgraph_xyz,roadgraph_dir_xyz,roadgraph_lane_id,controlled_lanes_id):
        for key in self.lane_type:
            lane_mask = np.where(roadgraph_type[:,0]==self.lane_type[key])[0]
            lane_pts = roadgraph_xyz[lane_mask,:2].T
            lane_dir = roadgraph_dir_xyz[lane_mask,:2].T
            lane_id = roadgraph_lane_id[lane_mask]
            # print(f"lane {key} has {lane_pts.shape} points")
            if(len(lane_mask)):
                lane_start = 0
                lane_coordinates = [(lane_pts[0,lane_start],lane_pts[1,lane_start])]
                for i,(pt_x,pt_y,dir_x,dir_y) in enumerate(zip(lane_pts[0,:],lane_pts[1,:],lane_dir[0,:],lane_dir[1,:])):
                    if dir_y==0 and dir_x==0:
                        if len(lane_coordinates)>1:
                            lane_polylines = LineString(lane_coordinates)
                            lane_polygon = Polygon(lane_polylines.buffer(self.lane_width[key]/2))
                            self.lane[key].append(lane_polygon)
                            self.lane_id[key].append(lane_id[i,0])
                            # append controlled lane polygon
                            if lane_id[i,0] in controlled_lanes_id:
                                self.controlled_lane['controlled_lane_polygon'].append(lane_polygon)
                                self.controlled_lane_id.append(lane_id[i,0])
                        else:
                            lane_point = Point(lane_coordinates[0]).buffer(self.lane_width[key]/2)
                            self.lane[key].append(lane_point)
                            self.lane_id[key].append(lane_id[i,0])
                        lane_start = i+1
                        if lane_start == len(lane_pts[0,:]):
                            break
                        else:
                            lane_coordinates = [(lane_pts[0,lane_start],lane_pts[1,lane_start])]
                    for _,(pt_x_2,pt_y_2) in enumerate(zip(lane_pts[0,:],lane_pts[1,:])):
                        if dir_x == 0 and dir_y != 0:
                            if pt_x_2 == pt_x:
                                lane_coordinates.append((pt_x_2,pt_y_2))
                                break
                        elif dir_y == 0 and dir_x != 0:
                            if pt_y_2 == pt_y:
                                lane_coordinates.append((pt_x_2,pt_y_2))
                                break
                        elif dir_x !=0 and dir_y != 0:
                            if np.abs((pt_x_2-pt_x)/dir_x-(pt_y_2-pt_y)/dir_y) < 1e-10:
                                lane_coordinates.append((pt_x_2,pt_y_2))
                                break

    def __create_other_object_polygon_set(self,roadgraph_type,roadgraph_xyz,roadgraph_dir_xyz):

        # line range for finding colinear points
        line_range = 20
        tolerance = 1e-5
        # create the crosswalk and speed bump polygon set
        for key in self.other_object_type:
            object_type_mask = np.where((roadgraph_type[:,0]==self.other_object_type[key]))[0]
            object_type_pts = roadgraph_xyz[object_type_mask,:2].T
            object_type_dir = roadgraph_dir_xyz[object_type_mask,:2].T
            found_pts = []
            if len(object_type_mask):
                # recursively find the polygon points
                # for i,(pt_x,pt_y,dir_x,dir_y) in enumerate(zip(object_type_pts[0,:],object_type_pts[1,:],object_type_dir[0,:],object_type_dir[1,:])):
                #     if i in found_pts:
                #         continue
                #     else:
                #         point_1 = Point(pt_x,pt_y)
                #         result = [point_1]
                #         found_pts = [i]
                #         result,found_pts = self.__found_other_object_polygon(i,pt_x,pt_y,dir_x,dir_y,line_range,object_type_pts[0,i+1],object_type_pts[1,i+1],tolerance,result,found_pts)
                # pass
                polygon_start = 0
                polygon_coordinates =  [(object_type_pts[0,polygon_start],object_type_pts[1,polygon_start])]
                tolerance = 1e-5
                for i,(pt_x,pt_y,dir_x,dir_y) in enumerate(zip(object_type_pts[0,:],object_type_pts[1,:],object_type_dir[0,:],object_type_dir[1,:])):
                    # dir_y==dir_x==0 means the end of the polygon
                    if dir_y==0 and dir_x==0:
                        # plolygon should have at least 3 points
                        if len(polygon_coordinates)>=3:
                            object_polygon = Polygon(polygon_coordinates)
                            self.other_object[key].append(object_polygon)
                        else:
                            pass
                        #check if the next point is the last second point,then break;else continue with a new polygon
                        polygon_start = i+1
                        if polygon_start == len(object_type_pts[0,:]):
                            break
                        else:
                            polygon_coordinates = [(object_type_pts[0,polygon_start],object_type_pts[1,polygon_start])]
                    for j,(pt_x_2,pt_y_2) in enumerate(zip(object_type_pts[0,:],object_type_pts[1,:])):
                        if len(polygon_coordinates)==4:
                            break
                        if i==j:
                            continue
                        if dir_x==0 and dir_y!=0:
                            if pt_x_2==pt_x:
                                polygon_coordinates.append((pt_x_2,pt_y_2))
                                break
                        elif dir_y==0 and dir_x!=0:
                            if pt_y_2==pt_y:
                                polygon_coordinates.append((pt_x_2,pt_y_2))
                                break
                        elif dir_x!=0 and dir_y!=0:
                            if np.abs((pt_x_2-pt_x)/dir_x-(pt_y_2-pt_y)/dir_y) < tolerance:
                                polygon_coordinates.append((pt_x_2,pt_y_2))
                                break

    def __found_other_object_polygon(self,index:int,pt_x:float,pt_y:float,dir_x:float,dir_y:float,object_range:float,pt_x_2:float,pt_y_2:float,tolerance:float,result,found_pts):
        line_1 = LineString([(pt_x-object_range*dir_x,pt_y-object_range*dir_x),(pt_x+dir_x,pt_y+dir_y)])
        point_2 = Point(pt_x_2,pt_y_2)
        if len(result)>=4:
            return result
        else:
            if self.__determine_colinear_points(line_1,point_2,tolerance):
                result.append(point_2)
                found_pts.append(index)
                return result

        pass

    def __determine_colinear_points(self,line_1,point_2,tolerance:float)->bool:
        if line_1.distance(point_2)<tolerance:
            return True
        else:
            return False
        
    def __reducing_traffic_lights_dim(self):
        self.traffic_lights['traffic_lights_state'] = self.original_data_light['traffic_lights_state']
        self.traffic_lights['traffic_lights_lane_id'] = self.original_data_light['traffic_lights_id']

        traffic_lights_x = self.original_data_light['traffic_lights_pos_x']
        traffic_lights_y = self.original_data_light['traffic_lights_pos_y']
        traffic_lights_valid = self.original_data_light['traffic_lights_valid']
        self.traffic_lights['points'] = []
        for traffic_light_x,traffic_light_y,traffic_light_valid in zip(traffic_lights_x.T,traffic_lights_y.T,traffic_lights_valid.T):
            valid = np.where(traffic_light_valid==1)[0]
            if len(valid):
                pos_x = np.average(traffic_light_x[valid])
                pos_y = np.average(traffic_light_y[valid])
                traffic_light_point = Point(pos_x,pos_y)
                self.traffic_lights['points'].append(traffic_light_point)
            else:
                continue
        return 0