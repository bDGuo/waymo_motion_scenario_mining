import pickle

import numpy as np

num_time_steps = 1 # default value

# carla data feature description
roadgraph_features = {
    'roadgraph_samples/id':
        np.full([20000, 1], -1, np.int64),
    'roadgraph_samples/type':
        np.full([20000, 1], -1, np.int64),
    'roadgraph_samples/valid':
        np.full([20000, 1], -1, np.int64),
    'roadgraph_samples/xyz':
        np.full([20000, 3], -1, np.float32),
}

# Features of other agents.
state_features = {
    'state/id':
        np.full([128,], -1, np.float32),
    'state/type':
        np.full([128,], -1, np.float32),
    'state/length':
        np.full([128, num_time_steps], -1, np.float32),
    'state/width':
        np.full([128, num_time_steps], -1, np.float32),
    'state/valid':
        np.full([128, num_time_steps], -1, np.float32),
    'state/velocity_x':
        np.full([128, num_time_steps], -1, np.float32),
    'state/velocity_y':
        np.full([128, num_time_steps], -1, np.float32),
    'state/bbox_yaw':
        np.full([128, num_time_steps], -1, np.float32),
    'state/vel_yaw':
        np.full([128, num_time_steps], -1, np.float32),
    'state/x':
        np.full([128, num_time_steps], -1, np.float32),
    'state/y':
        np.full([128, num_time_steps], -1, np.float32),
}

traffic_light_features = {
    'traffic_light_state/state':
        np.full([num_time_steps, 16], -1, np.int64),
    'traffic_light_state/valid':
        np.full([num_time_steps, 16], -1, np.int64),
    'traffic_light_state/x':
        np.full([num_time_steps, 16], -1, np.float32),
    'traffic_light_state/y':
        np.full([num_time_steps, 16], -1, np.float32),
    'traffic_light_state/id':
        np.full([num_time_steps, 16], -1, np.int64),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

def parse_carla_data(data_path):
    """
    parse carla simulated traffic scenario data
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        num_time_steps = data['state/length'][0,:].shape[0]
        data.update(traffic_light_features)
        # data['state/length'][0,:] = 4.2
        # data['state/width'][0,:] = 1.8
        # TODO:comment out the following line if you use CARLA data generated later than 22.03.2023
        data['state/bbox_yaw'][0,:] *= np.pi / 180
        data['state/vel_yaw'][0,:] *= np.pi / 180
        data['state/bbox_yaw'][1,:] *= np.pi / 180
        data['state/vel_yaw'][1,:] *= np.pi / 180
        return data

