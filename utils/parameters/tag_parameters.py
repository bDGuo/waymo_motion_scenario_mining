import numpy as np

# parameters default setting
# parameter for estimation of the actor approaching a static element
TTC_1 = 3
# parameter for estimation of two actors' interaction
TTC_2 = 5
sampling_frequency = 10

max_acc = [0,0.7,0.2,0.4,0]
a_cruise = [0,0.3,0.1,0.2,0]
delta_v = [0,1,0.2,0.5,0]
actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}


k_cruise = 10
k_h=6
# time_steps=283
# degree of smoothing spline
k=3
# default smoothing factor
# smoothing_factor = time_steps / 2
t_s = 1 / sampling_frequency
kernel = 6

intgr_threshold_turn = 45 / 180 * np.pi
intgr_threshold_swerv = 5 / 180 * np.pi
# sampling_threshold = intgr_threshold_turn / (t_s*time_steps)

bbox_extension = 2 # extend length and width of the bbox by 2 times
lane_key = ['freeway','surface_street','bike_lane']
dashed_road_line_key = ['brokenSingleWhite','brokenSingleYellow','brokenDoubleYellow']
other_object_key = ['cross_walk','speed_bump']

tags_param = {
    "TTC_1":float(TTC_1),
    "TTC_2":float(TTC_2),
    "sampling_frequency":float(sampling_frequency),
    "max_acc":max_acc,
    "a_cruise":a_cruise,
    "delta_v":delta_v,
    "actor_dict":actor_dict,
    "k_h":float(k_h),
    "k":float(k),
    "t_s":float(t_s),
    "kernel":float(kernel),
    "intgr_threshold_turn":intgr_threshold_turn,
    "intgr_threshold_swerv":intgr_threshold_swerv,
    "bbox_extension":bbox_extension,
    "lane_key":lane_key,
    "dashed_road_line_key":dashed_road_line_key,
    "other_object_key":other_object_key
}