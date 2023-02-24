# parameters default setting
# parameter for estimation of the actor approaching a static element
TTC_1 = 3
# parameter for estimation of two actors' interaction
TTC_2 = 5


max_acc = [0,0.7,0.2,0.4,0]
a_cruise = [0,0.3,0.1,0.2,0]
delta_v = [0,1,0.2,0.5,0]
actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}



k_h=6
time_steps=91
# degree of smoothing spline
k=3
# default smoothing factor
smoothing_factor = time_steps
t_s = 0.1
kernel = 6
sampling_threshold = 8.72e-2  # 0.087 rad. = 4.99 deg.
intgr_threshold_turn = sampling_threshold*9 # 8.72e-2*9 = 0.785 rad. = 44.97 deg.
intgr_threshold_swerv = sampling_threshold*1 # 8.72e-2*1 = 0.087 rad. = 4.99 deg.

bbox_extension = 2 # extend length and width of the bbox by 2 times
lane_key = ['freeway','surface_street','bike_lane']
dashed_road_line_key = ['brokenSingleWhite','brokenSingleYellow','brokenDoubleYellow']
other_object_key = ['cross_walk','speed_bump']

tags_param = {
    "TTC_1":float(TTC_1),
    "TTC_2":float(TTC_2),
    "max_acc":max_acc,
    "a_cruise":a_cruise,
    "delta_v":delta_v,
    "actor_dict":actor_dict,
    "k_h":float(k_h),
    "time_steps":float(time_steps),
    "k":float(k),
    "smoothing_factor":float(smoothing_factor),
    "t_s":float(t_s),
    "kernel":float(kernel),
    "sampling_threshold":sampling_threshold,
    "intgr_threshold_turn":intgr_threshold_turn,
    "intgr_threshold_swerv":intgr_threshold_swerv,
    "bbox_extension":bbox_extension,
    "lane_key":lane_key,
    "dashed_road_line_key":dashed_road_line_key,
    "other_object_key":other_object_key
}