# parameters default setting
# parameter for estimation of the actor approaching a static element
TTC_1 = 5
# parameter for estimation of two actors' interaction
TTC_2 = 9


max_acc = [0,0.7,0.2,0.4,0]
a_cruise = [0,0.3,0.1,0.2,0]
delta_v = [0,1,0.2,0.5,0]
actor_dict = {"vehicle":1,"pedestrian":2,"cyclist":3}
agent_state_dict = {"vehicle":{},"pedestrian":{},"cyclist":{}}
agent_pp_state_list = []


k_h=6
time_steps=91
# degree of smoothing spline
k=3
# default smoothing factor
smoothing_factor = time_steps
t_s = 0.1
kernel = 6
sampling_threshold = 8.72e-2  # 
time_steps = 91
intgr_threshold_turn = sampling_threshold*9 # 8.72e-2*9 = 0.785 rad. = 44.97 deg.
intgr_threshold_swerv = sampling_threshold*0 # 8.72e-2*4 = 0.349 rad. = 19.95 deg.

bbox_extension = 2 # extend length and width of the bbox by 2 times
lane_key = ['freeway','surface_street','bike_lane']
dashed_road_line_key = ['brokenSingleWhite','brokenSingleYellow','brokenDoubleYellow']