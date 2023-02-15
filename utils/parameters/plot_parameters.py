font1 = {'family' : 'Times New Roman','weight' : 'normal','size':20}
font2 = {'family' : 'Times New Roman','weight' : 'normal','size':30}
lane_color = {
    'freeway':'k',
    'surface_street':'slategray',
    'bike_lane':'maroon',
    'cross_walk':'lightgray',
    'speed_bump':'darkgoldenrod'
}
actor_color = {
    'host_a':{'color':'r','alpha':0.5},
    'host_e':{'color':'b','alpha':0.3},
    'guest_a':{'color':'yellow','alpha':0.3},
    'guest_e':{'color':'green','alpha':0.1}
}

size_pixels = 1000
lane_key = ['freeway','surface_street','bike_lane']
other_object_key = ['cross_walk','speed_bump']
bbox_extension = 2
# parameter for estimation of the actor approaching a static element
TTC_1 = 5
# parameter for estimation of two actors' interaction
TTC_2 = 9