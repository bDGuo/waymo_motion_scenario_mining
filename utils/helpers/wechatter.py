from wxauto import WeChat

wx = WeChat()

def wechatter(msg):
    # wx.GetSessionList()
    who = '文件传输助手'
    wx.ChatWith(who,RollTimes=0)
    wx.SendMsg(msg)

def wechatter_2(friends:str,msg:str):
    # wx.GetSessionList()
    wx.ChatWith(friends,RollTimes=0)
    wx.SendMsg(msg)