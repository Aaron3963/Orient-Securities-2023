from pytdx.hq import TdxHq_API
api = TdxHq_API()
with api.connect('119.147.212.81', 7709):
    data = api.get_security_bars(7, 0, '132026', 0, 240) # 123045 为转债代码 ，240 为获取 240个转债数据
    df = api.to_df(data)
    df=df.sort_values('datetime')