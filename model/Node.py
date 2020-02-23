class Node(object):
    def __init__(self, id, x, y, region_id, speed=0, max_capacity=10, through_times=0, max_through_time=4):
        self.id = id
        self.x = x
        self.y = y
        self.region_id = region_id
        self.speed = speed
        self.neighborNode = []
        self.max_capacity = max_capacity
        self.through_times = through_times
        self.max_through_times = max_through_time
        self.cache_data = []
        self.history_cache = {}  # 记录每个时刻缓存的数据{time:[a,b], time2: [c,d]}
        self.cache_time = {}  # 记录缓存的每个数据的次数{a:1, b:2}

class VirtualNode(object):
    def __init__(self, max_capacity=10, through_times=0, max_through_time=4):
        self.neighborNode = []
        self.id = -1
        self.max_capacity = max_capacity
        self.through_times = through_times
        self.max_through_times = max_through_time
        self.cache_time = {}  # 记录缓存的每个数据的次数{a:1, b:2}

    def __str__(self):
        return "%s, %s, %d, %d" % (self.id, self.speed, len(self.neighborNode), self.region_id)
