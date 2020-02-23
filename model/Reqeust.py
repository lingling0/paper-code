class Request(object):
    def __init__(self, time, source_nodes, request_packages):
        self.time = time
        self.source_nodes = source_nodes
        self.request_packages = request_packages


class RequestsTimeSlot(object):
    def __init__(self, time, requests):
        self.time = time
        self.requests = requests
