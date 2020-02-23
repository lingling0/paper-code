class Package(object):
    def __init__(self, id, data, length, start_time, expire_time, sourceNodes, destinationNodes, destinationNodeIds):
        self.id = id
        self.data = data
        self.length = length
        self.start_time = start_time
        self.expire_time = expire_time
        self.sourceNodes = sourceNodes
        self.destinationNodes = destinationNodes
        self.destinationNodeIds = destinationNodeIds

    def __str__(self):
        return "package id: {}, data: {}, length: {}, starttime: {}, expire-time: {}, {}".format(self.id, self.data,
                                                                                                 self.length,
                                                                                                 self.start_time,
                                                                                                 self.expire_time,
                                                                                                 self.destinationNodeIds)


# class Requestpackage(object):
#     def __init__(self, destination_node, package, deadline, wait_time=0):
#         self.destination_node = destination_node
#         self.package = package
#         self.wait_time = wait_time
#         self.deadline = deadline

class DestinationNode(object):
    def __init__(self, destination_node, deadline, wait_time=0):
        self.destination_node = destination_node
        self.wait_time = wait_time
        self.deadline = deadline

    def __str__(self):
        return "deadline: {}, wait_time: {}".format(self.deadline, self.wait_time)
