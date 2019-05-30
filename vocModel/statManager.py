import vocModel.nntools as nt

class DetectionStatsManager(nt.StatsManager):
    def __init__(self):
        super(DetectionStatsManager, self).__init__()
        
    def init(self):
        super(DetectionStatsManager, self).init()
        self.running_accuracy = 0
    def accumulate(self, loss, x, y, d):
        super(DetectionStatsManager, self).accumulate(loss, x, y, d)
     
    def summarize(self):
        loss = super(DetectionStatsManager, self).summarize()
        accuracy = self.running_accuracy / self.number_update
        return {'loss': loss}
