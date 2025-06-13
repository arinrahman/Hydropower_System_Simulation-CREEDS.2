class Distribution:

    def __init__(self, demand, hydroWeight, solarWeight):
        self.demand = demand
        self.hydroWeight = hydroWeight
        self.solarWeight = solarWeight
    
    def calculateRequired(self):
        hydroReq = self.demand * self.hydroWeight
        solarReq = self.demand * self.solarWeight
        return (hydroReq, solarReq)