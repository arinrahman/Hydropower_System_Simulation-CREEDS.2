class Notification:

    def __init__(self, required, hydroGen, solarGen):
        self.required = required
        self.hydroGen = hydroGen
        self.solarGen = solarGen
    
    def evaluate(self, hydroWeight, solarWeight):
        totalGenerated = self.hydroGen + self.solarGen
        # If we meet required we release
        if totalGenerated >= self.required:
            return (True, hydroWeight, solarWeight)
        # Else we redistribute our weights
        if hydroWeight < solarWeight:
            return (False, hydroWeight + .1, solarWeight - .1)
        return (False, hydroWeight - .1, solarWeight + .1)