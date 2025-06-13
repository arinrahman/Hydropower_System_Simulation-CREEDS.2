class DayInfo:
    
    def __init__(self, energyDemand, time, panelEfficency, panelAmount, inflow, hydroStored, waterReleased, hydroEnergy, hydroWeight, solarWeight) -> None:
        self.energyDemand = energyDemand
        self.time = time # 0 - 24
        self.panelEfficency = panelEfficency
        self.panelAmount = panelAmount

        self.inflow = inflow
        self.hydroStored = hydroStored
        self.waterReleased = waterReleased
        self.hydroEnergy = hydroEnergy

        self.hydroWeight = hydroWeight
        self.solarWeight = solarWeight
    
    def __lt__(self, other):
        return self.energyDemand < other.energyDemand