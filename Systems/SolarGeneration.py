import math
class SolarGeneration:

    def __init__(self, panelAmount, panelEfficency, time):
        self.panelAmount = panelAmount
        self.panelEfficency = panelEfficency
        self.time = time
    
    def generatePrediction(self):
        """
        Predict the energy release based on the time of day, modeled using a Gaussian distribution
        centered at 12 PM (noon). Distribution info link: https://www3.nd.edu/~rwilliam/stats1/x21.pdf
        """
        # Parameters for the Gaussian distribution
        standardGeneration = 100
        A = self.panelAmount * self.panelEfficency  * standardGeneration # Peak value at noon
        mu = 12  # Mean of the distribution (noon, 12 PM)
        sigma = 4  # Standard deviation controlling the width of the bell curve
        
        # Calculate the Gaussian distribution value at the given time
        generated = A * math.exp(-0.5 * ((self.time - mu) ** 2) / (sigma ** 2))
        
        return generated