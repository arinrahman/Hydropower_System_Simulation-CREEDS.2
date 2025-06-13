import math
class HydroRelease:

    def __init__(self, inflow, time, stored):
        self.inflow = inflow
        self.time = time
        self.stored = stored

    def releasePrediction(self):
        """
        Predict the energy release based on the time of day, modeled using a Gaussian distribution
        centered at 12 PM (noon). Distribution info link: https://www3.nd.edu/~rwilliam/stats1/x21.pdf
        """
        standardGeneration = 100
        A = self.inflow + self.stored # Peak value at noon
        mu = 12  # Mean of the distribution (noon, 12 PM)
        sigma = 4  # Standard deviation controlling the width of the bell curve
        exponent = math.exp(-0.5 * ((self.time - mu) ** 2) / (sigma ** 2))
        # Calculate the Gaussian distribution value at the given time
        release = A * exponent
        generated = release * standardGeneration
        
        return release, generated