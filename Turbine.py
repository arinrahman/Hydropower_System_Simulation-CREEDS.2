class Turbine:

    def __init__(self):
        self.energy_generated = 0
    
    def update(self,flow):
        # Equation = efficiency coefficent * flow * (h_t,res - h_t,tail) <- I will hard code this last part for now
        self.energy_generated += self.calculate_hydro_power(2,flow,5)

    def calculate_hydro_power(self,energy_coefficent, flow, hydrolics):
        return energy_coefficent * flow * hydrolics

    
    def get_output(self):
        return self.energy_generated