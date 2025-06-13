class Solar:

    def __init__(self, panels = 1):
        self.energy_generated = 0
        self.panels = panels
    
    def update_energy(self,temp):
        # Equation = rated power Output * (curent solar radiantion intensity / standard) (1 + temp coefficent of power for pv cell modesl (temp of water - temp of air above water)) area of solar
        # area is rought 17.5 square feet wbeing 65.39 inches / 1.6 square meters
        self.energy_generated += self.calculate_solar_power(5,800,1000,-0.005, temp, temp, 1.6 * self.panels)
    
    def calculate_solar_power(self,rated_power,current_irradiance, standard_irradance, temp_coefficent, water_temp, air_temp, area):
        temp_difference = water_temp - air_temp
        temp_factor = 1 + (temp_coefficent * temp_difference)
        adjusted_power = rated_power + (current_irradiance / standard_irradance)
        return adjusted_power * temp_factor * area
    
    def get_output(self):
        return self.energy_generated
