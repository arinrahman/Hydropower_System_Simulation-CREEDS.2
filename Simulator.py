# Importing necessary modules
# -------- Version using random data --------
from flask import Flask, render_template, session, request, jsonify
from Reservoir import Reservoir
from Ingestor import Ingestor
import math
import random

app = Flask(__name__)
app.secret_key = "something_random"

# Amnistad Output Data location
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv' 
amnistadInitialLevel = 'DataSetExport-Total Storage.Web-Daily-tcm@08450800-Instantaneous-TCM-20240622210630.csv'

# Level Metrics
reservoirMetrics = Ingestor(amnistadInitialLevel)

# Initial values for reservoir parameters
maxVolume = max(entry['Value'] for entry in reservoirMetrics.data)
currVolume = reservoirMetrics.data[-1]["Value"]

# Inital values for controlled variables
temperature = 79.5 # Found in https://waterdata.ibwc.gov/AQWebportal/Data/DataSet/Chart/Location/08374500/DataSet/Water%20Temp/Field%20Visits/Interval/Latest 
release = 0
inflow = 0
solar = 0

# Creating reservoir
reservoir = Reservoir(maxVolume,currVolume,temperature)

# List to store water level data over time
water_level_data = []
hydro_power_data = []
solar_power_data = []
general_power_data = []

@app.route('/home')
def home():
    water_level = calculate_water_level(reservoir.current_volume,reservoir.max_volume)
    dark_mode = session.get('dark_mode')
    return render_template('index.html', temperature=temperature, release=release, inflow=inflow, water_level=water_level, dark_mode = dark_mode, solar = solar)

@app.route('/metrics')
def metrics():
    dark_mode = session.get('dark_mode')
    return render_template('metrics.html', dark_mode=dark_mode)

@app.route('/toggle-dark-mode', methods=['POST'])
def toggle_dark_mode():
    session['dark_mode'] = not session.get('dark_mode')
    return 'Dark mode toggled successfully'

@app.route('/get_energy_output', methods=['GET'])
def get_energy_output():
    # Get energy output from the reservoir
    energy_output = reservoir.energy_output()
    solar_output = reservoir.energy_generation.get_solar_energy_output()
    hydro_output = reservoir.energy_generation.get_hydro_energy_output()
    return jsonify({'general_power_out':general_power_data, 'hydro_power_out':hydro_power_data, 'solar_power_out': solar_power_data, 'energy_output': energy_output, 'solar_output':solar_output, "hydro_output":hydro_output})

@app.route('/', methods = ['GET','POST'])
def index():
    water_level = calculate_water_level(reservoir.current_volume,reservoir.max_volume)
    dark_mode = session.get('dark_mode')
    return render_template('index.html', temperature=temperature, release=release, inflow=inflow, water_level=water_level, dark_mode = dark_mode, solar=solar)

@app.route('/update', methods=['POST'])
def update():
    global temperature, release, inflow, solar
    if request.method == 'POST':
        release = [entry['Value'] for entry in Ingestor(amnistadRelease).data]
        num_values = len(release)
        temperature = [79.5 for _ in range(num_values)]
        inflow = [random.randint(0, 4100) for _ in range(num_values)]
        solar = [random.randint(0, 100) for _ in range(num_values)] # Adjust based on the % you want to fill of the Reservoir.
        for i in range(num_values):
            reservoir.energy_generation.update_panel_amount(solar[i], maxVolume)
            hydro_power = reservoir.energy_generation.get_hydro_energy_output()
            solar_power = reservoir.energy_generation.get_solar_energy_output()
            energy_output = reservoir.energy_output()
            reservoir.inflow(inflow[i])
            reservoir.release(release[i])
            water_level = calculate_water_level(reservoir.current_volume, reservoir.max_volume)
            water_level_data.append({'time': len(water_level_data), 'water_level': water_level, 'temperature': temperature[i]})
            hydro_power_data.append({'time': len(hydro_power_data), 'hydro_power': hydro_power})
            solar_power_data.append({'time': len(solar_power_data), 'solar_power': solar_power})
            general_power_data.append({'time': len(general_power_data), 'energy_output': energy_output})
        return jsonify({'water_level': water_level})

@app.route('/get_water_level_data', methods=['GET'])
def get_water_level_data():
    return jsonify(water_level_data)

def calculate_water_level(current_volume,max_volume):
    water_level = math.floor((current_volume / max_volume) * 100)
    water_level = max(0, min(100, water_level))
    return water_level

if __name__ == '__main__':
    app.run(debug=True)
