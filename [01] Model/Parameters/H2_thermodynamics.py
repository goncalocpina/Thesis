from CoolProp.CoolProp import PropsSI

# Inputs
T_Celsius = 80          # Temperature in °C
T = T_Celsius + 273.15  # Temperature in K
p = 10e6                # Pressure in Pa (e.g. 5 MPa)
#Note: 1 MPa = 10 bar = 10^6 Pa


# Hydrogen density [kg/m³]
rho = PropsSI('D', 'T', T, 'P', p, 'Hydrogen')

LHV_MWh_per_kg = 120e6 / 3.6e9  # J/kg → MWh/kg




print("H2 density at", T, "K and", p, "Pa:", rho, "kg/m³")
print("LHV:", LHV_MWh_per_kg, "MWh/kg")
print(10e6)