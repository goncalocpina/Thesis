import json
from hashlib import md5
from io import StringIO

import pandas as pd
import requests

# scenario keys (kun je zelf aanpassen)
SOLAR = "solar"
WIND = "wind"

RN_API_URL = "https://www.renewables.ninja/api/data/{source}"
DATASETS = ("merra2", "sarah")


def get_hash(configs):
    params = wind_params(configs) | solar_params(configs)
    return md5(str(params).encode()).hexdigest()


def from_reninja(configs: dict, weather) -> pd.DataFrame:
    api_token = configs["location"]["RN_api_token"]

    full_params = dict()

    if WIND in configs["scenario_key"]:
        full_params["wind"] = wind_params(configs)
        wind_data = rn_query(api_token, "wind", full_params["wind"])
        wind_data.rename(columns={'electricity': 'electricity_wind'}, inplace=True)
        weather.update(wind_data)
    if SOLAR in configs["scenario_key"]:
        full_params["solar"] = solar_params(configs)
        solar_data = rn_query(api_token, "pv", full_params["solar"])
        solar_data.rename(columns={'electricity': 'electricity_solar'}, inplace=True)
        weather.update(solar_data)

    return weather


def generic_params(configs: dict) -> dict:
    location = configs["location"]

    dataset = location["RN_dataset"]
    assert dataset in DATASETS
    date_from = location["start_date"].strftime("%Y-%m-%d")
    date_to = location["end_date"].strftime("%Y-%m-%d")
    lat, lon = [float(f) for f in location["coordinates"]]

    params = dict(
        lat=float(lat),
        lon=float(lon),
        date_from=date_from,
        date_to=date_to,
        dataset=dataset,
        format="json",
        raw=True,
    )
    return params


def solar_params(configs: dict) -> dict:
    params = generic_params(configs)
    solar = configs["solar"]

    p_max = solar["datasheet"]['p_max']
    total_panels = solar["Solar_total_panels"]
    tracking = solar["Solar_panel_tracking"]
    tilt = solar["Solar_panel_tilt"]
    azim = solar["Solar_panel_azimuth"]

    available_power_pct = (100 - sum([
        solar["Solar_shaded_irradiance_percentage"],
        solar["Solar_irradiance_after_reflection_percentage"],
        solar["Solar_irradiance_after_soiling_percentage"],
        solar["Solar_other_loss_percentage"],
    ])) / 100 * solar["Solar_transmission_efficiency_percentage"] / 100
    assert available_power_pct >= 0

    system_loss = 1 - available_power_pct
    assert system_loss >= 0

    params.update(
        capacity=p_max * total_panels / 1000,
        system_loss=system_loss,
        tracking=tracking,
        tilt=tilt,
        azim=azim,
    )

    return params


def wind_params(configs: dict) -> dict:
    params = generic_params(configs)
    wind = configs["wind"]

    params.update(
        capacity=wind["Windturbine_installed_capacity_kwp"],
        height=wind["Windturbine_height"],
        turbine=wind["Windturbine_type"],
    )

    return params


def rn_query(api_token, source, params) -> pd.DataFrame:
    s = requests.session()
    s.headers = {'Authorization': 'Token ' + api_token}
    r = s.get(RN_API_URL.format(source=source), params=params)
    parsed = json.loads(r.text)

    df = pd.read_json(StringIO(json.dumps(parsed['data'])), orient='index')
    df.index = pd.to_datetime(df.index)

    return df


def preview_loaded_data_RN(resource_name: str, data: pd.DataFrame, n_head: int = 5):
    print(f"\n=== {resource_name.upper()} DATA CHECK ===")

    if data.empty:
        print("⚠️  Data is EMPTY!")
        return

    print(f"Shape: {data.shape}")
    print("Columns:", list(data.columns))
    print("First rows:")
    print(data.head(n_head))
