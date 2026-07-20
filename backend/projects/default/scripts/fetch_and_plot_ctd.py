import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

def simulate_download(filename):
    times = pd.date_range(start='2026-02-24', periods=24, freq='H')
    temp = np.array([15.5, 15.7, 16.0, 16.2, 16.5, 16.8, 17.1, 17.5, 17.8, 18.0, 18.2, 18.5, 18.8, 19.0, 19.2, 19.5, 19.8, 20.0, 20.2, 20.5, 20.8, 21.0, 21.2, 21.5])
    ds = xr.Dataset(
        {"temperature": (("time"), temp)},
        coords={"time": times}
    )
    ds.to_netcdf(filename)

def process_and_plot(ds, filename):
    df = ds.to_dataframe().dropna()
    df['time'] = pd.to_datetime(df['time'])
    df.to_csv(f'projects/{project_id}/processed_data/{filename}.csv')
    df.set_index('time')['temperature'].plot(figsize=(10, 6))
    plt.title('Simulated Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.savefig(f'projects/{project_id}/plots/{filename}.png')

if __name__ == "__main__":
    project_id = 'default'
    filename = 'ctd_temperature'
    simulate_download(filename)
    process_and_plot(xr.open_dataset(filename), filename)