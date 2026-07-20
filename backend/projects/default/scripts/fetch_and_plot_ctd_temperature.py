import json
import argparse
import datetime
import numpy
import xarray
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add backend to path
sys.path.append('backend')

def main():
    """
    Fetch the last 2 days of telemetered CTD data for sensor 01-CTDPFL104 at RS01SBPD,
    clean the data, and plot the temperature.
    """
    # Define the target sensor and method
    sensor = "01-CTDPFL104"
    method = "recovered_wfp"
    stream = "dpc_ctd_instrument_recovered"

    # Define the end time as the current time
    end_time = datetime.now()

    # Create the start time (2 days prior)
    start_time = end_time - datetime.timedelta(days=2)

    # Build the M2M API request parameters
    params = {
        "sensor": sensor,
        "method": method,
        "stream": stream,
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "length": 1200000
    }

    # Construct the M2M API request URL
    url = f"https://m2m.ooi.org/api/v1/data?sensor={sensor}&method={method}&stream={stream}&start_time={params['start_time']}&end_time={params['end_time']}&length={params['length']}")

    try:
        # Make the M2M API request
        response = requests.get(url, headers=headers).json()
        
        # Extract the data from the response
        data = response['data']

        # Create the NetCDF dataset from the raw data
        ds = xarray.open_dataset(
            data['file'],
            encoding={'temperature': {'method': 'NetCDF'}}
        )

        # Clean the data: remove rows where temperature is NaN
        ds = ds.dropna(subset='temperature')

        # Save the cleaned data to a CSV
        df = ds.to_dataframe().reset_index()
        df.to_csv('processed_data/ctd_temperature_cleaned.csv', index=False)

        # Plot the temperature
        plt.figure(figsize=(10,6))
        plt.plot(ds.time, ds.temperature)
        plt.title('CTD Temperature at RS01SBPD (last 2 days)')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        plt.savefig('plots/ctd_temperature.png')

    except Exception as e:
        print(f"Error fetching or plotting data: {e}")

if __name__ == "__main__":
    main()