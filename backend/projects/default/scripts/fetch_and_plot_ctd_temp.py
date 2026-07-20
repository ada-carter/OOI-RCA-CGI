import sys
os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.config import settings
from datetime import datetime, timedelta

def open_datafiles(files):
    import xarray
    return xarray.open_mfxd(files)

def clean_ctde(ds):
    return ds.where(ds.temp > -2.0, drop=True)

def open_read_and_process(ds):
    import pandas as pd
    df = ds.to_dataframe().reset_index()
    df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
    return df

end_dt = datetime.utcnow()
begin_dt = end_dt - timedelta(days=2)

client = M2MClient(settings.OOI_USERNAME, settings.OOI_TOKEN)
requested_files = client.request_and_download(
    subsite="RS01SBPD",
    node="DP01A",
    sensor="01-CTDPFL104",
    method="recovered_wfp",
    stream="dpc_ctd_instrument_recovered",
    begin_dt=begin_dt.isoformat(),
    end_dt=end_dt.isoformat(),
    dest_dir="projects/default/raw_data"
)

ds = open_datafiles(requested_files)
df = clean_ctde(ds)
df = open_read_and_process(df)

csv_path = "projects/default/processed_data/ctd_temp.csv"
df.to_csv(csv_path, index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['temp'], label='Temperature (°C)')
plt.title('CTD Temperature - Node DP01A')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig('projects/default/plots/ctd_temp_plot.png')