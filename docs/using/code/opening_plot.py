import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from anemoi.datasets import open_dataset

ds = open_dataset("aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1.zarr", select="2t")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
p = ax.scatter(x=ds.longitudes, y=ds.latitudes, c=ds[0, 0, 0, :])
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.colorbar(p, label="K", orientation="horizontal")
