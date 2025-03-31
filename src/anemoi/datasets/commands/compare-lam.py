# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import os

from anemoi.datasets import open_dataset

from . import Command

RADIUS_EARTH_KM = 6371.0  # Earth's radius in kilometers

LOG = logging.getLogger(__name__)


class HTML_Writer:
    def __init__(self):
        self.html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #e2e2e2;
                }
            </style>
        </head>
        <body>
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Global Mean</th>
                        <th>LAM Mean</th>
                        <th>Mean Diff (%)</th>
                        <th>Global Std</th>
                        <th>LAM Std</th>
                        <th>Std Diff (%)</th>
                        <th>Global Max</th>
                        <th>LAM Max</th>
                        <th>Max Diff (%)</th>
                        <th>Global Min</th>
                        <th>LAM Min</th>
                        <th>Min Diff (%)</th>
                    </tr>
                </thead>
                <tbody>
        """

    def update_table(
        self,
        v1,
        global_mean,
        lam_mean,
        mean_diff,
        global_std,
        lam_std,
        std_diff,
        global_max,
        lam_max,
        max_diff,
        global_min,
        lam_min,
        min_diff,
    ):

        # Determine inline style for HTML
        mean_bg_color = "background-color: #d4edda;" if abs(mean_diff) < 20 else "background-color: #f8d7da;"
        std_bg_color = "background-color: #d4edda;" if abs(std_diff) < 20 else "background-color: #f8d7da;"
        max_bg_color = "background-color: #d4edda;" if abs(max_diff) < 20 else "background-color: #f8d7da;"
        min_bg_color = "background-color: #d4edda;" if abs(min_diff) < 20 else "background-color: #f8d7da;"

        # Add a row to the HTML table with inline styles
        self.html_content += f"""
            <tr>
                <td style="background-color: #f2f2f2;">{v1}</td>
                <td>{global_mean}</td>
                <td>{lam_mean}</td>
                <td style="{mean_bg_color}">{mean_diff}%</td>
                <td>{global_std}</td>
                <td>{lam_std}</td>
                <td style="{std_bg_color}">{std_diff}%</td>
                <td>{global_max}</td>
                <td>{lam_max}</td>
                <td style="{max_bg_color}">{max_diff}%</td>
                <td>{global_min}</td>
                <td>{lam_min}</td>
                <td style="{min_bg_color}">{min_diff}%</td>
            </tr>
        """

    def save_table(self, save_path="stats_table.html"):
        # Close the HTML tags
        self.html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Save the HTML content to a file
        with open(save_path, "w") as f:
            f.write(self.html_content)

        LOG.info(f"\nHTML table saved to: {save_path}")


def plot_coordinates_on_map(lats, lons):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    """
    Plots the given latitude and longitude coordinates on a map using Cartopy and Matplotlib.

    Parameters:
    - lats: List of latitudes
    - lons: List of longitudes
    """

    if len(lats) != len(lons):
        raise ValueError("The length of latitude and longitude lists must be the same.")

    # Create a figure and axis using the PlateCarree projection

    # Define source (PlateCarree) and target (LambertConformal) projections
    target_proj = ccrs.LambertConformal(central_latitude=0, central_longitude=10, standard_parallels=[63.3, 63.3])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={"projection": target_proj})

    # Set the extent of the map based on the transformed coordinates
    margin = 10
    ax.set_extent(
        [min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin], crs=ccrs.PlateCarree()
    )
    # ax.set_extent([-25, 45, 30, 75], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), zorder=1, alpha=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", zorder=1)

    # Plot transformed coordinates
    ax.scatter(lons, lats, color="blue", s=1, edgecolor="b", transform=ccrs.PlateCarree(), alpha=0.3)
    ax.set_title("Latitude and Longitude")
    ax.title.set_size(20)

    # Show the plot
    return fig


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return RADIUS_EARTH_KM * c


def rectangle_area_km2(lat1, lon1, lat2, lon2):
    """Calculate the area of a rectangle given the coordinates of the top left and bottom right corners in
    latitude and longitude.

    Parameters:
    lat1, lon1 - Latitude and longitude of the top-left corner.
    lat2, lon2 - Latitude and longitude of the bottom-right corner.

    Returns:
    Area in square kilometers (km^2).
    """
    # Calculate the height (difference in latitude)
    height_km = haversine(lat1, lon1, lat2, lon1)

    # Calculate the width (difference in longitude)
    width_km = haversine(lat1, lon1, lat1, lon2)

    # Area of the rectangle
    area_km2 = height_km * width_km
    return area_km2


def check_order(vars_1, vars_2):
    for v1, v2 in zip(vars_1, vars_2):
        if v1 != v2 and ((v1 in vars_2) and (v2 in vars_1)):
            return False

    return True


def compute_wighted_diff(s1, s2, round_ndigits):
    return round((s2 - s1) * 100 / s2, ndigits=round_ndigits)


class CompareLAM(Command):
    """Compare statistic of two datasets. \
       This command compares the statistics of each variable in two datasets ONLY in the overlapping area between the two. \
    """

    def add_arguments(self, command_parser):
        command_parser.add_argument("dataset1", help="Path of the global dataset or the largest dataset.")
        command_parser.add_argument("dataset2", help="Path of the LAM dataset or the smallest dataset.")
        command_parser.add_argument(
            "-D", "--number-of-dates", type=int, default=10, help="Number of datapoints (in time) to compare over."
        )
        command_parser.add_argument("-O", "--outpath", default="./", help="Path to output folder.")
        command_parser.add_argument("-R", "--number-of-digits", type=int, default=4, help="Number of digits to keep.")
        command_parser.add_argument(
            "--selected-vars",
            nargs="+",
            default=["10u", "10v", "2d", "2t"],
            help="List of selected variables to use in the script.",
        )
        command_parser.add_argument(
            "--save-plots", action="store_true", help="Toggle to save a picture of the data grid."
        )

    def run(self, args):
        import matplotlib.pyplot as plt
        import numpy as np
        from prettytable import PrettyTable
        from termcolor import colored  # For coloring text in the terminal

        # Unpack args
        date_idx = args.number_of_dates
        round_ndigits = args.number_of_digits
        selected_vars = args.selected_vars
        global_name = args.dataset1
        lam_name = args.dataset2
        date_idx = 10  # "all" or specific index to stop at
        name = f"{global_name}-{lam_name}_{date_idx}"
        save_path = os.path.join(args.outpath, f"comparison_table_{name}.html")

        # Open LAM dataset
        lam_dataset = open_dataset(lam_name, select=selected_vars)
        lam_vars = list(lam_dataset.variables)
        lam_num_grid_points = lam_dataset[0, 0].shape[1]
        lam_area = rectangle_area_km2(
            max(lam_dataset.latitudes),
            max(lam_dataset.longitudes),
            min(lam_dataset.latitudes),
            min(lam_dataset.longitudes),
        )
        l_coords = (
            max(lam_dataset.latitudes),
            min(lam_dataset.longitudes),
            min(lam_dataset.latitudes),
            max(lam_dataset.longitudes),
        )

        if args.save_plots:
            _ = plot_coordinates_on_map(lam_dataset.latitudes, lam_dataset.longitudes)
            plt.savefig(os.path.join(args.outpath, "lam_dataset.png"))

        LOG.info(f"Dataset {lam_name}, has {lam_num_grid_points} grid points. \n")
        LOG.info("LAM (north, west, south, east): ", l_coords)
        LOG.info(f"Point every: {math.sqrt(lam_area / lam_num_grid_points)} km")

        # Open global dataset and cut it
        lam_start = lam_dataset.dates[0]
        lam_end = lam_dataset.dates[-1]
        global_dataset = open_dataset(global_name, start=lam_start, end=lam_end, area=l_coords, select=selected_vars)
        global_vars = list(global_dataset.variables)
        global_num_grid_points = global_dataset[0, 0].shape[1]
        global_area = rectangle_area_km2(
            max(global_dataset.latitudes),
            max(global_dataset.longitudes),
            min(global_dataset.latitudes),
            min(global_dataset.longitudes),
        )
        g_coords = (
            max(global_dataset.latitudes),
            min(global_dataset.longitudes),
            min(global_dataset.latitudes),
            max(global_dataset.longitudes),
        )

        if args.save_plots:
            _ = plot_coordinates_on_map(global_dataset.latitudes, global_dataset.longitudes)
            plt.savefig(os.path.join(args.outpath, "global_dataset.png"))

        LOG.info(f"Dataset {global_name}, has {global_num_grid_points} grid points. \n")
        LOG.info("Global-lam cut (north, west, south, east): ", g_coords)
        LOG.info(f"Point every: {math.sqrt(global_area / global_num_grid_points)} km")

        # Check variable ordering
        same_order = check_order(global_vars, lam_vars)
        LOG.info(f"Lam dataset has the same order of variables as the global dataset: {same_order}")

        LOG.info("\nComparing statistics..")
        table = PrettyTable()
        table.field_names = [
            "Variable",
            "Global Mean",
            "LAM Mean",
            "Mean Diff (%)",
            "Global Std",
            "LAM Std",
            "Std Diff (%)",
            "Global Max",
            "LAM Max",
            "Max Diff (%)",
            "Global Min",
            "LAM Min",
            "Min Diff (%)",
        ]

        # Create a styled HTML table
        html_writer = HTML_Writer()

        for v1, v2 in zip(global_vars, lam_vars):
            assert v1 == v2
            idx = global_vars.index(v1)

            if date_idx == "all":
                lam_mean = lam_dataset.statistics["mean"][idx]
                lam_std = lam_dataset.statistics["stdev"][idx]
                lam_max = lam_dataset.statistics["max"][idx]
                lam_min = lam_dataset.statistics["min"][idx]
                global_mean = global_dataset.statistics["mean"][idx]
                global_std = global_dataset.statistics["stdev"][idx]
                global_max = global_dataset.statistics["max"][idx]
                global_min = global_dataset.statistics["min"][idx]

            else:
                lam_mean = np.nanmean(lam_dataset[:date_idx], axis=(0, 3))[idx][0]
                lam_std = np.nanstd(lam_dataset[:date_idx], axis=(0, 3))[idx][0]
                lam_max = np.nanmax(lam_dataset[:date_idx], axis=(0, 3))[idx][0]
                lam_min = np.nanmin(lam_dataset[:date_idx], axis=(0, 3))[idx][0]
                global_mean = np.nanmean(global_dataset[:date_idx], axis=(0, 3))[idx][0]
                global_std = np.nanstd(global_dataset[:date_idx], axis=(0, 3))[idx][0]
                global_max = np.nanmax(global_dataset[:date_idx], axis=(0, 3))[idx][0]
                global_min = np.nanmin(global_dataset[:date_idx], axis=(0, 3))[idx][0]

            mean_diff = compute_wighted_diff(lam_mean, global_mean, round_ndigits)
            std_diff = compute_wighted_diff(lam_std, global_std, round_ndigits)
            max_diff = compute_wighted_diff(lam_max, global_max, round_ndigits)
            min_diff = compute_wighted_diff(lam_min, global_min, round_ndigits)

            mean_color = "red" if abs(mean_diff) >= 20 else "green"
            std_color = "red" if abs(std_diff) >= 20 else "green"
            max_color = "red" if abs(max_diff) >= 20 else "green"
            min_color = "red" if abs(min_diff) >= 20 else "green"

            table.add_row(
                [
                    v1,
                    round(global_mean, ndigits=round_ndigits),
                    round(lam_mean, ndigits=round_ndigits),
                    colored(f"{mean_diff}%", mean_color),
                    round(global_std, ndigits=round_ndigits),
                    round(lam_std, ndigits=round_ndigits),
                    colored(f"{std_diff}%", std_color),
                    round(global_max, ndigits=round_ndigits),
                    round(lam_max, ndigits=round_ndigits),
                    colored(f"{max_diff}%", max_color),
                    round(global_min, ndigits=round_ndigits),
                    round(lam_min, ndigits=round_ndigits),
                    colored(f"{min_diff}%", min_color),
                ]
            )

            html_writer.update_table(
                v1,
                global_mean,
                lam_mean,
                mean_diff,
                global_std,
                lam_std,
                std_diff,
                global_max,
                lam_max,
                max_diff,
                global_min,
                lam_min,
                min_diff,
            )

        html_writer.save_table(save_path)
        print(table)


command = CompareLAM
