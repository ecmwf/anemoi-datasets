# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from ..source import Source
from . import source_registry


LOG = logging.getLogger(__name__)

@source_registry.register("dwd_synop")
class DWDSYNOPSource(Source):
    """A source that reads the synop part from a feedback file 
        as used by DWD (Deutscher Wetterdienst)."""
  
    def __init__(
        self,
        context: any,
        path: str,
        columns: list = None,
        *args,
        **kwargs,
    ):
        """Initialize the DWDSYNOPSource. 

        Parameters
        ----------
        context : Any
            The context for the data source.
        filepath : str
            The path to the CSV file.
        columns : list, optional
            The list of columns to read from the CSV file.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)
 
        self.path = path
        self.columns = columns

    def execute(self, dates):
            import dacepy 
            
            # read in feed back file 
            feedback_file = dacepy.read_fdbk(self.path)  
            # convert to data frame useing dacepy 
            df = feedback_file.to_body().to_dataframe()

            # sub select columns 
            df = df[
                ["varno",
                 "obs",
                 "level",
                 "time",
                 "lon",
                 "lat",
                 "fr_land",
                 # "statid",
                 "r_state",
                 "sso_stdh",
                 "z_station"
                ]
            ]

            # create dictionary to map varno id to variables namev
            varno_to_var = self.create_varno_dictionaries()

            # create var name column 
            df["vname"] = df["varno"].map(varno_to_var)

            # reshape for each variable have its own column
            df_wide = df.pivot_table( 
                index=[
                    "level",
                    "time",
                    "lon",
                    "lat",
                    "fr_land",
                    # "statid",
                    "r_state",
                    "sso_stdh",
                    "z_station"
                ],
                columns='vname', values='obs').reset_index(
                [
                    "time", "lat", "lon", 
                    "level",
                    "fr_land",
                     # "statid",
                     "r_state",
                     "sso_stdh",
                     "z_station"
                ])

            df_wide = df_wide.rename(
                columns={
                    "time": "date", 
                    "lon": "longitude",
                    "lat": "latitude",
                }
            )
            print(f"{type(df_wide["date"])=}")

            # put datetime, lat and lon in front
            cols_to_move = ['date', 'latitude', 'longitude']
            new_columns = cols_to_move + [col for col in df_wide.columns if col not in cols_to_move]
            df_wide = df_wide[new_columns]

            print(df_wide)
            return df_wide

    def create_varno_dictionaries(self):
            import dacepy.fdbk.tables as ftables
            from dacepy.tables_io import get_table
            
            # create list of varno id's from names
            varno_list = [ftables.get_keys("varno", vname) for vname in self.columns ]
            
            # create dictionary to transform varno to variable shortname
            varno_to_var = {}
            for key, value in zip(varno_list, self.columns):
                varno_to_var[key] = value
            
            # TODO:remove if not needed:  create dict that maps varno to sub dict that holds {key, name, desctiption, units } INFO
            # varno_to_vardict = {} 
            # for element in  get_table("varno"):
            #     for id in varno_list:
            #         if element["key"] == id:
            #            varno_to_vardict[id] = element

            return varno_to_var

      
