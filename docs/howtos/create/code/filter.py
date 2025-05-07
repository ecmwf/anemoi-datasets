from typing import Iterator

import earthkit.data as ekd

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching



@filter_registry.register("w_2_wz", VerticalVelocity)
@filter_registry.register("wz_2_w", VerticalVelocity.reversed)
class VerticalVelocity(MatchingFieldsFilter):
    """A filter to convert vertical wind speed expressed in Pa/s in m/s using the hydrostatic hypothesis,
    and back.
    """

    @matching(
        select="param",
        forward=("w_component", "temperature", "humidity"),
        backward=("wz_component", "temperature", "humidity"),
    )
    def __init__(
        self,
        *,
        w_component="w",
        wz_component="wz",
        temperature="t",
        humidity="q",
    ):

        # wind speed in Pa/s
        self.w_component = w_component
        # wind speed in m/s
        self.wz_component = wz_component
        self.temperature = temperature
        self.humidity = humidity

    def forward_transform(
        self,
        w_component: ekd.Field,
        temperature: ekd.Field,
        humidity: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert vertical wind speed from Pa/s to m/s.

        Parameters
        ----------
        w_component : ekd.Field
            The vertical wind speed in Pa/s.
        temperature : ekd.Field
            The temperature field.
        humidity : ekd.Field
            The humidity field.

        Returns
        -------
        Iterator[ekd.Field]
            The vertical wind speed in m/s.
        """

        level = float(w_component._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        wz = (-1.0 / (rho * 9.80665 + 1e-8)) * w_component.to_numpy()

        yield self.new_field_from_numpy(wz, template=w_component, param=self.wz_component)
        yield temperature
        yield humidity

    def backward_transform(
        self, wz_component: ekd.Field, temperature: ekd.Field, humidity: ekd.Field
    ) -> Iterator[ekd.Field]:
        """Convert vertical wind speed from m/s to Pa/s.

        Parameters
        ----------
        wz_component : ekd.Field
            The vertical wind speed in m/s.
        temperature : ekd.Field
            The temperature field.
        humidity : ekd.Field
            The humidity field.

        Returns
        -------
        Iterator[ekd.Field]
            The vertical wind speed in Pa/s.
        """

        level = float(wz_component._metadata.get("levelist", None))
        # here the pressure gradient is assimilated to volumic weight : hydrostatic hypothesis
        rho = (100 * level) / (287 * temperature.to_numpy() * (1 + 0.61 * humidity.to_numpy()) + 1e-8)
        w = -1.0 * rho * 9.80665 * wz_component.to_numpy()

        yield self.new_field_from_numpy(w, template=wz_component, param=self.w_component)
        yield temperature
        yield humidity
