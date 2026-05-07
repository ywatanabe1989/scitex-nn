"""Channel- and frequency-domain augmentations for SSL training.

These layers perturb a ``(batch, channels, samples)`` tensor along
either the channel axis (``DropoutChannels``, ``SwapChannels``,
``ChannelGainChanger``), the time axis or arbitrary axis
(``AxiswiseDropout``), or in the frequency domain via band split
(``FreqGainChanger``). Each is a plain ``nn.Module`` — drop into any
``nn.Sequential``.
"""

from ._AxiswiseDropout import AxiswiseDropout
from ._ChannelGainChanger import ChannelGainChanger
from ._DropoutChannels import DropoutChannels
from ._FreqGainChanger import FreqGainChanger
from ._SwapChannels import SwapChannels

__all__ = [
    "AxiswiseDropout",
    "ChannelGainChanger",
    "DropoutChannels",
    "FreqGainChanger",
    "SwapChannels",
]
