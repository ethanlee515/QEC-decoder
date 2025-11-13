import numpy as np
import sinter
from .decoders import Decoder
from .sliding_window_decoders import SlidingWindow_Decoder


class SinterDecoderWrapper(sinter.Decoder):
    """Wrap a Decoder object as a sinter.Decoder object, so that it can be used in sinter.collect.
    """

    def __init__(self, decoder: Decoder | SlidingWindow_Decoder, obsmat: np.ndarray):
        """
        Parameters
        ----------
            decoder : Decoder | SlidingWindow_Decoder
                Decoder to be wrapped. Make sure that `decoder` is serializable by pickle (if not, you might want to customize 
                the `decoder.__getstate__` and `decoder.__setstate__` methods).

            obsmat : np.ndarray
                Observable matrix. Sinter decoders need this because they are meant to output observable predictions instead of 
                estimated errors.
        """
        self.decoder = decoder
        self.obsmat = obsmat

    def compile_decoder_for_dem(self, *, dem):
        """Dummy method to satisfy the interface of sinter.Decoder. You should never call this method."""
        return self

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        """Decode bit-packed syndromes, and return bit-packed observable predictions. This method is meant to be used by sinter.collect. 
        You are not supposed to call this method directly.
        """
        syndrome_batch = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little")[:, :self.decoder.num_checks]
        ehat = self.decoder.decode_batch(syndrome_batch)
        observable_predict = (ehat @ self.obsmat.T) % 2

        bit_packed_observable_predict = np.packbits(
            observable_predict, axis=1, bitorder="little")
        return bit_packed_observable_predict


__all__ = [
    "SinterDecoderWrapper",
]
