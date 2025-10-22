from .decoders import Decoder, MWPMDecoder, UnionFindDecoder, BPDecoder, DMemBPDecoder
from .utils import ceil_div
import numpy as np
from functools import cached_property


class SlidingWindow_Decoder:
    """Sliding window decoder. \n
    Assume that the whole parity check matrix has the following staircase form: \n
    [AB_______________] \n
    [_CAB_____________] \n
    [___CAB___________] \n
    ... \n
    [___________CAB___] \n
    [_____________CAB_] \n
    [_______________CA] \n
    Assume that the whole prior probability vector has the alternating pattern: \n
    [pi pb pa pb pa pb ... pa pb pa pb pf]
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        pa: np.ndarray | None,
        pb: np.ndarray,
        pi: np.ndarray,
        pf: np.ndarray,
        layers: int,
        window_size: int,
        commit_size: int,
    ):
        """
        Parameters
        ----------
            A, B, C : ndarray
                Submatrices that appear in the staircase form of the parity check matrix ∈ {0,1}.

            pa, pb, pi, pf : ndarray
                Subvectors that appear in the alternating pattern of the prior probability vector. If `layers == 2`, then `pa` will be 
                ignored; hence in this case, we can set `pa` to None.

            layers : int
                Number of layers (stairs) in the staircase form of the parity check matrix. Must be at least 2.

            window_size : int
                Number of layers in the decoding window (except for the last one). Must be at least 2 and at most `layers`.

            commit_size : int
                Number of layers in the commit region of each decoding window. Must be at least 1 and at most `window_size // 2`.
        """
        assert isinstance(A, np.ndarray) and A.ndim == 2
        assert isinstance(B, np.ndarray) and B.ndim == 2
        assert isinstance(C, np.ndarray) and C.ndim == 2
        assert isinstance(pb, np.ndarray) and pb.ndim == 1
        assert isinstance(pi, np.ndarray) and pi.ndim == 1
        assert isinstance(pf, np.ndarray) and pf.ndim == 1
        assert A.dtype == B.dtype == C.dtype
        assert pb.dtype == pi.dtype == pf.dtype
        assert A.shape[0] == B.shape[0] == C.shape[0]
        assert A.shape[1] == pi.shape[0] == pf.shape[0]
        assert B.shape[1] == C.shape[1] == pb.shape[0]
        assert 2 <= window_size <= layers
        assert 1 <= commit_size <= window_size // 2
        if layers > 2:
            assert isinstance(pa, np.ndarray) and pa.ndim == 1
            assert pa.dtype == pb.dtype
            assert pa.shape[0] == A.shape[1]

        self.A = A
        self.B = B
        self.C = C
        self.pa = pa
        self.pb = pb
        self.pi = pi
        self.pf = pf
        self.layers = layers
        self.window_size = window_size
        self.commit_size = commit_size
        self.buffer_size = window_size - commit_size

        self.rows_inc: int = A.shape[0]  # increment of rows when moving to the next layer # noqa: E501
        self.rows_tot = layers * self.rows_inc  # total number of rows of the pcm
        self.cols_inc: int = A.shape[1] + B.shape[1]  # increment of columns when moving to the next layer # noqa: E501
        self.cols_tot = layers * self.cols_inc - B.shape[1]  # total number of columns of the pcm # noqa: E501

        # Total number of decoding windows. (If this equals 1, which happens if and only if window_size == layers, then
        # sliding window decoding reduces to standard global decoding.)
        self.num_windows = ceil_div(layers - window_size, commit_size) + 1
        # Number of layers in the last decoding window.
        self.last_window_size = layers - (self.num_windows - 1) * commit_size

        # The pcm of all decoding windows except the last one.
        self.window_pcm = self._build_staircase_block_matrix(
            self.window_size, incl_bottom_right_B_block=True) if self.num_windows > 1 else None
        # The pcm of the last decoding window.
        self.last_window_pcm = self._build_staircase_block_matrix(
            self.last_window_size, incl_bottom_right_B_block=False)

        # The prior probability vector of all decoding windows except the first and the last one.
        self.window_prior = np.concatenate(
            [self.pa, self.pb] * self.window_size
        ) if self.num_windows > 2 else None
        # The prior probability vector of the first decoding window.
        self.first_window_prior = np.concatenate(
            [self.pi, self.pb] +
            [self.pa, self.pb] * (self.window_size - 1)
        ) if self.num_windows > 1 else None
        # The prior probability vector of the last decoding window.
        self.last_window_prior = np.concatenate(
            [self.pa, self.pb] * (self.window_size - 1) +
            [self.pf]
        ) if self.num_windows > 1 else np.concatenate(
            [self.pi, self.pb] +
            [self.pa, self.pb] * (self.layers - 2) +
            [self.pf]
        )

        # The inner decoder for all decoding windows except the first and the last one.
        self.inner_decoder: Decoder = None
        # The inner decoder for the first decoding window.
        self.first_inner_decoder: Decoder = None
        # The inner decoder for the last decoding window.
        self.last_inner_decoder: Decoder = None

    @classmethod
    def from_pcm_prior(
        cls,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        detectors_per_layer: int,
        window_size: int,
        commit_size: int
    ) -> "SlidingWindow_Decoder":
        """
        Alternative constructor that takes the whole parity check matrix and prior probability vector, and check if they have the desired pattern.

        Parameters
        ----------
            pcm : ndarray
                Parity check matrix ∈ {0,1}, shape=(rows_total, cols_total).

            prior : ndarray
                Prior error probabilities for each bit, shape=(cols_total,).

            detectors_per_layer : int
                Number of detectors per layer of the decoding graph. Must be less than and divide the number of rows of pcm.

            window_size : int
                Number of layers in the decoding window (except for the last one).

            commit_size : int
                Number of layers in the commit region of each decoding window.
        """
        assert isinstance(pcm, np.ndarray) and pcm.ndim == 2
        assert isinstance(prior, np.ndarray) and prior.ndim == 1
        assert pcm.shape[1] == prior.shape[0]
        assert pcm.shape[0] > detectors_per_layer
        assert pcm.shape[0] % detectors_per_layer == 0
        layers = pcm.shape[0] // detectors_per_layer
        assert layers >= 2
        cols_inc = np.max(np.nonzero(pcm[:detectors_per_layer, :])[1]) + 1
        cols_A = np.min(np.nonzero(
            pcm[detectors_per_layer:2 * detectors_per_layer, :])[1])
        assert cols_A < cols_inc

        A = pcm[:detectors_per_layer, :cols_A]
        B = pcm[:detectors_per_layer, cols_A:cols_inc]
        C = pcm[detectors_per_layer:2 * detectors_per_layer, cols_A:cols_inc]
        pi = prior[:cols_A]
        pb = prior[cols_A:cols_inc]
        pa = prior[cols_inc:cols_inc + cols_A] if layers > 2 else None
        pf = prior[-cols_A:]
        ret = cls(A, B, C, pa, pb, pi, pf, layers, window_size, commit_size)

        if not np.all(ret.pcm == pcm):
            raise ValueError("`pcm` does not have the desired staircase form.")
        if not np.allclose(ret.prior, prior):
            raise ValueError("`prior` does not have the desired pattern.")
        return ret

    @cached_property
    def pcm(self) -> np.ndarray:
        """
        The total parity check matrix.
        """
        pcm = self._build_staircase_block_matrix(
            self.layers, incl_bottom_right_B_block=False)
        assert pcm.shape == (self.rows_tot, self.cols_tot)
        return pcm

    @cached_property
    def prior(self) -> np.ndarray:
        """
        The total prior probability vector.
        """
        return np.concatenate(
            [self.pi, self.pb] +
            [self.pa, self.pb] * (self.layers - 2) +
            [self.pf]
        )

    def configure_inner_decoders(self, name: str, **kwargs):
        """
        Parameters
        ----------
            name : str
                Name of the inner decoder. Options are "MWPM", "UF", "BP", "DMemBP".

            kwargs : dict
                Keyword arguments for constructing the inner decoder. Peek into the file qecdec/decoders.py for more details.
        """
        if name == "MWPM":
            if self.num_windows > 2:
                self.inner_decoder = MWPMDecoder(
                    self.window_pcm, self.window_prior)
            if self.num_windows > 1:
                self.first_inner_decoder = MWPMDecoder(
                    self.window_pcm, self.first_window_prior)
            self.last_inner_decoder = MWPMDecoder(
                self.last_window_pcm, self.last_window_prior)
        elif name == "UF":
            if self.num_windows > 2:
                self.inner_decoder = UnionFindDecoder(self.window_pcm)
            if self.num_windows > 1:
                self.first_inner_decoder = UnionFindDecoder(self.window_pcm)
            self.last_inner_decoder = UnionFindDecoder(self.last_window_pcm)
        elif name == "BP":
            try:
                max_iter: int = kwargs.pop("max_iter")
            except KeyError:
                print("Missing some of the required arguments for the decoder.")
                return
            if self.num_windows > 2:
                self.inner_decoder = BPDecoder(
                    self.window_pcm, self.window_prior, max_iter=max_iter, **kwargs)
            if self.num_windows > 1:
                self.first_inner_decoder = BPDecoder(
                    self.window_pcm, self.first_window_prior, max_iter=max_iter, **kwargs)
            self.last_inner_decoder = BPDecoder(
                self.last_window_pcm, self.last_window_prior, max_iter=max_iter, **kwargs)
        elif name == "DMemBP":
            try:
                gamma: np.ndarray = kwargs.pop("gamma")
                max_iter: int = kwargs.pop("max_iter")
            except KeyError:
                print("Missing some of the required arguments for the decoder.")
                return
            if self.num_windows > 2:
                self.inner_decoder = DMemBPDecoder(
                    self.window_pcm, self.window_prior,
                    gamma=gamma[:self.window_pcm.shape[1]],
                    max_iter=max_iter,
                    **kwargs)
            if self.num_windows > 1:
                self.first_inner_decoder = DMemBPDecoder(
                    self.window_pcm, self.first_window_prior,
                    gamma=gamma[:self.window_pcm.shape[1]],
                    max_iter=max_iter,
                    **kwargs)
            self.last_inner_decoder = DMemBPDecoder(
                self.last_window_pcm, self.last_window_prior,
                gamma=gamma[:self.last_window_pcm.shape[1]],
                max_iter=max_iter,
                **kwargs)
        else:
            raise ValueError(f"Invalid inner decoder name: {name}")

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome : ndarray
                Vector of syndrome bits ∈ {0,1}, shape=(m,).

        Returns
        -------
            ehat : ndarray
                Decoded error vector ∈ {0,1}, shape=(n,).
        """
        if self.last_inner_decoder is None:
            raise RuntimeError("Inner decoder hasn't been specified.")

        assert isinstance(syndrome, np.ndarray)
        assert syndrome.shape == (self.rows_tot,)

        # remaining syndrome bits to be processed
        remain = syndrome.astype(np.uint8).copy()
        # decoded error components
        ehat_components = []

        window_rows = self.rows_inc * self.window_size
        commit_rows = self.rows_inc * self.commit_size
        commit_cols = self.cols_inc * self.commit_size

        if self.num_windows >= 2:  # first window
            e = self.first_inner_decoder.decode(remain[:window_rows])
            e_committed = e[:commit_cols]
            ehat_components.append(e_committed)
            remain = remain[commit_rows:]
            remain[:self.rows_inc] = (
                remain[:self.rows_inc] +
                self.C @ e_committed[-self.C.shape[1]:]
            ) % 2

        if self.num_windows >= 3:  # middle windows
            for _ in range(self.num_windows - 2):
                e = self.inner_decoder.decode(remain[:window_rows])
                e_committed = e[:commit_cols]
                ehat_components.append(e_committed)
                remain = remain[commit_rows:]
                remain[:self.rows_inc] = (
                    remain[:self.rows_inc] +
                    self.C @ e_committed[-self.C.shape[1]:]
                ) % 2

        e = self.last_inner_decoder.decode(remain)  # last window
        ehat_components.append(e)

        # Aggregate the decoded error components.
        ehat = np.concatenate(ehat_components)
        assert ehat.shape == (self.cols_tot,)

        return ehat

    def decode_batch(self, syndrome_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            syndrome_batch : ndarray
                Array of syndrome vectors ∈ {0,1}, shape=(batch_size,m).

        Returns
        -------
            ehat_batch : ndarray
                Array of decoded error vectors ∈ {0,1}, shape=(batch_size,n).
        """
        if self.last_inner_decoder is None:
            raise RuntimeError("Inner decoder hasn't been specified.")

        assert isinstance(syndrome_batch, np.ndarray)
        assert syndrome_batch.shape[1] == self.rows_tot
        batch_size = syndrome_batch.shape[0]

        # remaining syndrome bits to be processed
        remain = syndrome_batch.astype(np.uint8).copy()
        # decoded error components
        ehat_components = []

        window_rows = self.rows_inc * self.window_size
        commit_rows = self.rows_inc * self.commit_size
        commit_cols = self.cols_inc * self.commit_size

        if self.num_windows >= 2:  # first window
            e = self.first_inner_decoder.decode_batch(remain[:, :window_rows])
            e_committed = e[:, :commit_cols]
            ehat_components.append(e_committed)
            remain = remain[:, commit_rows:]
            remain[:, :self.rows_inc] = (
                remain[:, :self.rows_inc] +
                e_committed[:, -self.C.shape[1]:] @ self.C.T
            ) % 2

        if self.num_windows >= 3:  # middle windows
            for _ in range(self.num_windows - 2):
                e = self.inner_decoder.decode_batch(remain[:, :window_rows])
                e_committed = e[:, :commit_cols]
                ehat_components.append(e_committed)
                remain = remain[:, commit_rows:]
                remain[:, :self.rows_inc] = (
                    remain[:, :self.rows_inc] +
                    e_committed[:, -self.C.shape[1]:] @ self.C.T
                ) % 2

        e = self.last_inner_decoder.decode_batch(remain)  # last window
        ehat_components.append(e)

        # Aggregate the decoded error components.
        ehat_batch = np.hstack(ehat_components)
        assert ehat_batch.shape == (batch_size, self.cols_tot)

        return ehat_batch

    def _build_staircase_block_matrix(self, l: int, incl_bottom_right_B_block: bool) -> np.ndarray:
        """
        Build a block matrix of the staircase form with `l` layers.
        """
        h = self.A.shape[0]
        wa = self.A.shape[1]
        wb = self.B.shape[1]
        if incl_bottom_right_B_block:
            shape = (l * self.rows_inc, l * self.cols_inc)
        else:
            shape = (l * self.rows_inc, l * self.cols_inc - wb)
        M = np.zeros(shape, dtype=self.A.dtype)

        i, j = 0, 0
        for k in range(l):
            if k != 0:
                M[i:i + h, j:j + wb] = self.C
                j += wb
            M[i:i + h, j:j + wa] = self.A
            j += wa
            if k != l - 1:
                M[i:i + h, j:j + wb] = self.B
            if k == l - 1 and incl_bottom_right_B_block:
                M[i:i + h, j:j + wb] = self.B
                j += wb
            i += h
        assert (i, j) == shape

        return M


__all__ = [
    "SlidingWindow_Decoder",
]
