# QEC-decoder

A QEC decoding library that provides:
- [stim](https://github.com/quantumlib/stim)-based syndrome sampling from noisy circuits of memory experiments;
- implementation of different variants of the belief propagation decoder, written in Rust with Python bindings;
- implementation of sliding window decoding with various choices of inner decoders;
- a Python wrapper class that turns a `Decoder` object into a `sinter.Decoder` object for fast benchmarking via the [sinter](https://pypi.org/project/sinter/) package;
- toolkits for interactive visualization of the belief propagation decoding process (via [plotly](https://github.com/plotly/plotly.py));
- a [PyTorch](https://pytorch.org/)-based machine learning framework for training decoders with learnable parameters.

This codebase was originally designed to facilitate the search for a lightweight variant of the [RelayBP](https://github.com/trmue/relay) decoder.
The ultimate goal is a fast real-time decoder implementable on FPGA with submicrosecond latency (whose implentation is not in this repository).

### Dependencies and installation

- Python >= 3.8.
- [Install rust](https://www.rust-lang.org/tools/install).
- Download the repository and create a Python virtual environment:
  ```
  git clone https://github.com/caoyingkang/QEC-decoder.git
  cd QEC-decoder
  python3 -m venv .venv
  source .env/bin/activate
  pip install --upgrade pip
  ```
- Install maturin: `pip install maturin`.
- Build the Rust-based Python module in the current virtual environment: `maturin develop --release`.
- (Optional) To install [PyTorch](https://pytorch.org/), which is needed in the folder `/pytorch`, build with the following command: `maturin develop --release --extras=pytorch`.


### Usage examples

- Sample syndrome-observable pairs from repetition code memory experiment under circuit-level noise, and decode the syndromes using BP decoder:
  ```python
  from qecdec import RepetitionCode_Memory
  from qecdec import BPDecoder

  expmt = RepetitionCode_Memory(
      d=5,
      rounds=5,
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
      prep_error_rate=0.01,
      cnot_error_rate=0.01,
  )
  sampler = expmt.circuit.compile_detector_sampler(seed=42)
  syndromes, observables = sampler.sample(shots=10_000, separate_observables=True)

  decoder = BPDecoder(expmt.chkmat, expmt.prior, max_iter=50)
  decoded_errors = decoder.decode_batch(syndromes)
  ```

- Sample syndrome-observable pairs from rotated surface code Z-basis memory experiment under phenomenological noise, and decode the syndromes using a sliding window decoder whose inner decoder is MWPM:
  ```python
  from qecdec import RotatedSurfaceCode_Memory
  from qecdec import SlidingWindow_Decoder

  expmt = RotatedSurfaceCode_Memory(
      d=5,
      rounds=50,
      basis='Z',
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
  )
  sampler = expmt.circuit.compile_detector_sampler(seed=42)
  syndromes, observables = sampler.sample(shots=10_000, separate_observables=True)

  decoder = SlidingWindow_Decoder.from_pcm_prior(
      expmt.chkmat,
      expmt.prior,
      detectors_per_layer=expmt.num_detectors_per_layer,
      window_size=5,
      commit_size=1
  )
  decoder.configure_inner_decoders('MWPM')
  decoded_errors = decoder.decode_batch(syndromes)
  ```

- Use `sinter` to collect the decoding results of DMemBP decoder (with randomly selected memory parameters) for rotated surface code Z-basis memory experiment under phenomenological noise:
  ```python
  import numpy as np
  from qecdec import RotatedSurfaceCode_Memory
  from qecdec import DMemBPDecoder, SinterDecoderWrapper
  import sinter
  import os

  expmt = RotatedSurfaceCode_Memory(
      d=5,
      rounds=50,
      basis='Z',
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
  )

  decoder = DMemBPDecoder(
      expmt.chkmat,
      expmt.prior,
      gamma=np.random.uniform(0, 1, size=(expmt.num_error_mechanisms,)),
      max_iter=50
  )
  sinter_decoder = SinterDecoderWrapper(decoder, expmt.obsmat)
  custom_decoders = {'dmembp': sinter_decoder}

  tasks = [sinter.Task(
      circuit=expmt.circuit,
      decoder='dmembp',
      json_metadata={'d': 5, 'rounds': 5, 'p': 0.01},
  )]

  stats = sinter.collect(
      num_workers=os.cpu_count() - 1,
      max_shots=10_000_000,
      max_errors=100,
      tasks=tasks,
      custom_decoders=custom_decoders,
      print_progress=True,
  )
  ```

- More examples can be found under the folder `/notebooks` and `/pytorch`.
