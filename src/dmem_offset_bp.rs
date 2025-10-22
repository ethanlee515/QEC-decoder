use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt};

/// Given a probability `p`, return the log-likelihood ratio `ln((1-p)/p)`.
fn prob_to_llr(p: f64) -> f64 {
    // Clamp the probability to [EPS, 1-EPS] to avoid numerical instability.
    const EPS: f64 = 1e-10;
    let pp = if p < EPS {
        EPS
    } else if p > 1.0 - EPS {
        1.0 - EPS
    } else {
        p
    };
    ((1.0 - pp) / pp).ln()
}

/// Base struct for BP-based decoders.
struct BPBase {
    /// Log-likelihood ratios of the prior error probabilities.
    prior_llr: Array1<f64>,
    /// Number of check nodes (= number of rows of pcm).
    num_chks: usize,
    /// Number of variable nodes (= number of columns of pcm).
    num_vars: usize,
    /// `chk_nbrs[i]` is the list of VNs connected to CN `i` (ordered by VN indices).
    chk_nbrs: Vec<Vec<usize>>,
    /// `var_nbrs[j]` is the list of CNs connected to VN `j` (ordered by CN indices).
    var_nbrs: Vec<Vec<usize>>,
    /// `chk_nbr_pos[i][k]` is the relative position of CN `i` in the list of neighbors of the VN `chk_nbrs[i][k]`.
    /// I.e., if `chk_nbrs[i][k] == j`, then `var_nbrs[j][chk_nbr_pos[i][k]] == i`.
    chk_nbr_pos: Vec<Vec<usize>>,
    /// `var_nbr_pos[j][k]` is the relative position of VN `j` in the list of neighbors of the CN `var_nbrs[j][k]`.
    /// I.e., if `var_nbrs[j][k] == i`, then `chk_nbrs[i][var_nbr_pos[j][k]] == j`.
    var_nbr_pos: Vec<Vec<usize>>,
}

impl BPBase {
    fn new(pcm: ArrayView2<u8>, prior: ArrayView1<f64>) -> Self {
        let num_chks = pcm.nrows();
        let num_vars = pcm.ncols();

        let mut chk_nbrs = vec![Vec::new(); num_chks];
        let mut var_nbrs = vec![Vec::new(); num_vars];
        let mut chk_nbr_pos = vec![Vec::new(); num_chks];
        let mut var_nbr_pos = vec![Vec::new(); num_vars];
        for i in 0..num_chks {
            for j in 0..num_vars {
                if pcm[[i, j]] != 0 {
                    chk_nbr_pos[i].push(var_nbrs[j].len());
                    var_nbr_pos[j].push(chk_nbrs[i].len());
                    chk_nbrs[i].push(j);
                    var_nbrs[j].push(i);
                }
            }
        }
        for i in 0..num_chks {
            assert!(chk_nbrs[i].len() >= 2, "CN {} has less than 2 neighbors", i);
        }
        for j in 0..num_vars {
            assert!(var_nbrs[j].len() >= 1, "VN {} has less than 1 neighbor", j);
        }

        Self {
            prior_llr: prior.mapv(prob_to_llr),
            num_chks: num_chks,
            num_vars: num_vars,
            chk_nbrs: chk_nbrs,
            var_nbrs: var_nbrs,
            chk_nbr_pos: chk_nbr_pos,
            var_nbr_pos: var_nbr_pos,
        }
    }
}

/// Belief Propagation decoder (min-sum variant).
#[pyclass]
pub struct DMemOffsetBPDecoder {
    /// Base struct for BP-based decoders, which stores parity-check matrix and prior error probabilities.
    base: BPBase,
    /// Memory strength for each variable node.
    gamma: Array1<f64>,
    /// Offset parameter for each CN-to-VN edge.
    offset: Vec<Vec<f64>>,
    /// Normalization factor for each CN-to-VN edge.
    norm: Vec<Vec<f64>>,
    /// Maximum number of iterations.
    max_iter: usize,
    /// `chk_inmsg[i]` stores the incoming messages at CN `i` from its neighboring VNs during the current BP iteration.
    chk_inmsg: Vec<Vec<f64>>,
    /// `var_inmsg[j]` stores the incoming messages at VN `j` from its neighboring CNs during the current BP iteration.
    var_inmsg: Vec<Vec<f64>>,
}

#[pymethods]
impl DMemOffsetBPDecoder {
    /// Create a BP decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix (dtype=np.uint8). Every row (check) must have at least 2 nonzero entries.
    /// Every column (variable) must have at least 1 nonzero entry.
    /// - `prior`: Prior error probabilities (dtype=np.float64).
    /// - `gamma`: Memory strength for each variable node (dtype=np.float64). The value 0.0 means no memory.
    /// - `offset`: Offset parameter for each CN-to-VN edge.
    /// - `norm`: Normalization factor for each CN-to-VN edge.
    /// - `max_iter`: Maximum number of BP iterations.
    #[new]
    #[pyo3(signature = (pcm, prior, *, gamma, offset, norm, max_iter))]
    pub fn new(
        pcm: PyReadonlyArray2<'_, u8>,
        prior: PyReadonlyArray1<'_, f64>,
        gamma: PyReadonlyArray1<'_, f64>,
        offset: Vec<Vec<f64>>,
        norm: Vec<Vec<f64>>,
        max_iter: usize,
    ) -> Self {
        let pcm = pcm.as_array();
        let prior = prior.as_array();
        let gamma = gamma.as_array();
        let base = BPBase::new(pcm, prior);

        let mut var_inmsg = Vec::new();
        for j in 0..base.num_vars {
            var_inmsg.push(vec![0.0; base.var_nbrs[j].len()]);
        }

        let mut chk_inmsg = Vec::new();
        for i in 0..base.num_chks {
            chk_inmsg.push(vec![0.0; base.chk_nbrs[i].len()]);
        }

        Self {
            base: base,
            gamma: gamma.to_owned(),
            offset: offset,
            norm: norm,
            max_iter: max_iter,
            chk_inmsg: chk_inmsg,
            var_inmsg: var_inmsg,
        }
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector (dtype=np.uint8).
    ///
    /// Return: The decoded error vector.
    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
    ) -> Bound<'py, PyArray1<u8>> {
        let syndrome = syndrome.as_array();
        let ehat = self._decode(syndrome);
        PyArray1::from_owned_array(py, ehat)
    }

    /// Decode a syndrome vector and return detailed information about the decoding process.
    ///
    /// Parameters:
    /// - `syndrome`: Syndrome vector (dtype=np.uint8).
    /// - `record_llr_history`: Whether to record the history of posterior LLR values.
    ///
    /// Return: A Python dictionary with the following key-value pairs:
    /// - "ehat": The decoded error vector.
    /// - "converged": Whether the decoder converged (i.e. the syndrome is satisfied).
    /// - "num_iter": The number of iterations executed.
    /// - "llr_hist": The history of posterior LLR values (if `record_llr_history` is True).
    #[pyo3(signature = (syndrome, *, record_llr_history))]
    pub fn decode_detailed<'py>(
        &mut self,
        py: Python<'py>,
        syndrome: PyReadonlyArray1<'py, u8>,
        record_llr_history: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let syndrome = syndrome.as_array();
        let (ehat, converged, num_iter, llr_hist) =
            self._decode_detailed(syndrome, record_llr_history);
        let dict = PyDict::new(py);
        dict.set_item("ehat", PyArray1::from_owned_array(py, ehat))?;
        dict.set_item("converged", PyBool::new(py, converged))?;
        dict.set_item("num_iter", PyInt::new(py, num_iter))?;
        if let Some(arr) = llr_hist {
            dict.set_item("llr_hist", PyArray2::from_owned_array(py, arr))?;
        }
        Ok(dict)
    }

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors (dtype=np.uint8).
    ///
    /// Return: The batch of decoded error vectors.
    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'_, u8>,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size: usize = syndrome_batch.nrows();
        let mut ehat_batch = Array2::<u8>::zeros((batch_size, self.base.num_vars));

        for i in 0..batch_size {
            let ehat = self._decode(syndrome_batch.row(i));
            ehat_batch.row_mut(i).assign(&ehat);
        }

        PyArray2::from_owned_array(py, ehat_batch)
    }
}

impl DMemOffsetBPDecoder {
    /// (Re-)initialize the decoder. More specifically, initialize the VN-to-CN messages.
    fn init(&mut self) {
        for j in 0..self.base.num_vars {
            let msg = self.base.prior_llr[j];
            for (k, &i) in self.base.var_nbrs[j].iter().enumerate() {
                self.chk_inmsg[i][self.base.var_nbr_pos[j][k]] = msg;
            }
        }
    }

    /// Decode a syndrome vector.
    ///
    /// Parameters:
    /// - `synd`: Syndrome vector.
    ///
    /// Return: The decoded error vector.
    fn _decode(&mut self, synd: ArrayView1<u8>) -> Array1<u8> {
        self.init();
        // Estimated error vector at the current iteration.
        let mut ehat = Array1::<u8>::zeros(self.base.num_vars);
        // Posterior LLR values at the current iteration.
        let mut llr = self.base.prior_llr.to_vec();

        // Main BP iteration loop.
        for _ in 0..self.max_iter {
            // Message processing at CNs.
            for i in 0..self.base.num_chks {
                // List of incoming messages.
                let inmsg = &self.chk_inmsg[i];
                // List of sign parities of the incoming messages (0 for positive, 1 for negative).
                let inmsg_sgnpar: Vec<u8> =
                    inmsg.iter().map(|&x| if x < 0.0 { 1 } else { 0 }).collect();
                // Total sign parity of the incoming messages (i.e. XOR of the entries in inmsg_sgnpar).
                let total_sgnpar = inmsg_sgnpar.iter().fold(0, |acc, &x| acc ^ x);
                // Minimum absolute value of the incoming messages.
                let mut minabs1 = f64::MAX;
                // Second minimum absolute value of the incoming messages.
                let mut minabs2 = f64::MAX;
                // Index of the incoming message with minimum absolute value.
                let mut minidx = 0;
                for (k, &val) in inmsg.iter().enumerate() {
                    let val_abs = val.abs();
                    if val_abs < minabs1 {
                        minabs2 = minabs1;
                        minabs1 = val_abs;
                        minidx = k;
                    } else if val_abs < minabs2 {
                        minabs2 = val_abs;
                    }
                }
                // Calculate the outgoing messages.
                for (k, &j) in self.base.chk_nbrs[i].iter().enumerate() {
                    let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                    let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                    let msg_abs_offset = (msg_abs - self.offset[i][k]).max(0.0);
                    let msg = if msg_sgnpar == 0 {
                        msg_abs_offset
                    } else {
                        -msg_abs_offset
                    };
                    self.var_inmsg[j][self.base.chk_nbr_pos[i][k]] = self.norm[i][k] * msg;
                }
            }

            // Message processing at VNs.
            for j in 0..self.base.num_vars {
                // List of incoming messages.
                let inmsg = &self.var_inmsg[j];
                // Get posterior LLR.
                llr[j] = (1.0 - self.gamma[j]) * self.base.prior_llr[j]
                    + self.gamma[j] * llr[j]
                    + inmsg.iter().sum::<f64>();
                // Hard decision.
                ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
                // Calculate the outgoing messages.
                for (k, &i) in self.base.var_nbrs[j].iter().enumerate() {
                    self.chk_inmsg[i][self.base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
                }
            }

            // Check if the syndrome is satisfied. If so, early stop.
            let mut satisfied = true;
            for i in 0..self.base.num_chks {
                let mut parity = 0_u8;
                for &j in self.base.chk_nbrs[i].iter() {
                    parity ^= ehat[j];
                }
                if parity != synd[i] {
                    satisfied = false;
                    break;
                }
            }
            if satisfied {
                break;
            }
        }
        ehat
    }

    /// Decode a syndrome vector and return detailed information about the decoding process.
    ///
    /// Parameters:
    /// - `synd`: Syndrome vector.
    /// - `record_llr_history`: Whether to record the history of posterior LLR values.
    ///
    /// Return:
    /// - `ehat`: The decoded error vector.
    /// - `converged`: Whether the decoder converged (i.e. the syndrome is satisfied).
    /// - `num_iter`: The number of iterations executed.
    /// - `llr_hist`: The history of posterior LLR values (if `record_llr_history` is True).
    fn _decode_detailed(
        &mut self,
        synd: ArrayView1<u8>,
        record_llr_history: bool,
    ) -> (Array1<u8>, bool, usize, Option<Array2<f64>>) {
        self.init();
        // Estimated error vector at the current iteration.
        let mut ehat = Array1::<u8>::zeros(self.base.num_vars);
        // Posterior LLR values at the current iteration.
        let mut llr = self.base.prior_llr.to_vec();
        // History of posterior LLR values, stored as a flattened vector.
        let mut llr_hist_flattened = Vec::<f64>::new();

        // Main BP iteration loop.
        let mut num_iter = 0;
        let mut converged = false;
        while num_iter < self.max_iter {
            num_iter += 1;

            // Message processing at CNs.
            for i in 0..self.base.num_chks {
                // List of incoming messages.
                let inmsg = &self.chk_inmsg[i];
                // List of sign parities of the incoming messages (0 for positive, 1 for negative).
                let inmsg_sgnpar: Vec<u8> =
                    inmsg.iter().map(|&x| if x < 0.0 { 1 } else { 0 }).collect();
                // Total sign parity of the incoming messages (i.e. XOR of the entries in inmsg_sgnpar).
                let total_sgnpar = inmsg_sgnpar.iter().fold(0, |acc, &x| acc ^ x);
                // Minimum absolute value of the incoming messages.
                let mut minabs1 = f64::MAX;
                // Second minimum absolute value of the incoming messages.
                let mut minabs2 = f64::MAX;
                // Index of the incoming message with minimum absolute value.
                let mut minidx = 0;
                for (k, &val) in inmsg.iter().enumerate() {
                    let val_abs = val.abs();
                    if val_abs < minabs1 {
                        minabs2 = minabs1;
                        minabs1 = val_abs;
                        minidx = k;
                    } else if val_abs < minabs2 {
                        minabs2 = val_abs;
                    }
                }
                // Calculate the outgoing messages.
                for (k, &j) in self.base.chk_nbrs[i].iter().enumerate() {
                    let msg_sgnpar = synd[i] ^ total_sgnpar ^ inmsg_sgnpar[k];
                    let msg_abs = if k == minidx { minabs2 } else { minabs1 };
                    let msg_abs_offset = (msg_abs - self.offset[i][k]).max(0.0);
                    let msg = if msg_sgnpar == 0 {
                        msg_abs_offset
                    } else {
                        -msg_abs_offset
                    };
                    self.var_inmsg[j][self.base.chk_nbr_pos[i][k]] = self.norm[i][k] * msg;
                }
            }

            // Message processing at VNs.
            for j in 0..self.base.num_vars {
                // List of incoming messages.
                let inmsg = &self.var_inmsg[j];
                // Get posterior LLR.
                llr[j] = (1.0 - self.gamma[j]) * self.base.prior_llr[j]
                    + self.gamma[j] * llr[j]
                    + inmsg.iter().sum::<f64>();
                // Hard decision.
                ehat[j] = if llr[j] < 0.0 { 1 } else { 0 };
                // Calculate the outgoing messages.
                for (k, &i) in self.base.var_nbrs[j].iter().enumerate() {
                    self.chk_inmsg[i][self.base.var_nbr_pos[j][k]] = llr[j] - inmsg[k];
                }
            }

            // Record LLR values.
            if record_llr_history {
                llr_hist_flattened.extend_from_slice(&llr);
            }

            // Check if the syndrome is satisfied. If so, early stop.
            let mut satisfied = true;
            for i in 0..self.base.num_chks {
                let mut parity = 0_u8;
                for &j in self.base.chk_nbrs[i].iter() {
                    parity ^= ehat[j];
                }
                if parity != synd[i] {
                    satisfied = false;
                    break;
                }
            }
            if satisfied {
                converged = true;
                break;
            }
        }

        // Convert the flattened LLR history vector into a 2D array.
        let llr_hist = if record_llr_history {
            Some(
                Array2::from_shape_vec((num_iter, self.base.num_vars), llr_hist_flattened).unwrap(),
            )
        } else {
            None
        };

        (ehat, converged, num_iter, llr_hist)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DMemOffsetBPDecoder>()?;
    Ok(())
}
