use numpy::ndarray::{Array1, Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashSet, LinkedList, VecDeque};
use std::ops::Add;

/// Union-Find Data Structure: a partition of {0, 1, ..., n-1} into disjoint subsets that supports the following basic operations:
/// - Find the representative element of the set containing a given element.
/// - Merge the sets containing two given elements (if they are not already in the same set).
///
/// This data structure is realized as a forest of trees, one for each set. The representative element of each set is chosen to
/// be the root of the corresponding tree.
struct UFDS {
    /// Total number of elements in the universe.
    n: usize,
    /// Parent of each node in the forest. Root nodes are their own parents.
    parent: Vec<usize>,
    /// If `x` is the root node of a tree, then `size[x]` is the size of that tree; otherwise, `size[x]` is undefined.
    size: Vec<usize>,
}

impl UFDS {
    /// Create and initialize a Union-Find data structure for `n` elements.
    fn new(n: usize) -> Self {
        Self {
            n: n,
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Reset the Union-Find data structure to the initial state: every element forms a singleton set.
    fn reset(&mut self) {
        for i in 0..self.n {
            self.parent[i] = i;
            self.size[i] = 1;
        }
    }

    /// Find the representative element of the set containing `x`.
    fn find(&mut self, x: usize) -> usize {
        assert!(x < self.n, "Element index out of bounds");

        if self.parent[x] == x {
            return x;
        } else {
            let r = self.find(self.parent[x]);
            self.parent[x] = r; // path compression
            return r;
        }
    }

    /// Merge the sets containing `x` and `y` (if they are not already in the same set).
    ///
    /// Return:
    /// - `rx`: the representative element of the original set containing `x`
    /// - `ry`: the representative element of the original set containing `y`
    /// - `r`: the representative element of the merged set
    fn union(&mut self, x: usize, y: usize) -> (usize, usize, usize) {
        let rx = self.find(x);
        let ry = self.find(y);

        if rx == ry {
            // `x` and `y` are already in the same set
            return (rx, rx, rx);
        }

        // Union by size: set the parent of the root of the smaller tree to be the root of the larger tree.
        if self.size[rx] < self.size[ry] {
            self.parent[rx] = ry;
            self.size[ry] += self.size[rx];
            return (rx, ry, ry);
        } else {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
            return (rx, ry, rx);
        }
    }
}

/// Metadata for a cluster.
struct Cluster {
    /// Whether the cluster contains an odd number of nodes with nonzero syndrome.
    odd: bool,
    /// Whether the cluster contains a node that is connected to a virtual boundary node via a fully grown edge.
    touches_boundary: bool,
    /// Linked list of nodes in the cluster that are on the frontier. A node is on the frontier
    /// if at least one of the incident edges is not fully grown.
    frontier: LinkedList<usize>,
    /// A designated node in the cluster that tells the peeling decoder to consider it as the root when constructing a spanning tree.
    /// - If `touches_boundary` is false, `st_root` can be any node in the cluster.
    /// - If `touches_boundary` is true, `st_root` must be a node in the cluster that is connected to a virtual boundary
    /// node via a fully grown edge, but is otherwise arbitrary.
    st_root: usize,
}

impl Cluster {
    /// Create a cluster that contains a single node `x` with nonzero syndrome.
    fn new(x: usize) -> Self {
        Self {
            odd: true,
            touches_boundary: false,
            frontier: LinkedList::from([x]),
            st_root: x,
        }
    }

    /// Check if the cluster is active. An active cluster is one with `odd == true` and `touches_boundary == false`.
    fn is_active(&self) -> bool {
        self.odd && !self.touches_boundary
    }
}

impl Add for Cluster {
    type Output = Self;

    /// Combine the two clusters' metadata.
    fn add(self, other: Self) -> Self {
        let mut frontier = self.frontier;
        let mut other_frontier = other.frontier;
        // Concatenate the two linked lists in O(1) complexity.
        frontier.append(&mut other_frontier);

        Self {
            odd: self.odd ^ other.odd,
            touches_boundary: self.touches_boundary | other.touches_boundary,
            frontier: frontier,
            st_root: if !self.touches_boundary && other.touches_boundary {
                other.st_root
            } else {
                self.st_root
            },
        }
    }
}

/// Union-Find decoder.
#[pyclass]
pub struct UnionFindDecoder {
    /// Number of checks (= number of rows of pcm)
    num_chks: usize,
    /// Number of variables (= number of columns of pcm)
    num_vars: usize,
    /// List of variables involved in each check.
    chk2vars: Vec<Vec<usize>>,
    /// List of checks each variable is involved in.
    var2chks: Vec<Vec<usize>>,
    /// Decoding graph without virtual boundary nodes and dangling edges.
    /// Every node corresponds to a check, and every edge corresponds to a degree-2 variable.
    /// Note that degree-1 variables are not included in the graph, and are handled separately.
    /// For each node `x`, `graph[x]` is the list of (neighboring node, incident edge) pairs.
    graph: Vec<Vec<(usize, usize)>>,
    /// Union-Find data structure on `num_chks` elements to keep track of the clusters.
    /// `uf` maintains a partition of all the check nodes into disjoint sets, some of which are clusters.
    uf: UFDS,
    /// For each node `x`, if `x` is the representative node of a cluster, then `clusters[x]`
    /// stores the metadata of that cluster; otherwise, `clusters[x]` is `None`.
    clusters: Vec<Option<Cluster>>,
    /// Growth status of each edge: 0 if not grown, 1 if half-grown, 2 if fully grown.
    growth: Vec<u8>,
    /// For each node `x`, if `x` is contained in a cluster and is not a root node in the spanning
    /// forest constructed in the peeling decoding stage, then `sf_parent[x] == Some((p, e))` where
    /// `p` is the parent node of `x` and `e` is the edge that connects `x` and `p`; otherwise,
    /// `sf_parent[x] == None`.
    sf_parent: Vec<Option<(usize, usize)>>,
    /// For each node `x`, if `x` is contained in a cluster and is not a root node in the spanning
    /// forest, then `sf_nchild[x]` is the number of children of `x` in the spanning forest; otherwise,
    /// `sf_nchild[x]` is undefined.
    sf_nchild: Vec<usize>,
    /// The list of all leaves of the spanning forest.
    sf_leaves: Vec<usize>,
}

#[pymethods]
impl UnionFindDecoder {
    /// Create a Union-Find decoder.
    ///
    /// Parameters:
    /// - `pcm`: Parity-check matrix (dtype=np.uint8). Every row (check) must have at least 2 nonzero entries.
    /// Every column (variable) must have at least 1 and at most 2 nonzero entries.
    #[new]
    pub fn new(pcm: PyReadonlyArray2<'_, u8>) -> Self {
        let pcm = pcm.as_array();
        let num_chks = pcm.nrows();
        let num_vars = pcm.ncols();

        let mut chk2vars = vec![Vec::new(); num_chks];
        let mut var2chks = vec![Vec::new(); num_vars];
        for i in 0..num_chks {
            for j in 0..num_vars {
                if pcm[[i, j]] != 0 {
                    chk2vars[i].push(j);
                    var2chks[j].push(i);
                }
            }
        }
        for i in 0..num_chks {
            assert!(
                chk2vars[i].len() >= 2,
                "Check {} involves less than 2 variables",
                i
            );
        }
        for j in 0..num_vars {
            assert!(
                var2chks[j].len() >= 1,
                "Variable {} is not involved in any check",
                j
            );
            assert!(
                var2chks[j].len() <= 2,
                "Variable {} is involved in more than 2 checks",
                j
            );
        }

        let mut graph = vec![Vec::new(); num_chks];
        for e in 0..num_vars {
            if var2chks[e].len() == 1 {
                continue;
            }
            let x = var2chks[e][0];
            let y = var2chks[e][1];
            graph[x].push((y, e));
            graph[y].push((x, e));
        }

        Self {
            num_chks: num_chks,
            num_vars: num_vars,
            chk2vars: chk2vars,
            var2chks: var2chks,
            graph: graph,
            uf: UFDS::new(num_chks),
            clusters: (0..num_chks).map(|_| None).collect(),
            growth: vec![0; num_vars],
            sf_parent: vec![None; num_chks],
            sf_nchild: vec![0; num_chks],
            sf_leaves: Vec::new(),
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

    /// Decode a batch of syndrome vectors.
    ///
    /// Parameters:
    /// - `syndrome_batch`: Batch of syndrome vectors (dtype=np.uint8).
    ///
    /// Return: The batch of decoded error vectors.
    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        syndrome_batch: PyReadonlyArray2<'py, u8>,
    ) -> Bound<'py, PyArray2<u8>> {
        let syndrome_batch = syndrome_batch.as_array();
        let batch_size: usize = syndrome_batch.nrows();
        let mut ehat_batch = Array2::<u8>::zeros((batch_size, self.num_vars));

        for i in 0..batch_size {
            let ehat = self._decode(syndrome_batch.row(i));
            ehat_batch.row_mut(i).assign(&ehat);
        }

        PyArray2::from_owned_array(py, ehat_batch)
    }
}

impl UnionFindDecoder {
    /// (Re-)initialize the decoder for the given syndrome vector `synd`.
    ///
    /// Return: The list of the nodes with nonzero syndrome.
    fn init(&mut self, synd: ArrayView1<u8>) -> Vec<usize> {
        self.uf.reset();
        self.growth.fill(0);

        let mut nonzero_nodes = Vec::new();
        for x in 0..self.num_chks {
            if synd[x] != 0 {
                self.clusters[x] = Some(Cluster::new(x));
                nonzero_nodes.push(x);
            } else {
                self.clusters[x] = None;
            }
        }
        nonzero_nodes
    }

    /// Decode the given syndrome vector `synd`.
    ///
    /// Return: The decoded error vector.
    fn _decode(&mut self, synd: ArrayView1<u8>) -> Array1<u8> {
        // List of the representative nodes of active clusters.
        let mut active_clusters = self.init(synd);
        // Grow the clusters until no active cluster remains.
        while !active_clusters.is_empty() {
            active_clusters = self.grow_once(active_clusters);
        }
        // Perform peeling decoding.
        self.peel_decode(synd)
    }

    /// Perform one growing step.
    ///
    /// Parameters:
    /// - `active_clusters`: The list of the representative nodes of active clusters.
    ///
    /// Return: The updated list of the representative nodes of active clusters.
    fn grow_once(&mut self, active_clusters: Vec<usize>) -> Vec<usize> {
        //////////////////////////////////////////////////////////////
        // Grow the edges. Only self.growth will be updated in this stage.
        //////////////////////////////////////////////////////////////
        // Store the edges that newly become fully grown.
        let mut new_edges = Vec::new();
        // For each active cluster with representative node r
        for &r in active_clusters.iter() {
            let cluster = self.clusters[r].as_ref().unwrap();
            // For each frontier node x of the cluster
            for &x in cluster.frontier.iter() {
                // For each edge e incident to the node x
                for &e in self.chk2vars[x].iter() {
                    if self.growth[e] == 2 {
                        // This edge is already fully grown.
                        continue;
                    }
                    // Grow the edge e.
                    self.growth[e] += 1;
                    if self.growth[e] == 2 {
                        new_edges.push(e);
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////
        // Expand the active clusters along the newly fully grown edges.
        // Merge (active or not) clusters when they meet at a node.
        // Note that a previously non-active cluster may become active again
        // after merging with an active cluster.
        // Only self.uf and self.clusters will be updated in this stage.
        //////////////////////////////////////////////////////////////
        for &e in new_edges.iter() {
            if self.var2chks[e].len() == 1 {
                // This edge connects a check node to a virtual boundary node.
                let x = self.var2chks[e][0];
                let rx = self.uf.find(x);
                let cluster = self.clusters[rx].as_mut().unwrap();
                cluster.touches_boundary = true;
                cluster.st_root = x;
                continue;
            }

            // This edge connects two check nodes.
            let x = self.var2chks[e][0];
            let y = self.var2chks[e][1];
            let (rx, ry, r) = self.uf.union(x, y);
            if rx == ry {
                // The two check nodes are already in the same cluster.
                continue;
            }

            match (self.clusters[rx].take(), self.clusters[ry].take()) {
                (Some(mut cx), None) => {
                    // `x` is contained in a cluster, but `y` is not.
                    // Expand the cluster.
                    cx.frontier.push_back(y);
                    self.clusters[r] = Some(cx);
                }
                (None, Some(mut cy)) => {
                    // `y` is contained in a cluster, but `x` is not.
                    // Expand the cluster.
                    cy.frontier.push_back(x);
                    self.clusters[r] = Some(cy);
                }
                (Some(cx), Some(cy)) => {
                    // `x` and `y` are contained in two different clusters.
                    // Merge the clusters.
                    self.clusters[r] = Some(cx + cy);
                }
                (None, None) => {
                    panic!("You should never run into this case.");
                }
            }
        }

        //////////////////////////////////////////////////////////////
        // Obtain the new representative nodes of the clusters that were updated
        // in the previous stage. Only self.uf will be updated in this stage.
        //////////////////////////////////////////////////////////////
        let updated_clusters: HashSet<usize> = active_clusters
            .into_iter()
            .map(|x| self.uf.find(x))
            .collect();

        //////////////////////////////////////////////////////////////
        // Remove nodes from the frontier lists that are no longer on the frontier.
        // Only self.clusters will be updated in this stage.
        //////////////////////////////////////////////////////////////
        for &r in updated_clusters.iter() {
            let cluster = self.clusters[r].as_mut().unwrap();
            let mut new_frontier = LinkedList::new();
            while let Some(x) = cluster.frontier.pop_front() {
                let mut flag = false;
                for &e in self.chk2vars[x].iter() {
                    if self.growth[e] != 2 {
                        flag = true;
                        break;
                    }
                }
                if flag {
                    new_frontier.push_back(x);
                }
            }
            cluster.frontier = new_frontier;
        }

        //////////////////////////////////////////////////////////////
        // Return the new list of representative nodes of active clusters.
        //////////////////////////////////////////////////////////////
        updated_clusters
            .into_iter()
            .filter(|&r| self.clusters[r].as_ref().unwrap().is_active())
            .collect()
    }

    /// Construct a spanning tree for each cluster via breadth-first search, using only fully grown edges.
    /// `self.sf_parent`, `self.sf_nchild`, and `self.sf_leaves` will be modified in this function.
    fn construct_spanning_forest(&mut self) {
        self.sf_parent.fill(None);
        self.sf_leaves.clear();
        let mut visited = vec![false; self.num_chks];
        for r in 0..self.num_chks {
            let st_root = match self.clusters[r] {
                Some(ref cl) => cl.st_root,
                None => continue,
            };
            visited[st_root] = true;
            let mut queue = VecDeque::from([st_root]);
            while let Some(x) = queue.pop_front() {
                self.sf_nchild[x] = 0;
                for &(y, e) in self.graph[x].iter() {
                    if self.growth[e] != 2 || visited[y] {
                        continue;
                    }
                    self.sf_parent[y] = Some((x, e));
                    visited[y] = true;
                    queue.push_back(y);
                    self.sf_nchild[x] += 1;
                }
                if self.sf_nchild[x] == 0 {
                    self.sf_leaves.push(x);
                }
            }
        }
    }

    fn peel_decode(&mut self, synd: ArrayView1<u8>) -> Array1<u8> {
        self.construct_spanning_forest();
        let mut synd = synd.to_vec();
        let mut ehat = Array1::<u8>::zeros(self.num_vars);
        let mut queue = VecDeque::from_iter(self.sf_leaves.iter().cloned());
        while let Some(x) = queue.pop_front() {
            if let Some((p, e)) = self.sf_parent[x] {
                if synd[x] == 1 {
                    ehat[e] = 1;
                    synd[x] = 0;
                    synd[p] ^= 1;
                }
                self.sf_nchild[p] -= 1;
                if self.sf_nchild[p] == 0 {
                    queue.push_back(p);
                }
            } else {
                // `x` is a root node in the spanning forest.
                if synd[x] == 1 {
                    // It must be the case that `x` is connected to a virtual boundary node via a fully grown edge.
                    for &e in self.chk2vars[x].iter() {
                        if self.growth[e] == 2 && self.var2chks[e].len() == 1 {
                            ehat[e] = 1;
                            synd[x] = 0;
                            break;
                        }
                    }
                }
            }
        }
        assert!(
            synd.iter().all(|&s| s == 0),
            "Syndrome is not 0 after peeling decoding"
        );
        ehat
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UnionFindDecoder>()?;
    Ok(())
}
