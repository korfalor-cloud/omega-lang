/// GPU-style compute: parallel map, reduce, scan, and matrix operations.

use std::sync::{Arc, Mutex};
use std::thread;

/// Parallel map over a slice using multiple threads.
pub fn parallel_map<T, U, F>(data: &[T], f: F, num_threads: usize) -> Vec<U>
where
    T: Send + Sync + Clone + 'static,
    U: Send + 'static + Default,
    F: Fn(&T) -> U + Send + Sync + 'static,
{
    let f = Arc::new(f);
    let chunk_size = (data.len() + num_threads - 1) / num_threads;
    let result = Arc::new(Mutex::new(vec![U::default(); data.len()]));

    let mut handles = Vec::new();
    for (chunk_idx, chunk) in data.chunks(chunk_size).enumerate() {
        let chunk: Vec<T> = chunk.to_vec();
        let f = Arc::clone(&f);
        let result = Arc::clone(&result);
        let offset = chunk_idx * chunk_size;

        handles.push(thread::spawn(move || {
            for (i, item) in chunk.iter().enumerate() {
                let val = f(item);
                let mut res = result.lock().unwrap();
                res[offset + i] = val;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}

/// Parallel reduce.
pub fn parallel_reduce<T, F>(data: &[T], identity: T, f: F, num_threads: usize) -> T
where
    T: Send + Clone + 'static,
    F: Fn(&T, &T) -> T + Send + Sync + 'static,
{
    let f = Arc::new(f);
    let chunk_size = (data.len() + num_threads - 1) / num_threads;

    let mut handles = Vec::new();
    for chunk in data.chunks(chunk_size) {
        let chunk: Vec<T> = chunk.to_vec();
        let f = Arc::clone(&f);
        let identity = identity.clone();

        handles.push(thread::spawn(move || {
            chunk.iter().fold(identity, |acc, x| f(&acc, x))
        }));
    }

    let mut result = identity.clone();
    for handle in handles {
        let partial = handle.join().unwrap();
        result = f(&result, &partial);
    }

    result
}

/// Parallel prefix sum (scan).
pub fn parallel_scan(data: &[f64], num_threads: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return Vec::new(); }

    let chunk_size = (n + num_threads - 1) / num_threads;
    let chunks: Vec<Vec<f64>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

    // Phase 1: Compute local prefix sums
    let local_sums: Vec<Vec<f64>> = parallel_map(&chunks, |chunk| {
        let mut prefix = vec![0.0; chunk.len()];
        if !chunk.is_empty() {
            prefix[0] = chunk[0];
            for i in 1..chunk.len() {
                prefix[i] = prefix[i - 1] + chunk[i];
            }
        }
        prefix
    }, num_threads);

    // Phase 2: Compute offsets
    let mut offsets = vec![0.0; chunks.len()];
    for i in 1..chunks.len() {
        offsets[i] = offsets[i - 1] + local_sums[i - 1].last().copied().unwrap_or(0.0);
    }

    // Phase 3: Apply offsets
    let mut result = vec![0.0; n];
    let mut idx = 0;
    for (i, local) in local_sums.iter().enumerate() {
        for &val in local {
            result[idx] = val + offsets[i];
            idx += 1;
        }
    }

    result
}

/// Parallel matrix multiply.
pub fn parallel_matmul(a: &[Vec<f64>], b: &[Vec<f64>], num_threads: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();

    // Transpose B for cache efficiency
    let bt: Vec<Vec<f64>> = (0..n).map(|j| (0..k).map(|i| b[i][j]).collect()).collect();

    let rows: Vec<usize> = (0..m).collect();
    let result = Arc::new(Mutex::new(vec![vec![0.0; n]; m]));

    let chunk_size = (m + num_threads - 1) / num_threads;
    let mut handles = Vec::new();

    for chunk in rows.chunks(chunk_size) {
        let chunk = chunk.to_vec();
        let a = a.to_vec();
        let bt = bt.clone();
        let result = Arc::clone(&result);

        handles.push(thread::spawn(move || {
            for &i in &chunk {
                for j in 0..n {
                    let val: f64 = a[i].iter().zip(bt[j].iter()).map(|(x, y)| x * y).sum();
                    let mut res = result.lock().unwrap();
                    res[i][j] = val;
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}

/// Parallel sort using merge sort.
pub fn parallel_sort(data: &mut [f64], num_threads: usize) {
    let n = data.len();
    if n <= 1 { return; }

    // For small arrays, use regular sort
    if n < 1000 || num_threads <= 1 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return;
    }

    let mid = n / 2;
    let (left, right) = data.split_at_mut(mid);

    let mut handles = Vec::new();

    // Sort left half
    let mut left_vec = left.to_vec();
    handles.push(thread::spawn(move || {
        left_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        left_vec
    }));

    // Sort right half
    let mut right_vec = right.to_vec();
    handles.push(thread::spawn(move || {
        right_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        right_vec
    }));

    let sorted_left = handles[0].join().unwrap();
    let sorted_right = handles[1].join().unwrap();

    // Merge
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < sorted_left.len() && j < sorted_right.len() {
        if sorted_left[i] <= sorted_right[j] {
            data[k] = sorted_left[i];
            i += 1;
        } else {
            data[k] = sorted_right[j];
            j += 1;
        }
        k += 1;
    }
    while i < sorted_left.len() {
        data[k] = sorted_left[i];
        i += 1;
        k += 1;
    }
    while j < sorted_right.len() {
        data[k] = sorted_right[j];
        j += 1;
        k += 1;
    }
}

/// Histogram computation.
pub fn histogram(data: &[f64], bins: usize) -> Vec<(f64, f64, usize)> {
    if data.is_empty() || bins == 0 { return Vec::new(); }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-15 {
        return vec![(min, max, data.len())];
    }

    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];

    for &val in data {
        let bin = ((val - min) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    (0..bins).map(|i| {
        let lo = min + i as f64 * bin_width;
        let hi = lo + bin_width;
        (lo, hi, counts[i])
    }).collect()
}

/// Parallel dot product.
pub fn parallel_dot(a: &[f64], b: &[f64], num_threads: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    let chunk_size = (a.len() + num_threads - 1) / num_threads;

    let mut handles = Vec::new();
    for (a_chunk, b_chunk) in a.chunks(chunk_size).zip(b.chunks(chunk_size)) {
        let a_chunk = a_chunk.to_vec();
        let b_chunk = b_chunk.to_vec();
        handles.push(thread::spawn(move || {
            a_chunk.iter().zip(b_chunk.iter()).map(|(x, y)| x * y).sum::<f64>()
        }));
    }

    handles.into_iter().map(|h| h.join().unwrap()).sum()
}

/// Sparse matrix (CSR format).
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows, cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn from_triplets(triplets: &[(usize, usize, f64)], rows: usize, cols: usize) -> Self {
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        // Sort by row, then column
        let mut sorted = triplets.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Count elements per row
        for &(r, _, _) in &sorted {
            row_ptr[r + 1] += 1;
        }

        // Cumulative sum
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Extract columns and values
        for &(_, c, v) in &sorted {
            col_idx.push(c);
            values.push(v);
        }

        Self { rows, cols, row_ptr, col_idx, values }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for i in start..end {
            if self.col_idx[i] == col {
                return self.values[i];
            }
        }
        0.0
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for i in start..end {
            if self.col_idx[i] == col {
                self.values[i] = value;
                return;
            }
        }
    }

    /// Sparse matrix-vector multiply.
    pub fn mat_vec_mul(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for j in start..end {
                y[i] += self.values[j] * x[self.col_idx[j]];
            }
        }

        y
    }

    /// Sparse matrix-matrix multiply.
    pub fn mat_mul(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = SparseMatrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            let mut row_entries: Vec<(usize, f64)> = Vec::new();

            let self_start = self.row_ptr[i];
            let self_end = self.row_ptr[i + 1];

            for j in self_start..self_end {
                let k = self.col_idx[j];
                let a_val = self.values[j];

                let other_start = other.row_ptr[k];
                let other_end = other.row_ptr[k + 1];

                for l in other_start..other_end {
                    let col = other.col_idx[l];
                    let b_val = other.values[l];

                    // Add to existing entry or create new
                    if let Some(entry) = row_entries.iter_mut().find(|e| e.0 == col) {
                        entry.1 += a_val * b_val;
                    } else {
                        row_entries.push((col, a_val * b_val));
                    }
                }
            }

            row_entries.sort_by_key(|e| e.0);
            for (col, val) in row_entries {
                result.col_idx.push(col);
                result.values.push(val);
            }
            result.row_ptr[i + 1] = result.col_idx.len();
        }

        result
    }

    /// Transpose.
    pub fn transpose(&self) -> SparseMatrix {
        let mut triplets = Vec::new();
        for i in 0..self.rows {
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push((self.col_idx[j], i, self.values[j]));
            }
        }
        Self::from_triplets(&triplets, self.cols, self.rows)
    }

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// GPU-style parallel kernel execution simulation.
pub struct KernelExecutor {
    pub num_threads: usize,
}

impl KernelExecutor {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Element-wise kernel.
    pub fn elementwise<F>(&self, a: &[f64], b: &[f64], op: F) -> Vec<f64>
    where
        F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        let op = Arc::new(op);
        parallel_map(
            &a.iter().zip(b.iter()).map(|(&x, &y)| (x, y)).collect::<Vec<_>>(),
            |&(x, y)| op(x, y),
            self.num_threads,
        )
    }

    /// Reduction kernel.
    pub fn reduce<F>(&self, data: &[f64], identity: f64, op: F) -> f64
    where
        F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        parallel_reduce(data, identity, op, self.num_threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_map(&data, |x| x * x, 2);
        assert_eq!(result, vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    }

    #[test]
    fn test_parallel_reduce() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = parallel_reduce(&data, 0.0, |a, b| a + b, 2);
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_parallel_scan() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_scan(&data, 2);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_sparse_matrix() {
        let triplets = vec![
            (0, 0, 1.0), (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0), (2, 2, 5.0),
        ];
        let m = SparseMatrix::from_triplets(&triplets, 3, 3);

        let x = vec![1.0, 2.0, 3.0];
        let y = m.mat_vec_mul(&x);
        assert_eq!(y[0], 1.0 + 6.0);
        assert_eq!(y[1], 6.0);
        assert_eq!(y[2], 4.0 + 15.0);
    }

    #[test]
    fn test_parallel_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = parallel_dot(&a, &b, 2);
        assert_eq!(result, 70.0);
    }
}
