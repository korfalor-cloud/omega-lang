use crate::errors::{OmegaError, OmegaResult};

#[derive(Debug, Clone)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    pub fn ones(n: usize) -> Self {
        Self { data: vec![1.0; n] }
    }

    pub fn basis(n: usize, index: usize) -> Self {
        let mut data = vec![0.0; n];
        data[index] = 1.0;
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    pub fn add(&self, other: &Vector) -> OmegaResult<Vector> {
        if self.len() != other.len() {
            return Err(OmegaError::ValueError {
                message: "Vectors must have same length".to_string(),
            });
        }
        Ok(Vector::new(self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect()))
    }

    pub fn sub(&self, other: &Vector) -> OmegaResult<Vector> {
        if self.len() != other.len() {
            return Err(OmegaError::ValueError {
                message: "Vectors must have same length".to_string(),
            });
        }
        Ok(Vector::new(self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect()))
    }

    pub fn scale(&self, scalar: f64) -> Vector {
        Vector::new(self.data.iter().map(|x| x * scalar).collect())
    }

    pub fn dot(&self, other: &Vector) -> OmegaResult<f64> {
        if self.len() != other.len() {
            return Err(OmegaError::ValueError {
                message: "Vectors must have same length".to_string(),
            });
        }
        Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
    }

    pub fn cross(&self, other: &Vector) -> OmegaResult<Vector> {
        if self.len() != 3 || other.len() != 3 {
            return Err(OmegaError::ValueError {
                message: "Cross product requires 3D vectors".to_string(),
            });
        }
        Ok(Vector::new(vec![
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ]))
    }

    pub fn magnitude(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalize(&self) -> Vector {
        let mag = self.magnitude();
        if mag == 0.0 {
            self.clone()
        } else {
            self.scale(1.0 / mag)
        }
    }

    pub fn angle_between(&self, other: &Vector) -> OmegaResult<f64> {
        let cos_theta = self.dot(other)? / (self.magnitude() * other.magnitude());
        Ok(cos_theta.acos())
    }

    pub fn project_onto(&self, other: &Vector) -> OmegaResult<Vector> {
        let scalar = self.dot(other)? / other.dot(other)?;
        Ok(other.scale(scalar))
    }

    pub fn distance_to(&self, other: &Vector) -> OmegaResult<f64> {
        Ok(self.sub(other)?.magnitude())
    }

    pub fn manhattan_distance(&self, other: &Vector) -> OmegaResult<f64> {
        if self.len() != other.len() {
            return Err(OmegaError::ValueError {
                message: "Vectors must have same length".to_string(),
            });
        }
        Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| (a - b).abs()).sum())
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> OmegaResult<Self> {
        if data.is_empty() {
            return Ok(Self { data: Vec::new(), rows: 0, cols: 0 });
        }
        let rows = data.len();
        let cols = data[0].len();
        for row in &data {
            if row.len() != cols {
                return Err(OmegaError::ValueError {
                    message: "All rows must have same length".to_string(),
                });
            }
        }
        Ok(Self { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n {
            data[i][i] = 1.0;
        }
        Self { data, rows: n, cols: n }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }

    pub fn add(&self, other: &Matrix) -> OmegaResult<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(OmegaError::ValueError {
                message: "Matrices must have same dimensions".to_string(),
            });
        }
        let data = self.data.iter().zip(other.data.iter())
            .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect())
            .collect();
        Matrix::new(data)
    }

    pub fn sub(&self, other: &Matrix) -> OmegaResult<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(OmegaError::ValueError {
                message: "Matrices must have same dimensions".to_string(),
            });
        }
        let data = self.data.iter().zip(other.data.iter())
            .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a - b).collect())
            .collect();
        Matrix::new(data)
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let data = self.data.iter()
            .map(|row| row.iter().map(|x| x * scalar).collect())
            .collect();
        Matrix::new(data).unwrap()
    }

    pub fn mul(&self, other: &Matrix) -> OmegaResult<Matrix> {
        if self.cols != other.rows {
            return Err(OmegaError::ValueError {
                message: format!("Cannot multiply {}x{} by {}x{}", self.rows, self.cols, other.rows, other.cols),
            });
        }
        let mut data = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Matrix::new(data)
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }
        Matrix::new(data).unwrap()
    }

    pub fn determinant(&self) -> OmegaResult<f64> {
        if self.rows != self.cols {
            return Err(OmegaError::ValueError {
                message: "Determinant requires square matrix".to_string(),
            });
        }
        match self.rows {
            0 => Ok(0.0),
            1 => Ok(self.data[0][0]),
            2 => Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]),
            _ => {
                let mut det = 0.0;
                for j in 0..self.cols {
                    det += self.data[0][j] * self.cofactor(0, j)?;
                }
                Ok(det)
            }
        }
    }

    fn cofactor(&self, row: usize, col: usize) -> OmegaResult<f64> {
        let minor = self.minor(row, col)?;
        let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
        Ok(sign * minor.determinant()?)
    }

    fn minor(&self, row: usize, col: usize) -> OmegaResult<Matrix> {
        let mut data = Vec::new();
        for (i, r) in self.data.iter().enumerate() {
            if i == row { continue; }
            let mut new_row = Vec::new();
            for (j, &val) in r.iter().enumerate() {
                if j == col { continue; }
                new_row.push(val);
            }
            data.push(new_row);
        }
        Matrix::new(data)
    }

    pub fn inverse(&self) -> OmegaResult<Matrix> {
        let det = self.determinant()?;
        if det.abs() < 1e-10 {
            return Err(OmegaError::ValueError {
                message: "Matrix is not invertible".to_string(),
            });
        }
        let n = self.rows;
        let mut adjugate = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                adjugate.data[j][i] = self.cofactor(i, j)?;
            }
        }
        Ok(adjugate.scale(1.0 / det))
    }

    pub fn trace(&self) -> OmegaResult<f64> {
        if self.rows != self.cols {
            return Err(OmegaError::ValueError {
                message: "Trace requires square matrix".to_string(),
            });
        }
        Ok((0..self.rows).map(|i| self.data[i][i]).sum())
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|row| row.iter()).map(|x| x * x).sum::<f64>().sqrt()
    }
}
