//! Shape and stride types for tensor dimensions

use std::fmt;
use std::ops::{Index, IndexMut};

use crate::backend::{TensorError, Result};

/// Represents the shape of a tensor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from a vector of dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    /// Creates a shape from a slice of dimensions
    pub fn from_slice(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }
    
    /// Creates a scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }
    
    /// Creates a 1D shape
    pub fn vector(size: usize) -> Self {
        Self { dims: vec![size] }
    }
    
    /// Creates a 2D shape
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self { dims: vec![rows, cols] }
    }
    
    /// Returns the number of dimensions (rank)
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    /// Returns the dimensions as a slice
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }
    
    /// Returns the dimensions as a slice (alias for as_slice)
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    
    /// Returns a mutable reference to the dimensions
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.dims
    }
    
    /// Returns the total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Returns whether this is a scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    
    /// Returns whether this is a vector (1-dimensional)
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }
    
    /// Returns whether this is a matrix (2-dimensional)
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
    
    /// Checks if this shape is compatible with another for broadcasting
    pub fn is_broadcast_compatible(&self, other: &Shape) -> bool {
        if self.ndim() == 0 || other.ndim() == 0 {
            return true;
        }
        
        let min_ndim = self.ndim().min(other.ndim());
        let self_offset = self.ndim() - min_ndim;
        let other_offset = other.ndim() - min_ndim;
        
        for i in 0..min_ndim {
            let self_dim = self.dims[self_offset + i];
            let other_dim = other.dims[other_offset + i];
            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }
        
        true
    }
    
    /// Returns the broadcasted shape of two shapes
    pub fn broadcast_shape(&self, other: &Shape) -> Result<Shape> {
        if !self.is_broadcast_compatible(other) {
            return Err(TensorError::incompatible_shapes(
                "SHAPE_BROADCAST_INCOMPATIBLE",
                "Shapes are not compatible for broadcasting",
                "shape broadcasting",
                format!("{:?}", self),
                format!("{:?}", other),
                "Ensure shapes follow broadcasting rules"
            ));
        }
        
        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = vec![1; max_ndim];
        
        let self_offset = max_ndim - self.ndim();
        let other_offset = max_ndim - other.ndim();
        
        for i in 0..max_ndim {
            let self_dim = if i >= self_offset { self.dims[i - self_offset] } else { 1 };
            let other_dim = if i >= other_offset { other.dims[i - other_offset] } else { 1 };
            
            result_dims[i] = self_dim.max(other_dim);
        }
        
        Ok(Shape::new(result_dims))
    }
    
    /// Validates that the shape is valid
    pub fn validate(&self) -> Result<()> {
        // Check for overflow when computing total elements
        let mut total = 1usize;
        for &dim in &self.dims {
            if dim == 0 {
                return Err(TensorError::invalid_shape(
                    "SHAPE_ZERO_DIMENSION",
                    "Shape contains zero dimension",
                    format!("{:?}", self),
                    "shape validation",
                    "Zero dimension found",
                    "All shape dimensions must be positive"
                ));
            }
            total = total.checked_mul(dim)
                .ok_or_else(|| TensorError::invalid_shape(
                    "SHAPE_TOO_LARGE",
                    "Shape is too large",
                    format!("{:?}", self),
                    "shape validation",
                    "Overflow in element count",
                    "Use smaller dimensions to avoid overflow"
                ))?;
        }
        Ok(())
    }
    
    /// Computes default strides for this shape (row-major/C-order)
    pub fn default_strides(&self) -> Strides {
        Strides::from_shape(self)
    }
    
    /// Computes strides for column-major (Fortran-order) layout
    pub fn fortran_strides(&self) -> Strides {
        Strides::from_shape_fortran(self)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::from_slice(dims)
    }
}

/// Represents the strides of a tensor
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    strides: Vec<isize>,
}

impl Strides {
    /// Creates new strides from a vector
    pub fn new(strides: Vec<isize>) -> Self {
        Self { strides }
    }
    
    /// Creates strides from a shape in row-major (C) order
    pub fn from_shape(shape: &Shape) -> Self {
        let mut strides = vec![1isize; shape.ndim()];
        for i in (0..shape.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
        Self { strides }
    }
    
    /// Creates strides from a shape in column-major (Fortran) order
    pub fn from_shape_fortran(shape: &Shape) -> Self {
        let mut strides = vec![1isize; shape.ndim()];
        for i in 1..shape.ndim() {
            strides[i] = strides[i - 1] * shape[i - 1] as isize;
        }
        Self { strides }
    }
    
    /// Returns the strides as a slice
    pub fn as_slice(&self) -> &[isize] {
        &self.strides
    }
    
    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.strides.len()
    }
    
    /// Computes the offset for a given multi-dimensional index
    pub fn offset(&self, indices: &[usize]) -> isize {
        assert_eq!(indices.len(), self.strides.len(), "Index dimension mismatch");
        
        indices.iter()
            .zip(&self.strides)
            .map(|(idx, stride)| *idx as isize * stride)
            .sum()
    }
    
    /// Returns whether the strides represent a contiguous layout
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        if shape.ndim() != self.ndim() {
            return false;
        }
        
        // Check C-contiguous
        let c_strides = Strides::from_shape(shape);
        if self.strides == c_strides.strides {
            return true;
        }
        
        // Check Fortran-contiguous
        let f_strides = Strides::from_shape_fortran(shape);
        self.strides == f_strides.strides
    }
    
    /// Creates strides for a transposed tensor
    pub fn transpose(&self, axes: &[usize]) -> Result<Strides> {
        if axes.len() != self.ndim() {
            return Err(TensorError::invalid_shape(
                "STRIDES_TRANSPOSE_AXES_MISMATCH",
                format!("Transpose axes length {} doesn't match tensor dimensions {}", 
                    axes.len(), self.ndim()),
                format!("strides with {} dimensions", self.ndim()),
                "strides transpose",
                "Axes length mismatch",
                "Provide exactly one axis index for each dimension"
            ));
        }
        
        let mut new_strides = vec![0isize; self.ndim()];
        for (i, &axis) in axes.iter().enumerate() {
            if axis >= self.ndim() {
                return Err(TensorError::invalid_shape(
                    "STRIDES_TRANSPOSE_AXIS_OUT_OF_RANGE",
                    format!("Transpose axis {} is out of range for {} dimensions", axis, self.ndim()),
                    format!("strides with {} dimensions", self.ndim()),
                    "strides transpose",
                    "Axis index out of range",
                    "Ensure axis indices are within valid range"
                ));
            }
            new_strides[i] = self.strides[axis];
        }
        
        Ok(Strides::new(new_strides))
    }
    
    /// Creates strides for a sliced tensor
    pub fn slice(&self, ranges: &[(usize, usize, isize)]) -> Result<(Strides, isize)> {
        if ranges.len() != self.ndim() {
            return Err(TensorError::invalid_shape(
                "STRIDES_SLICE_RANGES_MISMATCH",
                format!("Slice ranges length {} doesn't match tensor dimensions {}", 
                    ranges.len(), self.ndim()),
                format!("strides with {} dimensions", self.ndim()),
                "strides slicing",
                "Ranges length mismatch",
                "Provide exactly one range for each dimension"
            ));
        }
        
        let mut new_strides = Vec::with_capacity(self.ndim());
        let mut offset = 0isize;
        
        for (i, &(start, _end, step)) in ranges.iter().enumerate() {
            if step == 0 {
                return Err(TensorError::invalid_shape(
                    "STRIDES_SLICE_ZERO_STEP",
                    "Slice step cannot be zero",
                    format!("strides with {} dimensions", self.ndim()),
                    "strides slicing",
                    "Zero step value",
                    "Use non-zero step values for slicing"
                ));
            }
            
            offset += start as isize * self.strides[i];
            new_strides.push(self.strides[i] * step);
        }
        
        Ok((Strides::new(new_strides), offset))
    }
}

impl Index<usize> for Strides {
    type Output = isize;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.strides[index]
    }
}

impl fmt::Display for Strides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, stride) in self.strides.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", stride)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_creation() {
        let shape = Shape::vector(10);
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.numel(), 10);
        
        let shape = Shape::matrix(3, 4);
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.numel(), 12);
    }
    
    #[test]
    fn test_broadcasting() {
        let shape1 = Shape::from_slice(&[1, 3, 1]);
        let shape2 = Shape::from_slice(&[5, 1, 4]);
        
        assert!(shape1.is_broadcast_compatible(&shape2));
        
        let broadcast = shape1.broadcast_shape(&shape2).unwrap();
        assert_eq!(broadcast.as_slice(), &[5, 3, 4]);
    }
    
    #[test]
    fn test_strides() {
        let shape = Shape::matrix(3, 4);
        let strides = shape.default_strides();
        
        assert_eq!(strides.as_slice(), &[4, 1]);
        assert_eq!(strides.offset(&[1, 2]), 6);
    }
}