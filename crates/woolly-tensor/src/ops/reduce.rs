//! Reduction operations on tensors

use crate::backend::Result;
use crate::shape::Shape;

/// Sum reduction
pub struct Sum;

impl Sum {
    /// Computes the sum of all elements
    pub fn reduce_all<T>(input: &[T]) -> T
    where
        T: Copy + std::ops::Add<Output = T> + Default,
    {
        input.iter().fold(T::default(), |acc, &x| acc + x)
    }
    
    /// Computes sum along specified axes
    pub fn reduce_axes<T>(
        input: &[T],
        input_shape: &Shape,
        axes: &[usize],
        output: &mut [T],
        _output_shape: &Shape,
    ) -> Result<()>
    where
        T: Copy + std::ops::Add<Output = T> + Default,
    {
        // This is a simplified implementation
        // A real implementation would handle multi-dimensional reductions properly
        
        // For now, just handle the case of reducing all dimensions
        if axes.len() == input_shape.ndim() {
            let sum = Self::reduce_all(input);
            output[0] = sum;
        } else {
            // TODO: Implement proper multi-dimensional reduction
            unimplemented!("Partial reduction not yet implemented");
        }
        
        Ok(())
    }
}

/// Mean reduction
pub struct Mean;

impl Mean {
    /// Computes the mean of all elements
    pub fn reduce_all<T>(input: &[T]) -> T
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + Default + From<f32>,
    {
        let sum = Sum::reduce_all(input);
        let count = T::from(input.len() as f32);
        sum / count
    }
    
    /// Computes mean along specified axes
    pub fn reduce_axes<T>(
        input: &[T],
        input_shape: &Shape,
        axes: &[usize],
        output: &mut [T],
        _output_shape: &Shape,
    ) -> Result<()>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + Default + From<f32>,
    {
        // Simplified implementation
        if axes.len() == input_shape.ndim() {
            let mean = Self::reduce_all(input);
            output[0] = mean;
        } else {
            unimplemented!("Partial reduction not yet implemented");
        }
        
        Ok(())
    }
}

/// Maximum reduction
pub struct MaxReduce;

impl MaxReduce {
    /// Finds the maximum element
    pub fn reduce_all<T>(input: &[T]) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        if input.is_empty() {
            return None;
        }
        
        let mut max_val = &input[0];
        for val in input.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }
        Some(*max_val)
    }
    
    /// Finds maximum along specified axes
    pub fn reduce_axes<T>(
        input: &[T],
        input_shape: &Shape,
        axes: &[usize],
        output: &mut [T],
        _output_shape: &Shape,
    ) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        if axes.len() == input_shape.ndim() {
            if let Some(max) = Self::reduce_all(input) {
                output[0] = max;
            }
        } else {
            unimplemented!("Partial reduction not yet implemented");
        }
        
        Ok(())
    }
}

/// Minimum reduction
pub struct MinReduce;

impl MinReduce {
    /// Finds the minimum element
    pub fn reduce_all<T>(input: &[T]) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        if input.is_empty() {
            return None;
        }
        
        let mut min_val = &input[0];
        for val in input.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }
        Some(*min_val)
    }
    
    /// Finds minimum along specified axes
    pub fn reduce_axes<T>(
        input: &[T],
        input_shape: &Shape,
        axes: &[usize],
        output: &mut [T],
        _output_shape: &Shape,
    ) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        if axes.len() == input_shape.ndim() {
            if let Some(min) = Self::reduce_all(input) {
                output[0] = min;
            }
        } else {
            unimplemented!("Partial reduction not yet implemented");
        }
        
        Ok(())
    }
}

/// Product reduction
pub struct Prod;

impl Prod {
    /// Computes the product of all elements
    pub fn reduce_all<T>(input: &[T]) -> T
    where
        T: Copy + std::ops::Mul<Output = T> + num_traits::One,
    {
        input.iter().fold(T::one(), |acc, &x| acc * x)
    }
}

/// Variance computation
pub struct Variance;

impl Variance {
    /// Computes variance of all elements
    pub fn compute(input: &[f32], ddof: usize) -> f32 {
        let mean = Mean::reduce_all(input);
        let n = input.len() as f32;
        
        if n <= ddof as f32 {
            return 0.0;
        }
        
        let sum_sq_diff: f32 = input.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();
        
        sum_sq_diff / (n - ddof as f32)
    }
}

/// Standard deviation computation
pub struct StdDev;

impl StdDev {
    /// Computes standard deviation of all elements
    pub fn compute(input: &[f32], ddof: usize) -> f32 {
        Variance::compute(input, ddof).sqrt()
    }
}

/// Argmax operation
pub struct ArgMax;

impl ArgMax {
    /// Finds the index of the maximum element
    pub fn compute<T>(input: &[T]) -> Option<usize>
    where
        T: PartialOrd,
    {
        if input.is_empty() {
            return None;
        }
        
        let mut max_idx = 0;
        let mut max_val = &input[0];
        
        for (i, val) in input.iter().enumerate().skip(1) {
            if val > max_val {
                max_idx = i;
                max_val = val;
            }
        }
        
        Some(max_idx)
    }
}

/// Argmin operation
pub struct ArgMin;

impl ArgMin {
    /// Finds the index of the minimum element
    pub fn compute<T>(input: &[T]) -> Option<usize>
    where
        T: PartialOrd,
    {
        if input.is_empty() {
            return None;
        }
        
        let mut min_idx = 0;
        let mut min_val = &input[0];
        
        for (i, val) in input.iter().enumerate().skip(1) {
            if val < min_val {
                min_idx = i;
                min_val = val;
            }
        }
        
        Some(min_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sum() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let sum = Sum::reduce_all(&input);
        assert_eq!(sum, 10.0);
    }
    
    #[test]
    fn test_mean() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mean = Mean::reduce_all(&input);
        assert_eq!(mean, 2.5);
    }
    
    #[test]
    fn test_variance() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = Variance::compute(&input, 0);
        assert!((var - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_argmax() {
        let input = vec![1.0, 5.0, 3.0, 2.0];
        let idx = ArgMax::compute(&input);
        assert_eq!(idx, Some(1));
    }
}