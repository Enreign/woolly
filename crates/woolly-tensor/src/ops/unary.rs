//! Unary operations on tensors

use crate::backend::Result;

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    /// Applies ReLU: max(0, x)
    pub fn apply<T>(input: &[T], output: &mut [T]) -> Result<()>
    where
        T: PartialOrd + Default + Copy,
    {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = if *i > T::default() { *i } else { T::default() };
        }
        
        Ok(())
    }
    
    /// Optimized ReLU for f32 using SIMD
    pub fn apply_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
        use super::simd::SimdF32;
        SimdF32::relu(input, output);
        Ok(())
    }
}

/// Exponential function
pub struct Exp;

impl Exp {
    /// Applies exponential function
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.exp();
        }
        
        Ok(())
    }
}

/// Natural logarithm
pub struct Ln;

impl Ln {
    /// Applies natural logarithm
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.ln();
        }
        
        Ok(())
    }
}

/// Square root
pub struct Sqrt;

impl Sqrt {
    /// Applies square root
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.sqrt();
        }
        
        Ok(())
    }
}

/// Absolute value
pub struct Abs;

impl Abs {
    /// Applies absolute value
    pub fn apply<T>(input: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + PartialOrd + std::ops::Neg<Output = T> + Default,
    {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = if *i < T::default() { -*i } else { *i };
        }
        
        Ok(())
    }
}

/// Negation
pub struct Neg;

impl Neg {
    /// Applies negation
    pub fn apply<T>(input: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Neg<Output = T>,
    {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = -*i;
        }
        
        Ok(())
    }
}

/// Sigmoid activation function
pub struct Sigmoid;

impl Sigmoid {
    /// Applies sigmoid: 1 / (1 + exp(-x))
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = 1.0 / (1.0 + (-*i).exp());
        }
        
        Ok(())
    }
}

/// Tanh activation function
pub struct Tanh;

impl Tanh {
    /// Applies tanh
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.tanh();
        }
        
        Ok(())
    }
}

/// GELU activation function
pub struct GeLU;

impl GeLU {
    /// Applies GELU: x * Φ(x) where Φ is the CDF of standard normal distribution
    /// Using approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            let x = *i;
            let x3 = x * x * x;
            *o = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x3)).tanh());
        }
        
        Ok(())
    }
}

/// Sine function
pub struct Sin;

impl Sin {
    /// Applies sine function
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.sin();
        }
        
        Ok(())
    }
}

/// Cosine function
pub struct Cos;

impl Cos {
    /// Applies cosine function
    pub fn apply(input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), output.len());
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = i.cos();
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        ReLU::apply(&input, &mut output).unwrap();
        
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_abs() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        Abs::apply(&input, &mut output).unwrap();
        
        assert_eq!(output, vec![2.0, 1.0, 0.0, 1.0, 2.0]);
    }
}