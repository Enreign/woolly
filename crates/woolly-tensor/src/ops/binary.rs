//! Binary operations on tensors

use crate::backend::Result;

/// Element-wise addition
pub struct Add;

impl Add {
    /// Adds two arrays element-wise
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Add<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = *a + *b;
        }
        
        Ok(())
    }
    
    /// Optimized addition for f32 using SIMD
    pub fn apply_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        use super::simd::SimdF32;
        SimdF32::add(a, b, output);
        Ok(())
    }
}

/// Element-wise subtraction
pub struct Sub;

impl Sub {
    /// Subtracts two arrays element-wise
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Sub<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = *a - *b;
        }
        
        Ok(())
    }
}

/// Element-wise multiplication
pub struct Mul;

impl Mul {
    /// Multiplies two arrays element-wise
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = *a * *b;
        }
        
        Ok(())
    }
    
    /// Optimized multiplication for f32 using SIMD
    pub fn apply_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        use super::simd::SimdF32;
        SimdF32::mul(a, b, output);
        Ok(())
    }
}

/// Element-wise division
pub struct Div;

impl Div {
    /// Divides two arrays element-wise
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Div<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = *a / *b;
        }
        
        Ok(())
    }
}

/// Element-wise maximum
pub struct Max;

impl Max {
    /// Computes element-wise maximum
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = if *a > *b { *a } else { *b };
        }
        
        Ok(())
    }
}

/// Element-wise minimum
pub struct Min;

impl Min {
    /// Computes element-wise minimum
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = if *a < *b { *a } else { *b };
        }
        
        Ok(())
    }
}

/// Element-wise power
pub struct Pow;

impl Pow {
    /// Raises elements of a to powers in b
    pub fn apply(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = a.powf(*b);
        }
        
        Ok(())
    }
}

/// Fused multiply-add: a * b + c
pub struct FusedMultiplyAdd;

impl FusedMultiplyAdd {
    /// Computes a * b + c element-wise
    pub fn apply<T>(a: &[T], b: &[T], c: &[T], output: &mut [T]) -> Result<()>
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        assert_eq!(a.len(), output.len());
        
        for (((a, b), c), o) in a.iter().zip(b.iter()).zip(c.iter()).zip(output.iter_mut()) {
            *o = *a * *b + *c;
        }
        
        Ok(())
    }
    
    /// Computes a * b + c element-wise using FMA instruction when available
    pub fn apply_f32(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        assert_eq!(a.len(), output.len());
        
        for (((a, b), c), o) in a.iter().zip(b.iter()).zip(c.iter()).zip(output.iter_mut()) {
            *o = a.mul_add(*b, *c);
        }
        
        Ok(())
    }
}

/// Element-wise equality comparison
pub struct Equal;

impl Equal {
    /// Compares two arrays element-wise for equality
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [bool]) -> Result<()>
    where
        T: PartialEq,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = a == b;
        }
        
        Ok(())
    }
    
    /// Compares two arrays element-wise for equality, returning u8 (0 or 1)
    pub fn apply_u8<T>(a: &[T], b: &[T], output: &mut [u8]) -> Result<()>
    where
        T: PartialEq,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = if a == b { 1 } else { 0 };
        }
        
        Ok(())
    }
}

/// Element-wise greater than comparison
pub struct Greater;

impl Greater {
    /// Compares two arrays element-wise for greater than
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [bool]) -> Result<()>
    where
        T: PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = a > b;
        }
        
        Ok(())
    }
    
    /// Compares two arrays element-wise for greater than, returning u8 (0 or 1)
    pub fn apply_u8<T>(a: &[T], b: &[T], output: &mut [u8]) -> Result<()>
    where
        T: PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = if a > b { 1 } else { 0 };
        }
        
        Ok(())
    }
}

/// Element-wise less than comparison
pub struct Less;

impl Less {
    /// Compares two arrays element-wise for less than
    pub fn apply<T>(a: &[T], b: &[T], output: &mut [bool]) -> Result<()>
    where
        T: PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = a < b;
        }
        
        Ok(())
    }
    
    /// Compares two arrays element-wise for less than, returning u8 (0 or 1)
    pub fn apply_u8<T>(a: &[T], b: &[T], output: &mut [u8]) -> Result<()>
    where
        T: PartialOrd,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        for ((a, b), o) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *o = if a < b { 1 } else { 0 };
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        
        Add::apply(&a, &b, &mut output).unwrap();
        
        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }
    
    #[test]
    fn test_max() {
        let a = vec![1.0, 5.0, 3.0, 7.0];
        let b = vec![4.0, 2.0, 6.0, 8.0];
        let mut output = vec![0.0; 4];
        
        Max::apply(&a, &b, &mut output).unwrap();
        
        assert_eq!(output, vec![4.0, 5.0, 6.0, 8.0]);
    }
    
    #[test]
    fn test_fma() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![9.0, 10.0, 11.0, 12.0];
        let mut output = vec![0.0; 4];
        
        FusedMultiplyAdd::apply_f32(&a, &b, &c, &mut output).unwrap();
        
        assert_eq!(output, vec![14.0, 22.0, 32.0, 44.0]);
    }
}