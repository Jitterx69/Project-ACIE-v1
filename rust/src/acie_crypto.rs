use num_bigint::{BigInt, BigUint, RandBigInt, Sign};
use num_traits::One;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::thread_rng;

#[pyclass]
pub struct RustPaillier {
    n: BigInt,
    nsquare: BigInt,
}

#[pymethods]
impl RustPaillier {
    #[new]
    fn new(n_str: &str) -> PyResult<Self> {
        let n: BigInt = n_str
            .parse()
            .map_err(|_| PyValueError::new_err("Invalid N string"))?;
        let nsquare = &n * &n;
        Ok(RustPaillier { n, nsquare })
    }

    /// Encrypt a 64-bit integer
    #[pyo3(text_signature = "($self, m)")]
    fn encrypt(&self, m: i64) -> String {
        let one = BigInt::one();
        let m_big = BigInt::from(m);

        let mut rng = thread_rng();
        // Generate random r: 1 <= r < n
        // We use unwrap because n is guaranteed positive from constructor usually,
        // but for safety we could handle it. Here unwrap is safe enough for demo.
        let n_uint = self.n.to_biguint().unwrap_or(BigUint::one());
        let r_uint = rng.gen_biguint_range(&BigUint::from(1u32), &n_uint);

        let r = BigInt::from_biguint(Sign::Plus, r_uint);

        // c = (1 + m*n) * r^n mod n^2
        let gm = (&one + &m_big * &self.n) % &self.nsquare;
        let rn = r.modpow(&self.n, &self.nsquare);

        ((gm * rn) % &self.nsquare).to_string()
    }

    #[pyo3(text_signature = "($self, c1, c2)")]
    fn add(&self, c1: &str, c2: &str) -> PyResult<String> {
        let a: BigInt = c1
            .parse()
            .map_err(|_| PyValueError::new_err("Invalid c1"))?;
        let b: BigInt = c2
            .parse()
            .map_err(|_| PyValueError::new_err("Invalid c2"))?;

        Ok(((a * b) % &self.nsquare).to_string())
    }

    #[pyo3(text_signature = "($self, c, k)")]
    fn mul(&self, c: &str, k: i64) -> PyResult<String> {
        let a: BigInt = c.parse().map_err(|_| PyValueError::new_err("Invalid c"))?;
        let k_big = BigInt::from(k);

        Ok(a.modpow(&k_big, &self.nsquare).to_string())
    }
}
