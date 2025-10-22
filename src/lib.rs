mod bp;
mod dmem_bp;
mod dmem_offset_bp;
mod union_find;
use pyo3::prelude::*;

#[pymodule]
fn qecdec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bp::register(m)?;
    dmem_bp::register(m)?;
    dmem_offset_bp::register(m)?;
    union_find::register(m)?;
    Ok(())
}
