use pyo3::{PyObject, PyResult};

mod decoder;



#[pyclass(unsendable)]
struct PyDecoder {
    decoder: Decoder,
}

#[pymethods]
impl PyDecoder {
    #[new]
    fn new(backend: &str, device: usize) -> PyDecoder {
        let backend = match backend {
            "gpu" => nv::nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID,
            "hardware" => nv::nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE,
            _ => panic!("backend must be one of 'gpu' or 'hardware'"),
        };
        PyDecoder {
            decoder: Decoder::new(backend, device),
        }
    }

    fn setup(&mut self, batch_size: i32) -> PyResult<()> {
        self.decoder.setup(batch_size)?;
        Ok(())
    }

    fn decode(&self, images: Vec<Vec<u8>>) -> PyResult<PyObject> {
        todo!()
    }
}