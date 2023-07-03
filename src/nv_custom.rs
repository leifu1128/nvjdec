/* Additions: */
use color_eyre::eyre::eyre;
use color_eyre::Result;

pub struct NvjpegHandle(pub(crate) Nvjpeg);
pub struct NvjpegJpegState(pub(crate) nvjpegJpegState_t);
pub struct CudaStream(pub(crate) cudaStream_t);

impl Default for NvjpegHandle {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl Default for NvjpegJpegState {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl Default for nvjpegBackend_t {
    fn default() -> Self {
        nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT
    }
}

impl nvjpegBackend_t {
    pub(crate) fn get_flag(&self) -> u32 {
        match self {
            nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE => NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE,
            nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID_DEVICE => NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE,
            nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE_DEVICE => NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE,
            _ => NVJPEG_FLAGS_DEFAULT,
        }
    }
}

pub type NvjpegResult<T> = Result<T>;
pub type CudaResult<T> = Result<T>;

pub trait IntoResult {
    fn into_result(self) -> Result<()>;
}

impl IntoResult for nvjpegStatus_t {
    fn into_result(self) -> Result<()> {
        match self {
            nvjpegStatus_t::NVJPEG_STATUS_SUCCESS => Ok(()),
            _ => Err(eyre!("{:?}", self)),
        }
    }
}

impl IntoResult for cudaError_t {
    fn into_result(self) -> Result<()> {
        match self {
            cudaError_t::cudaSuccess => Ok(()),
            _ => Err(eyre!("{:?}", self)),
        }
    }
}

impl nvjpegOutputFormat_t {
    pub fn n_channels(self) -> NvjpegResult<i32> {
        match self {
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_Y => Ok(1),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB => Ok(3),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR => Ok(3),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGBI => Ok(3),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI => Ok(3),
            _ => return Err(eyre!("Unsupported output format")),
        }
    }
}
