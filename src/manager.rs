use std::os::raw::c_int;
use color_eyre::eyre::{eyre, WrapErr};
use crate::decoder::Decode;
use crate::nv;
use crate::nv::{IntoResult, nvjpegBackend_t, nvjpegHandle_t, nvjpegJpegState_t, NvjpegResult};


pub struct Manager<D: Decode> {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    device: usize,
    decoder: Option<D>,
}

impl<D: Decode> Manager<D> {
    pub fn initialize(backend: nvjpegBackend_t, device: usize, flag: u32) -> NvjpegResult<Manager<D>> {
        let mut mngr = Manager {
            handle: std::ptr::null_mut(),
            state: std::ptr::null_mut(),
            device,
            decoder: None,
        };
        unsafe {
            nv::cudaSetDevice(device as c_int).into_result()
                .wrap_err("Failed to set device")?;
            nv::nvjpegCreateEx(
                backend,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                flag,
                &mut mngr.handle,
            ).into_result()
                .wrap_err("Failed to create handle")?;
            nv::nvjpegJpegStateCreate(
                mngr.handle,
                &mut mngr.state,
            ).into_result()
                .wrap_err("Failed to create jpeg state")?;
        }
        Ok(mngr)
    }

    pub fn set_device(&self) -> NvjpegResult<()> {
        unsafe {
            nv::cudaSetDevice(self.device as c_int).into_result()
                .wrap_err("Failed to set device")?;
        }
        Ok(())
    }

    pub fn set_decoder(&mut self, mut decoder: D) -> NvjpegResult<()> {
        self.set_device()?;
        decoder.init(self.handle, self.state)?;
        self.decoder = Some(decoder);
        Ok(())
    }

    pub fn decode(&mut self, input: D::Input<'_>) -> NvjpegResult<D::Output> {
        self.set_device()?;
        match &self.decoder {
            Some(decoder) => decoder.decode(input, self.handle, self.state),
            None => Err(eyre!("No decoder set")),
        }
    }

    fn teardown(&mut self) -> NvjpegResult<()> {
        self.set_device()?;
        unsafe{
            nv::nvjpegDestroy(self.handle).into_result()
                .wrap_err("Failed to destroy handle")?;
            nv::nvjpegJpegStateDestroy(self.state).into_result()
                .wrap_err("Failed to destroy jpeg state")?;
        }
        Ok(())
    }
}

impl<D: Decode> Drop for Manager<D> {
    fn drop(&mut self) {
        self.teardown()
            .wrap_err("Failed to teardown decoder").unwrap();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use tch::display::set_print_options_full;
    use pretty_assertions::{assert_eq};
    use tch::IndexOp;
    use crate::decoder::nv;

    #[test]
    fn test_initialize() {
        let backend = nv::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT;
        let device = 0;
        let flag = 0;

        let manager = Manager::initialize(backend, device, flag).unwrap();

        assert!(manager.handle.is_null());
        assert!(manager.state.is_null());
        assert_eq!(manager.device, device);
        assert!(manager.decoder.is_none());
    }

    #[test]
    fn test_set_device() {
        let backend = nv::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT;
        let device = 0;
        let flag = 0;

        let manager = Manager::initialize(backend, device, flag).unwrap();

        assert!(manager.set_device().is_ok());
    }

    #[test]
    fn test_set_decoder() {
        let backend = nv::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT;
        let device = 0;
        let flag = 0;

        let mut manager = Manager::initialize(backend, device, flag).unwrap();

        let decoder = MockDecoder::new();

        assert!(manager.set_decoder(decoder).is_ok());
        assert!(manager.decoder.is_some());
    }

    #[test]
    fn test_decode() {
        let backend = nv::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT;
        let device = 0;
        let flag = 0;

        let mut manager = Manager::initialize(backend, device, flag).unwrap();

        let decoder = MockDecoder::new();
        manager.set_decoder(decoder).unwrap();

        let input = MockInput::new();

        assert!(manager.decode(input).is_ok());
    }

    #[test]
    fn test_teardown() {
        let backend = nv::nvjpegBackend_t::NVJPEG_BACKEND_DEFAULT;
        let device = 0;
        let flag = 0;

        let mut manager = Manager::initialize(backend, device, flag).unwrap();

        assert!(manager.teardown().is_ok());
    }
}

