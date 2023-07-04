use std::os::raw::c_int;
use color_eyre::eyre::{eyre, WrapErr};
use pyo3::prelude::*;
use crate::decoder::image::{DeviceImage};
use crate::nv;
use crate::nv::{IntoResult, nvjpegHandle_t, nvjpegJpegState_t, NvjpegResult};


mod buffer;
mod image;

struct DecoderBackend {
    backend: nv::nvjpegBackend_t,
    handle: nv::nvjpegHandle_t,
    state: nv::nvjpegJpegState_t,
    device: usize,
}

impl DecoderBackend {
    fn setup(backend: nv::nvjpegBackend_t, device: usize, flag: u32) -> NvjpegResult<()> {
        let mut db = DecoderBackend{
            backend,
            handle: std::ptr::null_mut(),
            state: std::ptr::null_mut(),
            device,
        };

        unsafe {
            nv::cudaSetDevice(device as c_int).into_result()
                .wrap_err("Failed to set device")?;
            nv::nvjpegCreateEx(
                backend,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                flag,
                &mut db.handle,
            ).into_result()
                .wrap_err("Failed to create handle")?;
            nv::nvjpegJpegStateCreate(
                db.handle,
                &mut db.state,
            ).into_result()
                .wrap_err("Failed to create jpeg state")?;
        }
        Ok(())
    }

    fn teardown(&mut self) -> NvjpegResult<()> {
        unsafe{
            nv::cudaSetDevice(self.device as c_int).into_result()
                .wrap_err("Failed to set device")?;
            nv::nvjpegDestroy(self.handle).into_result()
                .wrap_err("Failed to destroy handle")?;
            nv::nvjpegJpegStateDestroy(self.state).into_result()
                .wrap_err("Failed to destroy jpeg state")?;
        }
        Ok(())
    }
}

impl Drop for DecoderBackend {
    fn drop(&mut self) {
        self.teardown()
            .wrap_err("Failed to teardown decoder").unwrap();
    }
}

