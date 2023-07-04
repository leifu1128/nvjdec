use std::os::raw::c_int;
use color_eyre::eyre::{eyre, WrapErr};
use crate::decoder::image::{DeviceImage};
use crate::manager::Manager;
use crate::nv;
use crate::nv::{IntoResult, nvjpegHandle_t, nvjpegJpegState_t, nvjpegOutputFormat_t, NvjpegResult};


mod buffer;
mod image;


pub fn get_image_info(handle: nvjpegHandle_t, image: &'_[u8], length: usize) -> NvjpegResult<[i32; 3]> {
    let mut dims: [i32; 3] = [0, 0, 0];
    unsafe {
        nv::nvjpegGetImageInfo(
            handle,
            image.as_ptr(),
            length,
            &mut dims[0],
            std::ptr::null_mut(),
            &mut dims[2],
            &mut dims[1],
        ).into_result()?;
    }
    Ok(dims)
}

pub trait Decode {
    type Input<'a>;
    type Output;
    fn init(&mut self, handle: nvjpegHandle_t, state: nvjpegJpegState_t) -> NvjpegResult<()>;
    fn decode(&self, input: Self::Input<'_>, handle: nvjpegHandle_t, state: nvjpegJpegState_t) -> NvjpegResult<Self::Output>;
}

struct BatchDecoder {
    batch_size: i32,
    format: nvjpegOutputFormat_t,
}

impl BatchDecoder {
    pub fn new(batch_size: i32, format: nvjpegOutputFormat_t) -> Self {
        BatchDecoder { batch_size, format }
    }
}

impl Decode for BatchDecoder {
    type Input<'a> = Vec<&'a[u8]>;
    type Output = Vec<DeviceImage>;

    fn init(&mut self, handle: nvjpegHandle_t, state: nvjpegJpegState_t) -> NvjpegResult<()> {
        unsafe {
            nv::nvjpegDecodeBatchedInitialize(
                handle,
                state,
                self.batch_size,
                1,
                self.format,
            ).into_result()
                .wrap_err("Failed to initialize decoder")?;
        }
        Ok(())
    }

    fn decode(&self, input: Self::Input<'_>, handle: nvjpegHandle_t, state: nvjpegJpegState_t) -> NvjpegResult<Self::Output> {
        unimplemented!()
    }
}
