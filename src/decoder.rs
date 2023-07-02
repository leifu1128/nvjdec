use std::os::raw::c_int;
use color_eyre::eyre::{eyre, WrapErr};
use pyo3::prelude::*;
use crate::decoder::image::{DeviceImage, OnDeviceFrom};
use crate::decoder::nv::{IntoResult, Nvjpeg, nvjpegJpegState_t, NvjpegResult};


mod nv;
mod buffer;
mod image;

pub struct DecoderBuilder {
    backend: nv::nvjpegBackend_t,
    device: usize,
}

impl DecoderBuilder {
    pub fn new(backend: nv::nvjpegBackend_t, device: usize) -> DecoderBuilder {
        DecoderBuilder { backend, device }
    }

    pub fn build(self) -> NvjpegResult<Decoder> {
        let mut n_devices = 0;
        unsafe {
            nv::cudaGetDeviceCount(&mut n_devices).into_result()
                .wrap_err("Failed to get device count")?;
        }
        if self.device >= n_devices as usize {
            return Err(eyre!("Device index out of range"))
        }

        let mut decoder = Decoder::default();
        decoder.setup()?;
        Ok(decoder)
    }

    pub fn with_batch_size(self, batch_size: i32) -> NvjpegResult<Decoder> {
        decoder.batch_init(batch_size)?;
        Ok(decoder)
    }
}

#[derive(Default)]
struct Decoder {
    backend: nv::nvjpegBackend_t,
    handle: nv::NvjpegHandle,
    state: nv::NvjpegJpegState,
    stream: nv::CudaStream,
    device: usize,
    batch_size: Option<i32>,
}

impl Decoder {
    fn setup(&mut self) -> NvjpegResult<()> {
        unsafe {
            nv::cudaSetDevice(self.device as c_int).into_result()
                .wrap_err("Failed to set device")?;

            nv::cudaStreamCreate(&mut self.stream.0).into_result()
                .wrap_err("Failed to create stream")?;

            nv::nvjpegCreateEx(
                self.backend,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.backend.get_flag(),
                &mut self.handle.0,
            ).into_result()
                .wrap_err("Failed to create handle")?;

            nv::nvjpegJpegStateCreate(
                self.handle.0,
                &mut self.state.0,
            ).into_result()
                .wrap_err("Failed to create jpeg state")?;
        }
        Ok(())
    }

    fn batch_init(&mut self, batch_size: i32) -> NvjpegResult<()> {
        unsafe {
            nv::nvjpegDecodeBatchedInitialize(
                self.handle.0,
                self.state.0,
                batch_size,
                1,
                nv::nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB
            ).into_result()
                .wrap_err("Failed to initialize decoder")?;
        }
        self.batch_size = Some(batch_size);
        Ok(())
    }

    fn get_image_info(&self, image: &[u8], length: usize) -> NvjpegResult<[i32; 3]> {
        let mut dims: [i32; 3] = [0, 0, 0];
        unsafe {
            nv::nvjpegGetImageInfo(
                self.handle.0,
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

    fn decode(&self, images: Vec<&[u8]>) -> NvjpegResult<Vec<DeviceImage>> {
        let batch_size = self.batch_size.expect("batch size not set, call setup first");
        assert_eq!(
            images.len(), batch_size as usize,
            "input batch size: {} does not match decoder batch size: {}",
            images.len(), batch_size
        );

        let mut image_ptrs: Vec<*const u8> = Vec::with_capacity(images.len());
        let mut lengths: Vec<usize> = Vec::with_capacity(images.len());
        let mut image_dims: Vec<[i32; 3]> = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let length = image.len();
            let image_dim = self.get_image_info(image, length)
                .wrap_err(format!("Failed to get image info for {}/{}", i, batch_size))?;
            image_ptrs.push(image.as_ptr());
            lengths.push(image.len());
            image_dims.push(image_dim);
        }

        let mut outputs: Vec<nv::nvjpegImage_t> = Vec::with_capacity(images.len());
        unsafe {
            nv::nvjpegDecodeBatched(
                self.handle.0,
                self.state.0,
                & image_ptrs[0],
                & lengths[0],
                &mut outputs[0],
                self.stream.0,
            ).into_result()?;
        }

        Ok(outputs.iter().zip(image_dims.iter()).map(
            |(img, dim)|{
                DeviceImage::ondev_from((*img, *dim), self.device)
            }).collect())
    }

    fn teardown(&mut self) -> NvjpegResult<()> {
        unsafe{
            nv::nvjpegDestroy(self.handle.0).into_result()
                .wrap_err("Failed to destroy handle")?;
            nv::nvjpegJpegStateDestroy(self.state.0).into_result()
                .wrap_err("Failed to destroy jpeg state")?;
        }
        self.batch_size = None;
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.teardown()
            .wrap_err("Failed to teardown decoder").unwrap();
    }
}
