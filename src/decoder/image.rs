use color_eyre::eyre::{eyre, Result, WrapErr};
use color_eyre::Report;
use tch::{Device, Kind, Tensor};
use crate::decoder::buffer::DeviceBuffer;
use crate::decoder::nv::{nvjpegOutputFormat_t, NvjpegResult};

use super::nv::nvjpegImage_t;


#[derive(Debug)]
pub struct DeviceImage {
    channels: DeviceBuffer<u8>,
    pub format: nvjpegOutputFormat_t,
    pub width: i32,
    pub height: i32,
    pub device: i32,
}

impl DeviceImage {
    pub fn try_new(format: nvjpegOutputFormat_t, width: i32, height: i32, device: i32) -> NvjpegResult<DeviceImage> {
        if height <= 0 || width <= 0 {
            Err(eyre!("Invalid image dimensions"))?;
        }
        if device < 0 {
            Err(eyre!("Invalid device index"))?;
        }

        let n_channels = format.n_channels()?;
        let channels = DeviceBuffer::uninitialized((width * height * n_channels) as usize)?;
        Ok(DeviceImage { channels, format, width, height, device })
    }

    pub fn get_strides(&self) -> NvjpegResult<Vec<i32>> {
        let (h, w) = (self.height, self.width);
        match self.format {
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_Y => Ok(vec![w, 1, 1]),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB |
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR => Ok(vec![h * w, w, 1]),
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGBI |
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI => Ok(vec![3, 3 * w, 1]),
            _ => return Err(eyre!("Unsupported output format"))
        }
    }

    pub fn as_nvjpeg_image(&mut self) -> NvjpegResult<nvjpegImage_t> {
        let mut channel = [std::ptr::null_mut(); 4];
        let mut pitch = [0 as usize; 4];
        match self.format {
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_Y |
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGBI |
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI => {
                channel[0] = self.channels.as_mut_ptr();
                pitch[0] = (self.width * self.format.n_channels()?) as usize;
            },
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB |
            nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGR => {
                for i in 0..3 {
                    channel[i] = unsafe {
                        self.channels.as_mut_ptr().clone()
                            .offset((i as i32 * self.width * self.height) as isize)
                    };
                    pitch[i] = self.width as usize;
                }
            },
            _ => Err(eyre!("Unsupported output format"))?,
        }
        Ok(nvjpegImage_t { channel, pitch })
    }
}

impl TryInto<Tensor> for DeviceImage {
    type Error = Report;

    fn try_into(self) -> Result<Tensor> {
        let (h, w) = (self.height, self.width);
        let c = self.format.n_channels()?;
        let strides: Vec<i64> = self.get_strides()?.into_iter().map(|x| x as i64).collect();
        let (ptr, _len) = self.channels.into_raw();
        unsafe {
            Tensor::f_from_blob(
                ptr,
                &[c as i64, h as i64, w as i64],
                strides.as_slice(),
                Kind::Uint8,
                Device::Cuda(self.device as usize),
            ).wrap_err("Failed to create tensor")
        }
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
    fn test_ondev_from_and_into_tensor() {
        set_print_options_full();

        let device = 0;
        let dims = [3, 2, 2];
        let mock_data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        // Create a DeviceImage from the mock data.
        let mut device_image = DeviceImage::try_new(
            nv::nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB, dims[1], dims[2], device
        ).unwrap();
        let mut nvjpeg_image = device_image.as_nvjpeg_image().unwrap();

        // Copy the mock data into the DeviceImage.
        for i in 0..dims[0] { unsafe {
            nv::cudaMemcpy(
                nvjpeg_image.channel[i],
                mock_data.as_ptr().offset((i * dims[1] * dims[2]) as isize) as *const std::ffi::c_void,
                (dims[1] * dims[2]) as usize * size_of::<u8>(),
                nv::cudaMemcpyKind::cudaMemcpyHostToDevice
            )
        } }

        // Convert the DeviceImage into a Tensor.
        let tensor: Tensor = device_image.try_into().unwrap();

        for i in 0..dims.len() {
            assert_eq!(tensor.size()[i] as i32, dims[i]);
        }

        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    assert_eq!(
                        tensor.int64_value(&[i as i64, j as i64, k as i64]) as u8,
                        mock_data[((i * dims[1] + j) * dims[2] + k) as usize]
                    );
                }
            }
        }
    }
}
