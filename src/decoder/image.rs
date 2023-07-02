use tch::{Device, Kind, Tensor};
use crate::decoder::buffer::DeviceBuffer;
use crate::decoder::nv;


pub type NvjpegDecOut = (nv::nvjpegImage_t, [i32; 3]);

pub struct DeviceImage {
    pub data: [Option<DeviceBuffer<u8>>; 4],
    pub dims: [i32; 3],
    pub device: Device,
}

impl DeviceImage {
    pub(crate) fn new(data: [Option<DeviceBuffer<u8>>; 4], dims: [i32; 3], device: Device) -> DeviceImage {
        DeviceImage { data, dims, device }
    }
}

pub trait OnDeviceFrom<T> {
    fn ondev_from(value: T, device: usize) -> Self;
}

impl OnDeviceFrom<NvjpegDecOut> for DeviceImage {
    fn ondev_from(value: NvjpegDecOut, device: usize) -> Self {
        let (image, dims) = value;
        let (chnls, height, _width) = (dims[0], dims[1], dims[2]);

        let mut data: [Option<DeviceBuffer<u8>>; 4] = [None, None, None, None];
        for c in 0..chnls{
            let (c, height) = (c as usize, height as usize);
            let dev_ptr = image.channel[c] as *mut u8;
            unsafe{
                data[c] = Some(DeviceBuffer::from_raw(dev_ptr, height * image.pitch[c]));
            }
        }

        DeviceImage::new(data, dims, Device::Cuda(device))
    }
}

impl Into<Tensor> for DeviceImage {
    fn into(self) -> Tensor {
        let from_blob = |data_ptr| unsafe {Tensor::from_blob(
            data_ptr,
            &[self.dims[1] as i64, self.dims[2] as i64],
            &[self.dims[2] as i64, 1],
            Kind::Uint8,
            self.device
        )};
        let chnl_tensors: Vec<Tensor> = self.data.into_iter().filter_map(|data|
            data.map(|data|
                from_blob(data.into_raw().0)
            )
        ).collect();
        Tensor::stack(&chnl_tensors, 0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use tch::display::set_print_options_full;
    use pretty_assertions::{assert_eq};
    use tch::IndexOp;

    #[test]
    fn test_ondev_from_and_into_tensor() {
        set_print_options_full();

        let device = 0;
        let dims = [3, 2, 2];
        let mock_data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        let mut mock_image = nv::nvjpegImage_t {
            channel: [std::ptr::null_mut(); 4],
            pitch: [2; 4],
        };

        // Allocate GPU memory and copy data for each channel.
        for i in 0..dims[0] as usize { unsafe {
            let mut dev_buf = DeviceBuffer::uninitialized(dims[1] as usize * dims[2] as usize).unwrap();
            dev_buf.copy_from(&mock_data[i*4..i*4+4]).unwrap();
            mock_image.channel[i] = dev_buf.into_raw().0;
        } }

        let mock_nvjpeg_dec_out = (mock_image, dims);

        // Create a DeviceImage from the NvjpegDecOut.
        let device_image = DeviceImage::ondev_from(mock_nvjpeg_dec_out, device);

        // Convert the DeviceImage into a Tensor.
        let tensor: Tensor = device_image.into();

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
