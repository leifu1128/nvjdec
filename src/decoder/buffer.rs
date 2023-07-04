use std::mem;
use std::os::raw::c_void;
use color_eyre::eyre::WrapErr;
use crate::decoder::nv;
use crate::decoder::nv::{cudaFree, cudaMemcpyKind, IntoResult, CudaResult};


#[derive(Debug)]
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    pub fn uninitialized(len: usize) -> CudaResult<Self> {
        let mut ptr: *mut T = std::ptr::null_mut();
        let bytes = len * std::mem::size_of::<T>();
        let mut c_ptr: *mut c_void = ptr as *mut c_void;
        unsafe {
            nv::cudaMalloc(&mut c_ptr, bytes).into_result()?;
        }
        ptr = c_ptr as *mut T;
        Ok(DeviceBuffer { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn into_raw(mut self) -> (*mut T, usize) {
        let raw_parts = (self.ptr, self.len);
        self.ptr = std::ptr::null_mut();
        self.len = 0;
        raw_parts
    }

    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn copy_from_host_slice(&mut self, src: &[T]) -> CudaResult<()> {
        assert_eq!(self.len, src.len());

        let src_ptr = src.as_ptr();
        let dst_ptr = self.ptr;
        let size = self.len * mem::size_of::<T>();
        unsafe{
            nv::cudaMemcpy(
                dst_ptr as *mut c_void,
                src_ptr as *const c_void,
                size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ).into_result()?;
        }
        Ok(())
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        if self.len > 0 && mem::size_of::<T>() > 0 {
            let ptr = mem::replace(&mut self.ptr, std::ptr::null_mut());
            unsafe {
                cudaFree(ptr as *mut c_void).into_result()
                    .wrap_err("Failed to free device memory").unwrap();
            }
        }
        self.len = 0;
    }
}
