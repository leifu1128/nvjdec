use std::convert::Into;
use libc::c_int;
use crate::nv;
use crate::nv::{cudaMalloc, cudaMallocAsync, nvjpegDevAllocator_t, nvjpegPinnedAllocator_t, nvjpegPinnedAllocatorV2_t};

/*
pub const SYNC_DEV_ALLOCATOR: nvjpegDevAllocator_t = nvjpegDevAllocator_t {
	dev_malloc: Some(|p, s| unsafe {
		cudaMalloc(p, s)
	} as c_int ),
	dev_free: Some(|p| unsafe {
		nv::cudaFree(p)
	} as c_int),
};

pub const SYNC_PINNED_ALLOCATOR: nvjpegPinnedAllocator_t = nvjpegPinnedAllocator_t {
	pinned_malloc: Some(|p, s, f| unsafe {
			nv::cudaHostAlloc(p, s, f)
	} as c_int),
	pinned_free: Some(|p| unsafe {
			nv::cudaFreeHost(p)
	} as c_int),
};
*/