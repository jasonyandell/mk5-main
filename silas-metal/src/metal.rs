// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The Metal backend: the only module that speaks to the GPU, and the only one with `unsafe`
//! (each use is a buffer's bytes in or out, or binding a buffer to a command; the shaders own
//! no pointer the host can see). The shader source is compiled once, at start, from
//! `shaders/epoch.metal`. A batch runs in two passes: count each work item's lines, then
//! write them at offsets the host computes, in chunks that bound the output buffer. A
//! nonzero error word refuses the whole batch.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::epoch::{Backend, Batch, CHILD_WORDS, Child, EpochError, PART_WORDS, SEG_WORDS};
use crate::tables::Tables;

const SOURCE: &str = include_str!("../shaders/epoch.metal");
const LADDER: &str = include_str!("../shaders/ladder.metal");

/// Children per emit command: 2^22 children of 64 bytes, 256 MiB.
pub const CHUNK_CHILDREN: usize = 1 << 22;

pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    count: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    emit: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) ladder: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Commands committed, for the receipt.
    pub commands: u64,
}

fn device_error(what: &str) -> EpochError {
    EpochError::Device(what.to_string())
}

impl Gpu {
    /// The system's default Metal device, with the epoch kernels compiled.
    pub fn new() -> Result<Gpu, EpochError> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| device_error("no Metal device"))?;
        let compile = |source: &str| {
            device
                .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                .map_err(|e| EpochError::Device(format!("the shaders did not compile: {e:?}")))
        };
        let (library, ladder_library) = (compile(SOURCE)?, compile(LADDER)?);
        let pipeline = |library: &ProtocolObject<dyn MTLLibrary>, name: &str| {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| EpochError::Device(format!("no kernel {name}")))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| EpochError::Device(format!("no pipeline for {name}: {e:?}")))
        };
        let (count, emit) = (pipeline(&library, "epoch_count")?, pipeline(&library, "epoch_emit")?);
        let ladder = pipeline(&ladder_library, "ladder_gather")?;
        let queue = device.newCommandQueue().ok_or_else(|| device_error("no command queue"))?;
        Ok(Gpu { device, queue, count, emit, ladder, commands: 0 })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(crate) fn upload(&self, words: &[u32]) -> Result<Buffer, EpochError> {
        let bytes = (words.len().max(1)) * 4;
        let one = [0u32; 1];
        let src = if words.is_empty() { &one[..] } else { words };
        let ptr = NonNull::new(src.as_ptr() as *mut c_void).ok_or_else(|| device_error("null slice"))?;
        // SAFETY: `ptr` points at `bytes` readable bytes for the duration of the call, which copies them.
        unsafe { self.device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared) }
            .ok_or_else(|| device_error("buffer allocation failed"))
    }

    pub(crate) fn zeroed(&self, words: usize) -> Result<Buffer, EpochError> {
        let buffer = self
            .device
            .newBufferWithLength_options(words.max(1) * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| device_error("buffer allocation failed"))?;
        // SAFETY: a fresh shared buffer of at least `words` u32s, written before any command uses it.
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr() as *mut u32, words.max(1)) }.fill(0);
        Ok(buffer)
    }

    /// The buffer's first `words` u32s, copied out. Called only after the command that wrote
    /// them completed.
    pub(crate) fn read(buffer: &Buffer, words: usize) -> Vec<u32> {
        assert!(buffer.length() >= words * 4, "a read past the buffer");
        // SAFETY: the buffer holds at least `words` u32s and no command is in flight on it.
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u32, words) }.to_vec()
    }

    /// Sets the buffer's first `words` u32s to `value`. Called only with no command in flight on it.
    pub(crate) fn fill(buffer: &Buffer, words: usize, value: u32) {
        assert!(buffer.length() >= words * 4, "a fill past the buffer");
        // SAFETY: the buffer holds at least `words` u32s and no command is in flight on it.
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr() as *mut u32, words) }.fill(value);
    }

    pub(crate) fn run(
        &mut self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&Buffer],
        threads: usize,
    ) -> Result<(), EpochError> {
        let command = self.queue.commandBuffer().ok_or_else(|| device_error("no command buffer"))?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| device_error("no encoder"))?;
        encoder.setComputePipelineState(pipeline);
        for (i, b) in buffers.iter().enumerate() {
            // SAFETY: each buffer is retained by the caller until the command completes below.
            unsafe { encoder.setBuffer_offset_atIndex(Some(b), 0, i) };
        }
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: threads.max(1), height: 1, depth: 1 },
            MTLSize { width, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        self.commands += 1;
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(EpochError::Device(format!("the command failed: {:?}", command.error())));
        }
        Ok(())
    }
}

impl Backend for Gpu {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn epoch(&mut self, tables: &Tables, batch: Batch<'_>) -> Result<Vec<Child>, EpochError> {
        let n = batch.items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let n32 = u32::try_from(n).map_err(|_| device_error("too many work items"))?;
        let tab = self.upload(tables.words())?;
        let segs: Vec<u32> = batch.segments.iter().flat_map(|s| s.words()).collect();
        let parts: Vec<u32> = batch.particles.iter().flat_map(|p| p.words()).collect();
        let items: Vec<u32> = batch.items.iter().flat_map(|(s, p)| [*s, *p]).collect();
        debug_assert_eq!(segs.len(), batch.segments.len() * SEG_WORDS);
        debug_assert_eq!(parts.len(), batch.particles.len() * PART_WORDS);
        let (segs, parts, items) = (self.upload(&segs)?, self.upload(&parts)?, self.upload(&items)?);
        let counts = self.zeroed(n)?;
        let err = self.zeroed(1)?;
        let n_items = self.upload(&[n32])?;
        let count = self.count.clone();
        self.run(&count, &[&tab, &segs, &parts, &items, &counts, &err, &n_items], n)?;
        if let Some(e) = EpochError::from_code(Self::read(&err, 1)[0]) {
            return Err(e);
        }
        let counts_host = Self::read(&counts, n);
        let total: usize = counts_host.iter().map(|c| *c as usize).sum();
        let mut out = Vec::with_capacity(total);
        let mut start = 0usize;
        let emit = self.emit.clone();
        while start < n {
            // The largest run of items whose children fit one chunk (at least one item).
            let mut end = start;
            let mut lines = 0usize;
            let mut offsets = Vec::new();
            while end < n && (end == start || lines + counts_host[end] as usize <= CHUNK_CHILDREN) {
                offsets.push(lines as u32);
                lines += counts_host[end] as usize;
                end += 1;
            }
            let out_buf = self.zeroed(lines * CHILD_WORDS)?;
            let offsets = self.upload(&offsets)?;
            let params = self.upload(&[(end - start) as u32, lines as u32, start as u32])?;
            self.run(&emit, &[&tab, &segs, &parts, &items, &counts, &offsets, &out_buf, &err, &params], end - start)?;
            if let Some(e) = EpochError::from_code(Self::read(&err, 1)[0]) {
                return Err(e);
            }
            let words = Self::read(&out_buf, lines * CHILD_WORDS);
            out.extend(words.chunks_exact(CHILD_WORDS).map(Child::from_words));
            start = end;
        }
        Ok(out)
    }
}
