﻿using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using System.Diagnostics;
using System.Threading.Tasks;
using CoreGraphics;
using System.Collections.Generic;
using System.Linq;
using System.Buffers;

namespace MetalTensors
{
    public static class MPSImageExtensions
    {
        public static unsafe void CopyTo (this MPSImage source, Span<float> destination)
        {
            var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;
            var dtype = source.PixelFormat;
            switch (dtype) {
                case MTLPixelFormat.R32Float:
                case MTLPixelFormat.RG32Float:
                case MTLPixelFormat.RGBA32Float: {
                        fixed (float* dataPtr = destination) {
                            source.ReadBytes ((IntPtr)dataPtr, dataLayout, 0);
                        }
                    }
                    break;
                default:
                    throw new NotSupportedException ($"Cannot copy image with pixel format {dtype}");
            }
        }

        public static void CopyTo (this MPSImage source, MPSImage destination)
        {
            //var n = source.FeatureChannels * source.Width * source.Height;
            //var pool = ArrayPool<float>.Shared;
            //var a = pool.Rent ((int)n);
            //try {
            //    unsafe {
            //        fixed (float* p = a) {
            //            if (source.PixelFormat == destination.PixelFormat) {
            //                source.ReadBytes ((IntPtr)p, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
            //            }
            //            else {
            //                using var view = source.Texture.CreateTextureView (destination.PixelFormat);
            //                if (view == null)
            //                    throw new Exception ($"Cannot create {destination.PixelFormat} view");
            //                view.GetBytes ((IntPtr)p, view.Width * destination.PixelSize, MTLRegion.Create2D (0, 0, view.Width, view.Height), 0);
            //            }
            //            destination.WriteBytes ((IntPtr)p, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
            //        }
            //    }
            //}
            //finally {
            //    pool.Return (a);
            //}
            //using var nspool = new NSAutoreleasePool ();
            using var queue = destination.Device.CreateCommandQueue ();
            if (queue is null)
                throw new Exception ($"Failed to create queue for image copy to");
            CopyToAsync (source, destination, queue).Wait ();
        }

        public static Task CopyToAsync (this MPSImage source, MPSImage destination, IMTLCommandQueue queue)
        {
            using var nspool = new NSAutoreleasePool ();
            var tcs = new TaskCompletionSource<object?> ();
            using var commands = MPSCommandBuffer.Create (queue);
            commands.AddCompletedHandler (b => {
                //using var nspool = new NSAutoreleasePool ();
                if (b.Error is NSError e) {
                    try {
                        b.Dispose ();
                    }
                    catch (Exception ex) {
                        Console.WriteLine ($"Failed to dispose command buffer: {ex}");
                    }
                    tcs.TrySetException (new NSErrorException (e));
                }
                else {
                    try {
                        b.Dispose ();
                    }
                    catch (Exception ex) {
                        Console.WriteLine ($"Failed to dispose command buffer: {ex}");
                    }
                    tcs.TrySetResult (null);
                }
            });
            EncodeToCommandBuffer (source, destination, commands);
            commands.Commit ();
            return tcs.Task;
        }

        public static void Dispose (this MPSImage[]? images)
        {
            if (images != null) {
                foreach (var imgs in images) {
                    imgs.Dispose ();
                }
            }
        }

        public static void EncodeToCommandBuffer (this MPSImage source, MPSImage destination, MPSCommandBuffer commands)
        {
            destination.Synchronize (commands);
            using var blit = commands.BlitCommandEncoder;
            blit.Copy (source.Texture, destination.Texture);
            blit.EndEncoding ();
        }

        public static unsafe void Fill (this MPSImage image, float constant)
        {
            var dtype = image.PixelFormat;
            var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;

            switch (dtype) {
                case MTLPixelFormat.R32Float:
                case MTLPixelFormat.RG32Float:
                case MTLPixelFormat.RGBA32Float: {
                        var len = (int)(image.Height * image.Width * image.FeatureChannels);
                        Span<float> dataSpan = len < 1024 ?
                            stackalloc float[len] :
                            new float[len];
                        for (var i = 0; i < len; i++) {
                            dataSpan[i] = constant;
                        }
                        fixed (float* dataPtr = dataSpan) {
                            image.WriteBytes ((IntPtr)dataPtr, dataLayout, 0);
                        }
                    }
                    break;
                default:
                    throw new NotSupportedException ($"Cannot fill images with pixel format {dtype}");
            }
        }

        public static MPSImage Filter (this MPSImage image, MPSCnnNeuron neuron, IMTLDevice? device = null)
        {
            var dev = device.Current ();
            using var queue = dev.CreateCommandQueue ();
            if (queue is null)
                throw new Exception ($"Failed to create queue to filter image");
            using var commands = MPSCommandBuffer.Create (queue);
            var desc = neuron.GetDestinationImageDescriptor (NSArray<MPSImage>.FromNSObjects (image)!, null);
            var result = new MPSImage (dev, desc);
            neuron.EncodeToCommandBuffer (commands, image, result);
            result.Synchronize (commands);
            commands.Commit ();
            commands.WaitUntilCompleted ();
            return result;
        }

        public static MPSImage Linear (this MPSImage image, float a, float b, IMTLDevice? device = null)
        {
            var dev = device.Current ();
            using var neuron = new MPSCnnNeuronLinear (dev, a, b);
            return Filter (image, neuron, dev);
        }

        public static float ReduceMeanValueAndDispose (this NSArray<MPSImage>? images)
        {
            if (images == null)
                return 0.0f;
            var sum = 0.0f;
            var n = images.Count;
            for (nuint i = 0; i < n; i++) {
                using var image = images.GetItem<MPSImage> (i);
                sum += image.ReduceMeanValue ();
            }
            images.Dispose ();
            if (n > 0)
                return sum / n;
            return 0.0f;
        }

        public static float ReduceMeanValue (this MPSImage image)
        {
            var n = image.FeatureChannels * image.Width * image.Height;
            if (n == 0)
                return 0.0f;
            Span<float> floats = n <= 64 ?
                stackalloc float[(int)n] :
                new float[n];
            unsafe {
                fixed (float* p = floats) {
                    image.ReadBytes ((IntPtr)p, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                }
            }
            if (n > 0) {
                var sum = 0.0f;
                for (nuint i = 0; i < n; i++) {
                    sum += floats[(int)i];
                }
                return sum / n;
            }
            return 0.0f;
        }

        public static CGImage ToCGImage (this MPSImage image, int numComponents = 3)
        {
            if (image == null) {
                throw new ArgumentNullException (nameof (image));
            }

            var width = (nint)image.Width;
            var height = (nint)image.Height;
            var imagePixelBytes = (nint)image.PixelSize;

            var disposeByteTexture = false;
            var byteTexture = image.Texture;
#if NET6_0_OR_GREATER
            var storageMode = byteTexture.StorageMode;
#else
            var storageMode = byteTexture.GetStorageMode ();
#endif
            if (storageMode == MTLStorageMode.Private)
                throw new InvalidOperationException ($"Cannot create images from private textures");
            nint bitsPerComponent = 8;
            var outputNumComponents = 4;
            CGBitmapFlags bitmapFlags;
            switch (byteTexture.PixelFormat) {
                case MTLPixelFormat.BGRA8Unorm_sRGB:
                    bitmapFlags = numComponents > 3 ? CGBitmapFlags.First : CGBitmapFlags.NoneSkipFirst;
                    bitmapFlags |= CGBitmapFlags.ByteOrder32Little;
                    break;
                case MTLPixelFormat.RGBA8Unorm:
                    bitmapFlags = numComponents > 3 ? CGBitmapFlags.Last : CGBitmapFlags.NoneSkipLast;
                    bitmapFlags |= CGBitmapFlags.ByteOrder32Big;
                    break;
                case MTLPixelFormat.RGBA32Float:
                    bitsPerComponent = 32;
                    bitmapFlags = numComponents > 3 ? CGBitmapFlags.Last : CGBitmapFlags.NoneSkipLast;
                    bitmapFlags |= CGBitmapFlags.FloatComponents | CGBitmapFlags.ByteOrder32Little;
                    break;
                case MTLPixelFormat.R32Float:
                    bitsPerComponent = 32;
                    bitmapFlags = CGBitmapFlags.None | CGBitmapFlags.FloatComponents | CGBitmapFlags.ByteOrder32Little;
                    outputNumComponents = 1;
                    //{
                    //    bitsPerComponent = 32;
                    //    bitmapFlags = numComponents > 3 ? CGBitmapFlags.Last : CGBitmapFlags.NoneSkipLast;
                    //    bitmapFlags |= CGBitmapFlags.FloatComponents | CGBitmapFlags.ByteOrder32Little;
                    //    var v = byteTexture.Create (
                    //        MTLPixelFormat.R32Float,
                    //        MTLTextureType.k2D,
                    //        new NSRange (0, 1),
                    //        new NSRange (0, 1),
                    //        new MTLTextureSwizzleChannels {
                    //            Red = MTLTextureSwizzle.Red,
                    //            Green = MTLTextureSwizzle.Red,
                    //            Blue = MTLTextureSwizzle.Red,
                    //            Alpha = MTLTextureSwizzle.One,
                    //        });
                    //    if (v == null)
                    //        throw new Exception ($"Failed to convert tensor data to bytes");
                    //    byteTexture = v;
                    //    disposeByteTexture = true;
                    //}
                    break;
                default:
                    throw new NotSupportedException ($"Cannot process images with pixel format {image.PixelFormat} ({imagePixelBytes} bytes per pixel, feature channel format {image.FeatureChannelFormat})");
            }

            try {                
                var outputBytesPerRow = (width * bitsPerComponent * outputNumComponents) / 8;
                var cs = outputNumComponents == 1 ? CGColorSpace.CreateGenericGray() : CGColorSpace.CreateSrgb ();
                using var c = new CGBitmapContext (null, width, height, bitsPerComponent, outputBytesPerRow, cs, bitmapFlags);
                byteTexture.GetBytes (c.Data, (nuint)c.BytesPerRow, MTLRegion.Create2D (0, 0, width, height), 0);
                var cgimage = c.ToImage ();
                if (cgimage == null)
                    throw new Exception ($"Failed to create core graphics image");
                return cgimage;
            }
            finally {
                if (disposeByteTexture) {
                    byteTexture.Dispose ();
                }
            }
        }

        //[DllImport("__Internal")]
        //static extern unsafe void vDSP_vfixru8 (float* __A, int __IA, byte* __C, nint __IC, nint __N);
    }

    public static partial class MetalHelpers
    {
        public static MPSImage CreateUninitializedImage (int[] shape)
        {
            var imageTensor = shape.Length switch {
                0 => new Tensors.MPSImageTensor (height: 1, width: 1, featureChannels: 1),
                1 => new Tensors.MPSImageTensor (height: 1, width: 1, featureChannels: shape[0]),
                2 => new Tensors.MPSImageTensor (height: 1, width: shape[0], featureChannels: shape[1]),
                3 => new Tensors.MPSImageTensor (height: shape[0], width: shape[1], featureChannels: shape[2]),
                var l => throw new InvalidOperationException ($"Cannot get image for constant data with {l} element shape"),
            };
            var image = imageTensor.MetalImage;
            return image;
        }

        public static MPSImage CreateConstantImage (int[] shape, float constantValue)
        {
            var image = CreateUninitializedImage (shape);
            image.Fill (constantValue);
#if DEBUG
            var data = new Tensors.MPSImageTensor (image).ToArray (((IMTLDevice?)null).Current ());
            Debug.Assert (data[0] == constantValue);
#endif
            return image;
        }
    }
}
