using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using System.Diagnostics;
using System.Threading.Tasks;
using CoreGraphics;

namespace MetalTensors
{
    public static class MetalExtensions
    {
        static IMTLDevice? currentDevice;

        public static IMTLDevice Current (this IMTLDevice? device)
        {
            if (device != null)
                return device;
            if (currentDevice != null)
                return currentDevice;
            var def = Default (null);
            var old = Interlocked.CompareExchange (ref currentDevice, def, null);
            if (old == null) {
                Console.WriteLine ("DEVICE = " + def.Name);
            }
            return currentDevice;
        }

        static IMTLDevice Default (this IMTLDevice? device)
        {
            if (device != null)
                return device;
            var def = MPSKernel.GetPreferredDevice (MPSDeviceOptions.Default);
            if (def == null || def.Name.Contains ("iOS simulator")) {
                throw new NotSupportedException ("Metal is not supported on this device");
            }
            return def;
        }

        public static void ValidateNoError (this NSError? error)
        {
            if (error != null) {
                throw new NSErrorException (error);
            }
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

        public static MPSImage Filter (this MPSImage image, MPSCnnNeuron neuron, IMTLDevice ? device = null)
        {
            var dev = device.Current ();
            using var queue = dev.CreateCommandQueue ();
            if (queue is null)
                throw new Exception ($"Failed to create queue to filter image");
            using var commands = MPSCommandBuffer.Create (queue);
            var desc = neuron.GetDestinationImageDescriptor (NSArray<MPSImage>.FromNSObjects (image), null);
            var result = new MPSImage (dev, desc);
            neuron.EncodeToCommandBuffer (commands, image, result);
            result.Synchronize (commands);
            commands.Commit ();
            commands.WaitUntilCompleted ();
            return result;
        }

        public static MPSImage Linear (this MPSImage image, float a, float b, IMTLDevice ? device = null)
        {
            var dev = device.Current ();
            using var neuron = new MPSCnnNeuronLinear (dev, a, b);
            return Filter (image, neuron, dev);
        }

        public static MPSVectorDescriptor VectorDescriptor (int length, MPSDataType dataType = MPSDataType.Float32) =>
            MPSVectorDescriptor.Create ((nuint)length, dataType);

        public static MPSVector Vector (IMTLDevice device, MPSVectorDescriptor descriptor)
        {
            if (descriptor.Length <= 0)
                throw new ArgumentOutOfRangeException (nameof (descriptor), "Vector lengths must be > 0");
            var v = new MPSVector (device, descriptor);
            if (v.Data == null)
                throw new Exception ($"Failed to create vector with length {descriptor.Length}");
            return v;
        }

        public static MPSVector Vector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
        {
            var v = Vector (device, descriptor);
            Fill (v, initialValue);
            return v;
        }

        public static MPSVector Vector (IMTLDevice device, MPSVectorDescriptor descriptor, Tensor initialValue)
        {
            if (descriptor.Length <= 0)
                throw new ArgumentOutOfRangeException (nameof (descriptor), "Vector lengths must be > 0");

            var v = new MPSVector (device, descriptor);
            if (v.Data == null)
                throw new Exception ($"Failed to create vector with length {descriptor.Length}");
            initialValue.Copy (v.ToSpan (), device);
            return v;
        }


        /// <summary>
        /// Informs the GPU that the CPU has modified the vector.
        /// </summary>
        public static void MarkAsModifiedByCpu (this MPSVector vector)
        {
            var data = vector.Data;
            data.DidModify (new NSRange (0, (nint)data.Length));
        }

        public static unsafe Span<float> ToSpan (this MPSVector vector)
        {
            var vspan = new Span<float> ((float*)vector.Data.Contents, (int)vector.Length);
            return vspan;
        }

        public static void SetElements (this MPSVector vector, Span<float> elements)
        {
            elements.CopyTo (vector.ToSpan ());
        }

        public static void Zero (this MPSVector vector)
        {
            Fill (vector, 0);
        }

        public static void Fill (this MPSVector vector, float constant)
        {
            var vectorByteSize = GetByteSize (vector);
            if (vectorByteSize > 0) {
                unsafe {
                    var biasInitPtr = stackalloc float[4];
                    biasInitPtr[0] = constant;
                    biasInitPtr[1] = constant;
                    biasInitPtr[2] = constant;
                    biasInitPtr[3] = constant;
                    memset_pattern16 (vector.Data.Contents, (IntPtr)biasInitPtr, vectorByteSize);
                }
            }
            vector.MarkAsModifiedByCpu ();
        }

        [System.Runtime.InteropServices.DllImport (@"__Internal", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
        static extern void memset_pattern16 (IntPtr b, IntPtr pattern16, nint len);

        public static Task UniformInitAsync (this MPSVector vector, float minimum, float maximum, int seed, IMTLDevice device)
        {
            var descriptor = MPSMatrixRandomDistributionDescriptor.CreateUniform (minimum, maximum);
            return RandomInitAsync (vector, descriptor, seed, device);
        }

        public static Task NormalInitAsync (this MPSVector vector, float mean, float standardDeviation, int seed, IMTLDevice device)
        {
            var descriptor = new MPSMatrixRandomDistributionDescriptor {
                DistributionType = MPSMatrixRandomDistribution.Normal,
                Mean = mean,
                StandardDeviation = standardDeviation
            };
            return RandomInitAsync (vector, descriptor, seed, device);
        }

        public static Task RandomInitAsync (this MPSVector vector, MPSMatrixRandomDistributionDescriptor descriptor, int seed, IMTLDevice device)
        {
            using var pool = new NSAutoreleasePool ();
            var queue = device.CreateCommandQueue ();
            if (queue == null)
                throw new Exception ($"Failed to create command queue to generate random data");
            var command = MPSCommandBuffer.Create (queue);
            var random = new MPSMatrixRandomMtgp32 (device, vector.DataType, (nuint)seed, descriptor);
            random.EncodeToCommandBuffer (command, vector);
            random.Synchronize (command);
            var tcs = new TaskCompletionSource<bool> ();
            command.AddCompletedHandler (cs => {
                if (cs.Status == MTLCommandBufferStatus.Error) {
                    tcs.TrySetException (new NSErrorException (cs.Error));
                }
                else if (cs.Status == MTLCommandBufferStatus.Completed) {
                    tcs.TrySetResult (true);
                }
            });
            command.Commit ();
            return tcs.Task;
        }

        public static float[] ToArray (this MPSVector vector)
        {
            var ar = new float[vector.Length];
            Marshal.Copy (vector.Data.Contents, ar, 0, ar.Length);
            return ar;
        }

        public static bool IsFinite (this MPSVector vector)
        {
            var ar = vector.ToSpan ();
            for (var i = 0; i < ar.Length; i++) {
                if (!float.IsFinite (ar[i]))
                    return false;
            }
            return true;
        }

        public static int GetByteSize (this MPSVector vector) =>
            (int)vector.Length * GetByteSize (vector.DataType);

        public static int GetByteSize (this MPSVectorDescriptor descriptor) =>
            (int)descriptor.Length * GetByteSize (descriptor.DataType);

        public static int GetByteSize (this MPSDataType dataType) =>
            dataType switch
            {
                MPSDataType.Unorm8 => 1,
                MPSDataType.Float32 => 4,
                var x => throw new NotSupportedException ($"Cannot get size of {x}")
            };
#if __IOS__
        public static void DidModify (this IMTLBuffer buffer, NSRange range)
        {
        }
#endif

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

        public static CGImage ToCGImage (this MPSImage image, int numComponents = 3)
        {
            if (image == null) {
                throw new ArgumentNullException (nameof (image));
            }

            var width = (nint)image.Width;
            var height = (nint)image.Height;
            var imagePixelBytes = (nint)image.PixelSize;

            var disposeByteTexture = false;
            var bitmapFlags = numComponents > 3 ? CGBitmapFlags.Last : CGBitmapFlags.NoneSkipLast;
            var byteTexture = image.Texture;
            nint bitsPerComponent = 8;
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
                    bitmapFlags = CGBitmapFlags.NoneSkipLast | CGBitmapFlags.FloatComponents | CGBitmapFlags.ByteOrder32Little;
                    var v = byteTexture.Create (
                        MTLPixelFormat.R32Float,
                        MTLTextureType.k2D,
                        new NSRange (0, 1),
                        new NSRange (0, 1),
                        new MTLTextureSwizzleChannels {
                            Red = MTLTextureSwizzle.Red,
                            Green = MTLTextureSwizzle.Red,
                            Blue = MTLTextureSwizzle.Red,
                            Alpha = MTLTextureSwizzle.One,
                        });
                    if (v == null)
                        throw new Exception ($"Failed to convert tensor data to bytes");
                    byteTexture = v;
                    disposeByteTexture = true;
                    break;
                default:
                    //var v = byteTexture.CreateTextureView (MTLPixelFormat.BGRA8Unorm_sRGB);
                    //if (v == null)
                    //    throw new Exception ($"Failed to convert tensor data to bytes");
                    //byteTexture = v;
                    //disposeByteTexture = true;
                    throw new NotSupportedException ($"Cannot process images with pixel format {image.PixelFormat} ({imagePixelBytes} bytes per pixel, feature channel format {image.FeatureChannelFormat})");

            }

            try {
                using var cs = CGColorSpace.CreateSrgb ();
                var bytesPerRow = (width * bitsPerComponent * 4) / 8;
                using var c = new CGBitmapContext (null, width, height, bitsPerComponent, bytesPerRow, cs, bitmapFlags);
                image.Texture.GetBytes (c.Data, (nuint)c.BytesPerRow, MTLRegion.Create2D (0, 0, width, height), 0);
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
}
