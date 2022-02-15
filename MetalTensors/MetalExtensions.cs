using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using System.Diagnostics;
using System.Threading.Tasks;

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
                throw new Exception (error.ToString ());
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

        public static void DidModify (this MPSVector vector)
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

        public static bool IsValid (this MPSVector vector)
        {
            var ar = vector.ToArray ();
            for (var i = 0; i < ar.Length; i++) {
                var v = ar[i];
                if (float.IsNaN (v))
                    return false;
                if (float.IsInfinity (v))
                    return false;
                if (float.IsNegativeInfinity (v))
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
    }
}
