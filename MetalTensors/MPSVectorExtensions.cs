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
    public static class MPSVectorExtensions
    {
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
                switch (cs.Status) {
                    case MTLCommandBufferStatus.Error:
                        tcs.TrySetException (new NSErrorException (cs.Error));
                        break;
                    case MTLCommandBufferStatus.Completed:
                        tcs.TrySetResult (true);
                        break;
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
            (int)vector.Length * vector.DataType.GetByteSize ();

        public static int GetByteSize (this MPSVectorDescriptor descriptor) =>
            (int)descriptor.Length * descriptor.DataType.GetByteSize ();

    }

    public static partial class MetalHelpers
    {
        public static MPSVectorDescriptor VectorDescriptor (int length, MPSDataType dataType = MPSDataType.Float32) =>
            MPSVectorDescriptor.Create ((nuint)length, dataType);

        public static MPSVector Vector (int length, MPSDataType dataType = MPSDataType.Float32, IMTLDevice? device = null) =>
            Vector (VectorDescriptor (length, dataType), device);

        public static MPSVector Vector (MPSVectorDescriptor descriptor, IMTLDevice? device = null)
        {
            if (descriptor.Length <= 0)
                throw new ArgumentOutOfRangeException (nameof (descriptor), "Vector lengths must be > 0");
            var dev = device.Current ();
            var v = new MPSVector (dev, descriptor);
            if (v.Data == null)
                throw new Exception ($"Failed to create vector with length {descriptor.Length}");
            return v;
        }

        public static MPSVector Vector (float initialValue, MPSVectorDescriptor descriptor, IMTLDevice? device = null)
        {
            var v = Vector (descriptor, device);
            v.Fill (initialValue);
            return v;
        }

        public static MPSVector Vector (float initialValue, int length, MPSDataType dataType = MPSDataType.Float32, IMTLDevice? device = null)
        {
            var v = Vector (VectorDescriptor (length, dataType), device);
            v.Fill (initialValue);
            return v;
        }

        public static MPSVector Vector (Tensor initialValue, MPSVectorDescriptor descriptor, IMTLDevice? device = null)
        {
            if (descriptor.Length <= 0)
                throw new ArgumentOutOfRangeException (nameof (descriptor), "Vector lengths must be > 0");
            var dev = device.Current ();
            var v = new MPSVector (dev, descriptor);
            if (v.Data == null)
                throw new Exception ($"Failed to create vector with length {descriptor.Length}");
            initialValue.Copy (v.ToSpan (), dev);
            return v;
        }
    }
}
