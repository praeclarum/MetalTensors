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
using ObjCRuntime;

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

        public static void Init (this MPSVector vector, Memory<float> values)
        {
            vector.MarkAsModifiedByCpu ();
            throw new NotImplementedException ();
        }

        public static Task UniformInitAsync (this MPSVector vector, float minimum, float maximum, int seed, bool downloadToCpu = true)
        {
            var descriptor = MPSMatrixRandomDistributionDescriptor.CreateUniform (minimum, maximum);
            return RandomInitAsync (vector, descriptor, seed, downloadToCpu);
        }

        public static Task NormalInitAsync (this MPSVector vector, float mean, float standardDeviation, int seed, bool downloadToCpu = true)
        {
            var descriptor = new MPSMatrixRandomDistributionDescriptor {
                DistributionType = MPSMatrixRandomDistribution.Normal,
                Mean = mean,
                StandardDeviation = standardDeviation,
                Maximum = float.PositiveInfinity,
                Minimum = float.NegativeInfinity,
            };
            //var d2 = Runtime.GetNSObject<MPSMatrixRandomDistributionDescriptor> (IntPtr_objc_msgSend_float_float (classMPSMatrixRandomDistributionDescriptor, selCreateNormal, mean, standardDeviation));
            //if (d2 == null)
            //    throw new Exception ($"Failed to create normal distribution descriptor");
            return RandomInitAsync (vector, descriptor, seed, downloadToCpu);
        }

        // MISSING XAMARIN BINDING
        //[DllImport ("/usr/lib/libobjc.dylib", EntryPoint = "objc_msgSend")]
        //public static extern IntPtr IntPtr_objc_msgSend_float_float (IntPtr receiver, IntPtr selector, float arg1, float arg2);
        //static readonly IntPtr selCreateNormal = Selector.GetHandle ("normalDistributionDescriptorWithMean:standardDeviation:");
        //static readonly IntPtr classMPSMatrixRandomDistributionDescriptor = Class.GetHandle ("MPSMatrixRandomDistributionDescriptor");

        public static Task RandomInitAsync (this MPSVector vector, MPSMatrixRandomDistributionDescriptor descriptor, int seed, bool downloadToCpu = true)
        {
            using var pool = new NSAutoreleasePool ();
            var device = vector.Device;
            var queue = device.CreateCommandQueue ();
            if (queue == null)
                throw new Exception ($"Failed to create command queue to generate random data");
            var command = MPSCommandBuffer.Create (queue);
            var random = new MPSMatrixRandomMtgp32 (device, vector.DataType, (nuint)seed, descriptor);
            random.EncodeToCommandBuffer (command, vector);
            if (downloadToCpu) vector.Synchronize (command);
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
            return vector.ToSpan ().ToArray ();
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

        /// <summary>
        /// Creates an uninitialized vector
        /// </summary>
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
            v.MarkAsModifiedByCpu ();
            return v;
        }
    }
}
