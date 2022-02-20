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
            initialValue.CopyTo (v.ToSpan (), device);
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

        public static void DownloadFromGpu (this MPSVector vector, IMTLCommandBuffer commandBuffer)
        {
            vector.Synchronize (commandBuffer);
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

        public static void Init (this MPSVector vector, ReadOnlySpan<float> values)
        {
            var dest = vector.ToSpan ();
            values.CopyTo (dest);
            vector.MarkAsModifiedByCpu ();
        }

        public static Task UniformInitAsync (this MPSVector vector, float minimum, float maximum, int seed, bool downloadToCpu = true, IMTLCommandQueue? queue = null)
        {
            var descriptor = MPSMatrixRandomDistributionDescriptor.CreateUniform (minimum, maximum);
            return RandomInitAsync (vector, descriptor, seed, downloadToCpu, queue);
        }

        public static Task NormalInitAsync (this MPSVector vector, float mean, float standardDeviation, int seed, bool downloadToCpu = true, IMTLCommandQueue? queue = null)
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
            return RandomInitAsync (vector, descriptor, seed, downloadToCpu, queue);
        }

        // MISSING XAMARIN BINDING
        //[DllImport ("/usr/lib/libobjc.dylib", EntryPoint = "objc_msgSend")]
        //public static extern IntPtr IntPtr_objc_msgSend_float_float (IntPtr receiver, IntPtr selector, float arg1, float arg2);
        //static readonly IntPtr selCreateNormal = Selector.GetHandle ("normalDistributionDescriptorWithMean:standardDeviation:");
        //static readonly IntPtr classMPSMatrixRandomDistributionDescriptor = Class.GetHandle ("MPSMatrixRandomDistributionDescriptor");

        public static Task RandomInitAsync (this MPSVector vector, MPSMatrixRandomDistributionDescriptor descriptor, int seed, bool downloadToCpu = true, IMTLCommandQueue? queue = null)
        {
            using var pool = new NSAutoreleasePool ();
            var device = vector.Device;
            if (queue == null) {
                queue = device.CreateCommandQueue ();
                if (queue == null)
                    throw new Exception ($"Failed to create command queue to generate random data");
            }
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
            initialValue.CopyTo (v.ToSpan (), dev);
            v.MarkAsModifiedByCpu ();
            return v;
        }
    }

    /// <summary>
    /// Just as easy collection of MPSVectors to keep track of the optimization state
    /// </summary>
    public sealed class OptimizableVector : IDisposable
    {
        public readonly int VectorLength;
        public readonly int VectorByteSize;
        public readonly MPSVectorDescriptor VectorDescriptor;
        public readonly MPSVector Value;
        public readonly MPSVector Momentum;
        public readonly MPSVector Velocity;
        public readonly IntPtr ValuePointer;
        bool disposed;

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is left uninitialized.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.GetByteSize ();
            VectorDescriptor = descriptor;
            Value = MetalHelpers.Vector (descriptor, device);
            Momentum = MetalHelpers.Vector (0.0f, descriptor, device);
            Velocity = MetalHelpers.Vector (0.0f, descriptor, device);
            ValuePointer = Value.Data.Contents;
        }

        public void Dispose ()
        {
            if (!disposed) {
                disposed = true;
                Velocity.Dispose ();
                Momentum.Dispose ();
                Value.Dispose ();
            }
        }

        /// <summary>
        /// Flush the underlying MTLBuffer from the device's caches, and invalidate any CPU caches if needed.
        /// This will call[id < MTLBlitEncoder > synchronizeResource: ] on the vector's MTLBuffer, if any.
        /// This is necessary for all MTLStorageModeManaged resources.For other resources, including temporary
        /// resources (these are all MTLStorageModePrivate), and buffers that have not yet been allocated, nothing is done.
        /// It is more efficient to use this method than to attempt to do this yourself with the data property.
        /// </summary>
        /// <param name="commandBuffer"></param>
        public void DownloadFromGpu (IMTLCommandBuffer commandBuffer)
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (OptimizableVector));
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }


        /// <summary>
        /// Informs the GPU that the CPU has modified the vectors.
        /// </summary>
        public void MarkAsModifiedByCpu ()
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (OptimizableVector));
            Value.MarkAsModifiedByCpu ();
            Momentum.MarkAsModifiedByCpu ();
            Velocity.MarkAsModifiedByCpu ();
        }

        public bool IsFinite ()
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (OptimizableVector));
            return Value.IsFinite () && Momentum.IsFinite () && Velocity.IsFinite ();
        }

        public void Zero ()
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (OptimizableVector));
            Value.Zero ();
            Velocity.Zero ();
            Momentum.Zero ();
        }
    }
}
