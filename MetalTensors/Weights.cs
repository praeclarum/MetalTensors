using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;

using static MetalTensors.MetalHelpers;

namespace MetalTensors
{
    /// <summary>
    /// Responsible for storing weights when they are Purged from memory.
    /// Also, conveniently, can be serialized.
    /// </summary>
    public class Weights : Configurable
    {
        public ConcurrentDictionary<string, Memory<float>> Variables { get; } = new ConcurrentDictionary<string, Memory<float>> ();

        public Weights ()
        {
        }

        public void Read (string variableName, MPSVector vector, float initialValue)
        {
            if (Variables.TryGetValue (variableName, out var memory)) {
                vector.Init (memory);
            }
            else {
                vector.Fill (initialValue);
                Variables[variableName] = vector.ToSpan ().ToArray ();
            }
        }

        public void Read (string variableName, OptimizableVector vector, float initialValue)
        {
            if (Variables.TryGetValue (variableName, out var memory)) {
                vector.Value.Init (memory);
            }
            else {
                vector.Value.Fill (initialValue);
                Variables[variableName] = vector.Value.ToSpan ().ToArray ();
            }
        }

        public async Task ReadAsync (string variableName, OptimizableVector vector, WeightsInit initialValue, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            if (Variables.TryGetValue (variableName, out var memory)) {
                vector.Value.Init (memory);
            }
            else {
                var seed = (int)DateTime.Now.Ticks;
                await initialValue.InitWeightsAsync (vector.Value, seed, fanIn: fanIn, fanOut: fanOut, queue: queue).ConfigureAwait (false);
                Variables[variableName] = vector.Value.ToSpan ().ToArray ();
            }
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
        private bool disposed;

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is left uninitialized.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.GetByteSize ();
            VectorDescriptor = descriptor;
            Value = Vector (descriptor, device);
            Momentum = Vector (0.0f, descriptor, device);
            Velocity = Vector (0.0f, descriptor, device);
            ValuePointer = Value.Data.Contents;
        }

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is copied from the tensor.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, Tensor initialValue)
            : this (device, descriptor)
        {
            initialValue.Copy (Value.ToSpan (), device);
        }

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is filled with a constant.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
            : this (device, descriptor)
        {
            Value.Fill (initialValue);
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
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }


        /// <summary>
        /// Informs the GPU that the CPU has modified the vectors.
        /// </summary>
        public void MarkAsModifiedByCpu ()
        {
            Value.MarkAsModifiedByCpu ();
            Momentum.MarkAsModifiedByCpu ();
            Velocity.MarkAsModifiedByCpu ();
        }

        public bool IsFinite ()
        {
            return Value.IsFinite () && Momentum.IsFinite () && Velocity.IsFinite ();
        }

        public void Zero ()
        {
            Value.Zero ();
            Velocity.Zero ();
            Momentum.Zero ();
        }
    }
}
