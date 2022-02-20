using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;

using static MetalTensors.MetalHelpers;

namespace MetalTensors
{
    /// <summary>
    /// Stores references to weights used by a layer.
    /// Responsible for reading and writing weights to archives.
    /// </summary>
    public class Weights : Configurable, IHasBuffers
    {
        public ConcurrentDictionary<string, Memory<float>> Values { get; } = new ConcurrentDictionary<string, Memory<float>> ();
        public ConcurrentDictionary<string, MPSVector> Vectors { get; } = new ConcurrentDictionary<string, MPSVector> ();

        public IWeightsDataSource? DataSource { get; set; }

        public Weights ()
        {
        }

        public void AddParameter (string parameterName, MPSVector vector, float initialValue)
        {
            if (Values.TryGetValue (parameterName, out var memory)) {
                vector.Init (memory);
            }
            else {
                vector.Fill (initialValue);
                Values[parameterName] = vector.ToSpan ().ToArray ();
                Vectors[parameterName] = vector;
            }
        }

        public void AddParameter (string parameterName, OptimizableVector vectors, float initialValue)
        {
            var vector = vectors.Value;
            if (Values.TryGetValue (parameterName, out var memory)) {
                vector.Init (memory);
            }
            else {
                vector.Fill (initialValue);
                Values[parameterName] = vector.ToSpan ().ToArray ();
                Vectors[parameterName] = vector;
            }
        }

        public async Task AddParameter (string parameterName, OptimizableVector vectors, WeightsInit initialValue, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            var vector = vectors.Value;
            if (Values.TryGetValue (parameterName, out var memory)) {
                vector.Init (memory);
            }
            else {
                var seed = (int)DateTime.Now.Ticks;
                await initialValue.InitWeightsAsync (vector, seed, fanIn: fanIn, fanOut: fanOut, queue: queue).ConfigureAwait (false);
                Values[parameterName] = vector.ToSpan ().ToArray ();
                Vectors[parameterName] = vector;
            }
        }

        public void ReadBuffers (ReadBuffer reader)
        {
            throw new NotImplementedException ();
        }

        public void WriteBuffers (WriteBuffer writer)
        {
            var vs = Vectors.ToArray ();
            if (vs.Length == 0)
                return;
            DataSource?.DownloadWeightsFromGpu ();
            foreach (var v in vs) {
                writer (v.Key, v.Value.ToSpan ());
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
